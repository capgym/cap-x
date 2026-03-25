import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki import init_pyroki
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from capx.integrations.vision.sam2 import init_sam2
from capx.integrations.franka.common import (
    apply_tcp_offset,
    build_segmentation_map_from_sam2,
    compute_bbox_indices,
    draw_boxes,
    save_segmentation_debug,
    select_instance_from_box,
)
from capx.integrations.vision.sam3 import init_sam3, visualize_sam3_results
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)


# ------------------------------- Control API ------------------------------
class FrankaControlSpillWipeApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] = [0.0, 0.0, -0.107],
        use_sam3: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        print("init franka control api")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg = None
        self.use_sam3 = use_sam3
        self.debug = debug
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            print("init sam3 seg fn")
        else:
            self.owl_vit_det_fn = init_owlvit(device="cuda")
            print("init owlvit det fn")
            self.sam2_seg_fn = init_sam2()
            print("init sam2 seg fn")

    def functions(self) -> dict[str, Any]:
        fns = {
            "get_object_pose": self.get_object_pose,
            "goto_pose": self.goto_pose,
        }
        return fns

    def _get_segmentation_map(
        self, obs: dict[str, Any], rgb: np.ndarray, box: list[float] = None
    ) -> np.ndarray:
        return build_segmentation_map_from_sam2(
            self.sam2_seg_fn, rgb, obs["robot0_robotview"]["images"], box=box
        )

    def _save_segmentation_debug(self, segmentation: np.ndarray, path: pathlib.Path) -> None:
        save_segmentation_debug(segmentation, path)

    def _compute_bbox_indices(
        self, box: list[float], shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        return compute_bbox_indices(box, shape)

    def _select_instance_from_box(
        self, segmentation: np.ndarray, box: list[float]
    ) -> tuple[int, np.ndarray]:
        return select_instance_from_box(segmentation, box)

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.
            return_bbox_extent:  Whether to return the extent of the oriented bounding box (oriented by quaternion_wxyz). Default is False.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) XYZ in meters (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        start_time = time.time()
        obs = self._env.get_observation()

        rbg_imgs = obs_get_rgb(obs)
        assert len(rbg_imgs.keys()) > 0, "No RGB images in obs"

        rgb = list(rbg_imgs.values())[0]

        depth = obs["robot0_robotview"]["images"]["depth"]

        # Debug image saves TODO: Remove this eventually, or add a debug mode branch
        # save depth image with colormap
        depth_img = depth_to_rgb(depth[:, :, 0])
        depth_img_out = Image.fromarray(depth_img)
        depth_img_out.save("depth_image.jpg")

        binary_map_nan_is_zero = (~np.isnan(depth[:, :, 0])).astype(int)

        if self.use_sam3:
            results = self.sam3_seg_fn(rgb, text_prompt=object_name)
            if len(results) == 0:
                raise ValueError("No sam3 detections")
            scores = [result["score"] for result in results]

            box = results[np.argmax(scores)]["box"]
            mask = results[np.argmax(scores)]["mask"]

            if self.debug:
                visualize_sam3_results(
                    Image.fromarray(rgb),
                    object_name,
                    results,
                    output_dir=pathlib.Path("."),
                    show=False,
                )
            idxs = np.where(mask.flatten() & binary_map_nan_is_zero.flatten().astype(bool))
        else:
            dets = self.owl_vit_det_fn(rgb, texts=[[object_name]])

            if len(dets) == 0:
                raise ValueError("No detections; environment constraints or model mismatch")

            boxes = [d["box"] for d in dets]
            labels = [d["label"] for d in dets]
            scores = [d["score"] for d in dets]

            box = boxes[np.argmax(scores)]

            if self.debug:
                img_out = _draw_boxes(
                    rgb, [box], [labels[np.argmax(scores)]], scores=[scores[np.argmax(scores)]]
                )
                out_file = pathlib.Path("owlvit_det.jpg")
                img_out.save(out_file)
                assert out_file.exists() and out_file.stat().st_size > 0

            # save segmentation image
            segmentation = self._get_segmentation_map(obs, rgb, box=box)
            if self.debug:
                self._save_segmentation_debug(segmentation, pathlib.Path("segmentation_image.jpg"))

            queried_instance_idx, seg_crop = self._select_instance_from_box(segmentation, box)
            if self.debug:
                self._save_segmentation_debug(seg_crop, pathlib.Path("seg_crop_image.jpg"))

            # idxs = np.where(segmentation.flatten() == queried_instance_idx) # Old assumes there are no Nans in the depth map (happens in real ZED returns)
            idxs = np.where(
                segmentation.flatten()[binary_map_nan_is_zero.flatten().astype(bool)]
                == queried_instance_idx
            )

        # points = depth_to_pointcloud(depth[:, :, 0], obs["robot0_robotview"]["intrinsics"])[idxs]
        points, color = depth_color_to_pointcloud(
            depth[:, :, 0], rgb, obs["robot0_robotview"]["intrinsics"]
        )

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points[idxs])

        obb = o3d_points.get_oriented_bounding_box()

        # Exposing these to the low level environment for viser

        self._env.cube_points = points[idxs]
        self._env.cube_color = color[idxs]

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs["robot0_robotview"]["pose"][3:]),
            translation=obs["robot0_robotview"]["pose"][:3],
        )
        obb_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R), translation=obb.center
        )
        obb_tf_world = cam_extr_tf @ obb_tf

        self._env.cube_center = obb_tf_world.translation()
        self._env.cube_rot = obb_tf_world.rotation().as_matrix()

        x1, y1, x2, y2 = box

        # Camera intrinsics and extrinsics
        K = obs["robot0_robotview"]["intrinsics"]  # (3,3)
        pose_mat = obs["robot0_robotview"]["pose_mat"]  # (4,4)

        # Camera pose in world: world to camera
        R_cw = pose_mat[:3, :3]
        t_cw = pose_mat[:3, 3]

        # Function to backproject a pixel to world point on table plane z=0
        def pixel_to_world(u, v):
            # Pixel direction in camera coordinates
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            dir_cam /= np.linalg.norm(dir_cam)
            # Direction in world coordinates
            dir_world = R_cw @ dir_cam
            # Solve for t such that (t_cw + t*dir_world)[2] = 0
            t = -t_cw[2] / dir_world[2]
            point_world = t_cw + t * dir_world
            return point_world

        # Compute world coordinates of bounding box corners
        p1 = pixel_to_world(x1, y1)
        p2 = pixel_to_world(x2, y1)
        p3 = pixel_to_world(x2, y2)
        p4 = pixel_to_world(x1, y2)

        # Determine min/max x and y based on the bounding box corners
        xs = np.array([p1[0], p2[0], p3[0], p4[0]])
        ys = np.array([p1[1], p2[1], p3[1], p4[1]])
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        extent = np.array([xmax - xmin, ymax - ymin, 0.001])

        print(f"get_object_pose in {time.time() - start_time} seconds")
        if return_bbox_extent:
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], extent
        else:
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], None

    def goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if self.cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def breakpoint_code_block(self) -> None:
        """Call this function to mark a significant checkpoint where you want to evaluate progress and potentially regenerate the remaining code.

        Args:
            None
        """
        return None


def _draw_boxes(
    rgb: np.ndarray, boxes: list[list[float]], labels: list[str], scores: list[float] | None = None
) -> Image.Image:
    return draw_boxes(rgb, boxes, labels, scores)
