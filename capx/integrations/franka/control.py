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

# from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from capx.integrations.vision.sam2 import init_sam2
from capx.integrations.vision.sam3 import init_sam3, visualize_sam3_results
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb
from capx.integrations.franka.common import (
    apply_tcp_offset,
    build_segmentation_map_from_sam2,
    close_gripper as _close_gripper,
    compute_bbox_indices,
    draw_boxes,
    open_gripper as _open_gripper,
    save_segmentation_debug,
    select_instance_from_box,
)
from capx.utils.visualization_utils import (
    draw_oriented_bounding_box,
    overlay_segmentation_masks,
)


# ------------------------------- Control API ------------------------------
class FrankaControlApi(ApiBase):
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
        real: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        print("init franka control api")
        self.grasp_net_plan_fn = (
            init_contact_graspnet()
        )  # TODO: refactor this and use registered api instead
        print("init grasp net plan fn")
        self.use_sam3 = use_sam3
        self.debug = debug
        self.real = real
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            print("init sam3 seg fn")
        else:
            self.owl_vit_det_fn = init_owlvit(
                device="cuda"
            )  # TODO: refactor this and use registered api instead
            print("init owlvit det fn")
            self.sam2_seg_fn = init_sam2()
            print("init sam2 seg fn")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg = None

    def functions(self) -> dict[str, Any]:
        fns = {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            # "home_pose": self.home_pose,
            # "get_observation": self.get_observation,
        }
        if not self.real: # Only include home pose in simulation
            fns["home_pose"] = self.home_pose
        return fns

    # def get_observation(self) -> dict[str, Any]:
    #     """Get the observation of the environment.
    #     Returns:
    #         observation:
    #             A dictionary containing the observation of the environment.
    #             The dictionary contains the following keys:
    #             - ["robot_cartesian_pos"]: Current cartesian position of the robot as a numpy array of shape (7,), dtype float64.
    #                 - [0:3]: Position of the robot in the world frame.
    #                 - [3:7]: Quaternion of the robot in the world frame.
    #                 - [7]: Gripper position in metric units.
    #             - ["robot_joint_pos"]: Current joint position of the robot as a numpy array of shape (7,), dtype float64.
    #                 - [0:7]: Joint positions of the robot.
    #                 - [7]: Gripper position in metric units.
    #     """
    #     return self._env.get_observation()

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
        self._log_step("get_object_pose", f"Detecting object **'{object_name}'** …")
        start_time = time.time()
        obs = self._env.get_observation()

        rbg_imgs = obs_get_rgb(obs)
        assert len(rbg_imgs.keys()) > 0, "No RGB images in obs"

        rgb = list(rbg_imgs.values())[0]
        self._log_step_update(images=rgb)

        depth = obs["robot0_robotview"]["images"]["depth"]

        # Debug image saves TODO: Remove this eventually, or add a debug mode branch
        # save depth image with colormap
        depth_img = depth_to_rgb(depth[:, :, 0])
        depth_img_out = Image.fromarray(depth_img)
        depth_img_out.save("depth_image.jpg")

        binary_map_nan_is_zero = (~np.isnan(depth[:, :, 0])).astype(int)

        if self.use_sam3:
            self._log_step("SAM3 Segmentation", f"Running SAM3 text-prompt segmentation for '{object_name}' …")
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
            vis_masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
            if vis_masks:
                vis = overlay_segmentation_masks(rgb, vis_masks)
                self._log_step_update(text=f"Best detection score: {max(scores):.3f}", images=vis)
            else:
                self._log_step_update(text=f"Best detection score: {max(scores):.3f}")
            idxs = np.where(mask.flatten()[binary_map_nan_is_zero.flatten().astype(bool)].astype(bool))
        else:
            self._log_step("OWL-ViT Detection", f"Running OWL-ViT detection for '{object_name}' …")
            dets = self.owl_vit_det_fn(rgb, texts=[[object_name]])

            if len(dets) == 0:
                raise ValueError("No detections; environment constraints or model mismatch")

            boxes = [d["box"] for d in dets]
            labels = [d["label"] for d in dets]
            scores = [d["score"] for d in dets]

            box = boxes[np.argmax(scores)]
            self._log_step_update(text=f"Best detection: '{labels[np.argmax(scores)]}' score={max(scores):.3f}")

            if self.debug:
                img_out = _draw_boxes(
                    rgb, [box], [labels[np.argmax(scores)]], scores=[scores[np.argmax(scores)]]
                )
                out_file = pathlib.Path("owlvit_det.jpg")
                img_out.save(out_file)
                assert out_file.exists() and out_file.stat().st_size > 0

            # save segmentation image
            self._log_step("SAM2 Segmentation", "Running SAM2 segmentation on detected region …")
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
        self._log_step("Point Cloud + OBB", "Computing oriented bounding box from depth point cloud …")
        points, color = depth_color_to_pointcloud(
            depth[:, :, 0], rgb, obs["robot0_robotview"]["intrinsics"]
        )

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points[idxs])

        o3d_points, ind = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        obb = o3d_points.get_oriented_bounding_box()

        # Exposing these to the low level environment for viser
        self._env.cube_center = obb.center
        self._env.cube_rot = obb.R

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

        elapsed = time.time() - start_time
        pos_str = np.array2string(obb_tf_world.wxyz_xyz[-3:], precision=4)
        self._log_step_update(text=f"Position: {pos_str} ({elapsed:.1f}s)")

        # print(f"get_object_pose in {time.time() - start_time} seconds")
        print(f"Object position for {object_name}: {obb_tf_world.wxyz_xyz[-3:]}")
        print(f"Object quaternion wxyz for {object_name}: {obb_tf_world.wxyz_xyz[:4]}")
        if return_bbox_extent:
            print(f"Object extent for {object_name}: {obb.extent}")
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], obb.extent
        else:
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], None

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        self._log_step("sample_grasp_pose", f"Planning grasp for **'{object_name}'** …")
        start_time = time.time()
        obs = self._env.get_observation()

        rbg_imgs = obs_get_rgb(obs)
        assert len(rbg_imgs.keys()) > 0, "No RGB images in obs"

        rgb = list(rbg_imgs.values())[0]
        self._log_step_update(images=rgb)

        depth = obs["robot0_robotview"]["images"]["depth"]

        # Debug image saves TODO: Remove this eventually, or add a debug mode branch
        # save depth image with colormap
        depth_img = depth_to_rgb(depth[:, :, 0])
        depth_img_out = Image.fromarray(depth_img)
        depth_img_out.save("depth_image.jpg")

        binary_map_nan_is_zero = (~np.isnan(depth[:, :, 0])).astype(int)

        if self.use_sam3:
            self._log_step("SAM3 Segmentation", f"Running SAM3 for grasp target '{object_name}' …")
            results = self.sam3_seg_fn(rgb, text_prompt=object_name)
            if len(results) == 0:
                raise ValueError("No sam3 detections")
            scores = [result["score"] for result in results]

            box = results[np.argmax(scores)]["box"]
            segmentation = results[np.argmax(scores)]["mask"][:, :, None]

            if self.debug:
                visualize_sam3_results(
                    Image.fromarray(rgb),
                    object_name,
                    results,
                    output_dir=pathlib.Path("."),
                    show=False,
                )
            vis_masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
            if vis_masks:
                vis = overlay_segmentation_masks(rgb, vis_masks)
                self._log_step_update(text=f"Best detection score: {max(scores):.3f}", images=vis)
            else:
                self._log_step_update(text=f"Best detection score: {max(scores):.3f}")
            idxs = np.where(segmentation.flatten()[binary_map_nan_is_zero.flatten().astype(bool)].astype(bool))
            queried_instance_idx = 1
        else:
            self._log_step("OWL-ViT Detection", f"Running OWL-ViT for grasp target '{object_name}' …")
            dets = self.owl_vit_det_fn(rgb, texts=[[object_name]])

            if len(dets) == 0:
                raise ValueError("No detections; environment constraints or model mismatch")

            boxes = [d["box"] for d in dets]
            labels = [d["label"] for d in dets]
            scores = [d["score"] for d in dets]

            box = boxes[np.argmax(scores)]
            self._log_step_update(text=f"Best detection: '{labels[np.argmax(scores)]}' score={max(scores):.3f}")

            if self.debug:
                img_out = _draw_boxes(
                    rgb, [box], [labels[np.argmax(scores)]], scores=[scores[np.argmax(scores)]]
                )
                out_file = pathlib.Path("owlvit_det.jpg")
                img_out.save(out_file)
                assert out_file.exists() and out_file.stat().st_size > 0

            # save segmentation image
            self._log_step("SAM2 Segmentation", "Running SAM2 segmentation for grasp mask …")
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

        self._env.cube_points = points[idxs]
        self._env.cube_color = color[idxs]

        self._log_step("Contact GraspNet", "Running grasp candidate planning …")
        self._env.grasp_sample, self._env.grasp_scores, self._env.grasp_contact_pts = (
            self.grasp_net_plan_fn(
                depth[:, :, 0],
                obs["robot0_robotview"]["intrinsics"],
                segmentation[:, :, 0],
                queried_instance_idx,
                # local_regions=False,
            )
        )
        self._env.grasp_sample_tf = vtf.SE3.from_matrix(
            self._env.grasp_sample[self._env.grasp_scores.argmax()]
        ) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs["robot0_robotview"]["pose"][3:]),
            translation=obs["robot0_robotview"]["pose"][:3],
        )
        grasp_sample_tf_world = cam_extr_tf @ self._env.grasp_sample_tf

        elapsed = time.time() - start_time
        n_candidates = len(self._env.grasp_scores)
        best_score = float(self._env.grasp_scores.max())
        pos_str = np.array2string(grasp_sample_tf_world.wxyz_xyz[-3:], precision=4)
        self._log_step_update(
            text=f"{n_candidates} candidates, best score={best_score:.3f}\nGrasp position: {pos_str} ({elapsed:.1f}s)"
        )

        # print(f"sample_grasp_pose in {time.time() - start_time} seconds")
        print(f"Grasp sample position for {object_name}: {grasp_sample_tf_world.wxyz_xyz[-3:]}")
        print(
            f"Grasp sample quaternion wxyz for {object_name}: {grasp_sample_tf_world.wxyz_xyz[:4]}"
        )
        return grasp_sample_tf_world.wxyz_xyz[-3:], grasp_sample_tf_world.wxyz_xyz[:4]

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        pos_str = np.array2string(np.asarray(position), precision=4)
        approach_info = f" (z_approach={z_approach:.3f})" if z_approach != 0.0 else ""
        self._log_step("goto_pose", f"Moving to position {pos_str}{approach_info} …")

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        # Align with legacy env: apply TCP offset in end-effector frame
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if self.real:
            quat_wxyz = (vtf.SO3(wxyz=quat_wxyz) @ vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4+np.pi/2)).wxyz

        if (
            z_approach != 0.0
        ):  # If z_approach is not 0.0, approach the object from above by z_approach meters
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None or self.real:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                )
            else:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                    prev_cfg=self.cfg,
                )
                # prev_cfg = self.cfg

                # for i in range(15): # run w/ multiple iterations when using vel_cost ik solver
                #     self.cfg = self.ik_solve_fn(
                #         target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                #         prev_cfg = prev_cfg,
                #     )
                #     if prev_cfg is not None:
                #         # print(f"Error: {np.linalg.norm(self.cfg - prev_cfg)}", np.allclose(self.cfg, prev_cfg, atol=1e-3))
                #         if np.allclose(self.cfg, prev_cfg, atol=1e-3):
                #             break
                #         else:
                #             prev_cfg = self.cfg

            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None or self.real:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                prev_cfg=self.cfg,
            )
            # prev_cfg = self.cfg

            # for i in range(15): # run w/ multiple iterations when using vel_cost ik solver
            #     self.cfg = self.ik_solve_fn(
            #         target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            #         prev_cfg = prev_cfg,
            #     )
            #     if prev_cfg is not None:
            #         # print(f"Error: {np.linalg.norm(self.cfg - prev_cfg)}", np.allclose(self.cfg, prev_cfg, atol=1e-3))
            #         if np.allclose(self.cfg, prev_cfg, atol=1e-3):
            #             break
            #         else:
            #             prev_cfg = self.cfg
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self._log_step_update(text="Motion complete.")

    def home_pose(self) -> None:
        """
        Move the robot to a safe home pose.
        Args:
            None
        Returns:
            None
        """
        self._log_step("home_pose", "Moving robot to home configuration …")

        # joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
        joints = np.array(
            [
                -2.95353726e-02,
                1.69197371e-01,
                2.39244731e-03,
                -2.64089311e00,
                -2.01237851e-03,
                2.94565778e00,
                8.31390616e-01,
            ]
        )
        self._env.move_to_joints_blocking(joints)
        self._log_step_update(text="Home position reached.")

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        self._log_step("open_gripper", "Opening gripper …")
        _open_gripper(self._env, steps=30)
        self._log_step_update(text="Gripper opened.")

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        self._log_step("close_gripper", "Closing gripper …")
        _close_gripper(self._env, steps=30)
        self._log_step_update(text="Gripper closed.")

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
