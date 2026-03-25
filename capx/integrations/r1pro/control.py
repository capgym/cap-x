import pathlib
import time
from typing import Any, Tuple
import math
import torch

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.envs.simulators.r1pro_b1k import R1ProBehaviourLowLevel
from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki import init_pyroki

# from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from capx.integrations.vision.sam2 import init_sam2
from capx.integrations.vision.molmo import init_molmo
from capx.integrations.vision.sam3 import init_sam3, visualize_sam3_results, init_sam3_point_prompt
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb

from scipy.spatial import ConvexHull
from torchvision.transforms.functional import to_pil_image
from omnigibson.sensors.vision_sensor import VisionSensor
import matplotlib.pyplot as plt
import viser
import numpy.typing as npt
from capx.integrations.r1pro.utils import *


from capx.integrations.motion import pyroki_snippets as pks
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from viser.extras import ViserUrdf
import os
from capx.utils.video_utils import _write_video
import jax.numpy as jnp

# pyroki:
# ('torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6', 'left_arm_joint7', 'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6', 'right_arm_joint7')
#dict:
# {
#     'torso_joint1': 0,
#     'torso_joint2': 1,
#     'torso_joint3': 2,
#     'torso_joint4': 3,
#     'left_arm_joint1': 4,
#     'left_arm_joint2': 5,
#     'left_arm_joint3': 6,
#     'left_arm_joint4': 7,
#     'left_arm_joint5': 8,
#     'left_arm_joint6': 9,
#     'left_arm_joint7': 10,
#     'right_arm_joint1': 11,
#     'right_arm_joint2': 12,
#     'right_arm_joint3': 13,
#     'right_arm_joint4': 14,
#     'right_arm_joint5': 15,
#     'right_arm_joint6': 16,
#     'right_arm_joint7': 17,
# }
# robot controller:
# ict_keys(['base_footprint_x_joint', 'base_footprint_y_joint', 'base_footprint_z_joint', 'base_footprint_rx_joint', 'base_footprint_ry_joint', 'base_footprint_rz_joint', 
# 'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2'])
# {
#     'torso_joint1': 0,
#     'torso_joint2': 1,
#     'torso_joint3': 2,
#     'torso_joint4': 3,
#     'left_arm_joint1': 4,
#     'right_arm_joint1': 5,
#     'left_arm_joint2': 6,
#     'right_arm_joint2': 7,
#     'left_arm_joint3': 8,
#     'right_arm_joint3': 9,
#     'left_arm_joint4': 10,
#     'right_arm_joint4': 11,
#     'left_arm_joint5': 12,
#     'right_arm_joint5': 13,
#     'left_arm_joint6': 14,
#     'right_arm_joint6': 15,
#     'left_arm_joint7': 16,
#     'right_arm_joint7': 17,
# }
#mapping:
# [0,1,2,3,4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16, 10, 17]
#reverse mapping:
# [0,1,2,3, 4,6,8,10,12,14,16, 5,7,9,11,13,15,17]

        
# ------------------------------- Control API ------------------------------
class R1ProControlApi(ApiBase):
    """Robot control helpers for R1Pro.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    def __init__(
        self,
        env: R1ProBehaviourLowLevel,
        use_sam3: bool = True,
        debug: bool = False,
        sam3_score_threshold: float = 0.1,
    ) -> None:
        super().__init__(env)
        
        print("init franka control api")
        self.grasp_net_plan_fn = (
            init_contact_graspnet()
        )  # TODO: refactor this and use registered api instead
        print("init grasp net plan fn")
        self.use_sam3 = use_sam3
        self.sam3_score_threshold = sam3_score_threshold
        self.debug = debug
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            self.sam3_point_prompt_fn = init_sam3_point_prompt()
            print("init sam3 seg fn")
        else:
            self.owl_vit_det_fn = init_owlvit(
                device="cuda"
            )  # TODO: refactor this and use registered api instead
            print("init owlvit det fn")
            self.sam2_seg_fn = init_sam2()
            print("init sam2 seg fn")
        self.molmo_point_fn = init_molmo()
        print("init molmo point fn")
        self.cfg = None
        
        self._TCP_OFFSET = np.array([0.0, 0.2, 0.0], dtype=np.float64)
        ctx = get_pyroki_context(os.path.join(ROBOT_ASSETS_ROOT, "models/r1pro/urdf/r1pro_ik.urdf"), target_link_name="left_gripper_link")
        self._pks = pks
        self._robot = ctx.robot
        torso_1_joint_idx = self._robot.joints.actuated_names.index("torso_joint1")
        torso_2_joint_idx = self._robot.joints.actuated_names.index("torso_joint2")
        torso_3_joint_idx = self._robot.joints.actuated_names.index("torso_joint3")
        torso_4_joint_idx = self._robot.joints.actuated_names.index("torso_joint4")
        self._rest_cost_weights = jnp.array([1.0] * self._robot.joints.num_actuated_joints)
        self._rest_cost_weights = self._rest_cost_weights.at[torso_1_joint_idx].set(20.0)   # some height movement allowed
        self._rest_cost_weights = self._rest_cost_weights.at[torso_2_joint_idx].set(20.0) 
        self._rest_cost_weights = self._rest_cost_weights.at[torso_3_joint_idx].set(20.0)
        self._rest_cost_weights = self._rest_cost_weights.at[torso_4_joint_idx].set(20.0)
        
        self.pyroki_joint_mapping = [0,1,2,3,4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16, 10, 17] # from pyroki joint names to controller joint names
        self.controller_joint_mapping = [0,1,2,3, 4,6,8,10,12,14,16, 5,7,9,11,13,15,17] # from controller joint names to pyroki joint names
        
        if self.debug:
        # if True:
            self.viser_server = viser.ViserServer()
            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584
            self.cube_center = None
            self.cube_rot = None
            self.cube_points = None
            self.cube_color = None
            self.grasp_sample = None
            self.grasp_scores = None

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/panda_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )
        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="robot0_robotview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect= 512 / 512,
                scale=0.05,
            )
            
    def functions(self) -> dict[str, Any]:
        fns = {
            "segment_sam3_text_prompt": self.segment_sam3_text_prompt,
            "segment_sam3_point_prompt": self.segment_sam3_point_prompt,
            "point_prompt_molmo": self.point_prompt_molmo,
            "navigate_to_pose": self.navigate_to_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "move_hand": self.move_hand,
            "get_robot_position": self.get_robot_position,
            "get_robot_relative_eef_pose": self.get_robot_relative_eef_pose,
            "reset_torso": self.reset_torso,
            "move_to_joint_positions": self.move_to_joint_positions,
            "get_current_eef_pose": self.get_current_eef_pose,
            "get_current_joint_positions": self.get_current_joint_positions,
            "solve_ik": self.solve_ik,
            "lift_arm": self.lift_arm,
            "check_object_in_hand": self.check_object_in_hand,
            "get_env_observation": self.get_env_observation,
            "write_video": self.write_video,
            "get_sam3_mask": self.get_sam3_mask,
            
            "get_object_pose": self.get_object_pose,
            "find_object_base_rotate": self.find_object_base_rotate,
            "find_object_torso_rotate": self.find_object_torso_rotate,
            "get_navigation_pose": self.get_navigation_pose,
            "save_current_observation": self.save_current_observation,
            "grasp_object": self.grasp_object,
            "sample_grasp_pose": self.sample_grasp_pose,
            # "debug_pyroik": self.debug_pyroik,
        }
        return fns
    
    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image conditioned on a text prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Text prompt for SAM3 segmentation.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool or uint8,
                              where True/1 means the pixel belongs to the instance.
                  - "box": list [x1, y1, x2, y2] in pixel coordinates.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> masks = segment_sam3(rgb, text_prompt="red mug")
        """
        return self.sam3_seg_fn(rgb, text_prompt=text_prompt)
    
    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_coords:
                (x, y) pixel coordinates of the point prompt.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool or uint8,
                              where True/1 means the pixel belongs to the instance.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        return self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)
    
    def point_prompt_molmo(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Use Molmo to point to a coordinate in the image based on a text prompt.

        Args:
            image: np.ndarray: The RGB image to process. Shape: (H, W, 3), dtype uint8.
            text_prompt: str: The text prompt to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """
        return self.molmo_point_fn(Image.fromarray(image), objects=[text_prompt])
    
    
    def get_navigation_pose(self, P_table: np.ndarray, P_object: np.ndarray) -> tuple[float, float, float]:
        """Get the navigation pose for the robot to navigate to the object on the table.
        The navigation pose is the pose that the robot should navigate to in order to reach the object.
        since the object is on the table, we need to navigate to the closest point on the table edge to the object.
        Args:
            P_table: point cloud of the table in the environment in np.ndarray format.
            P_object: point cloud of the object in the environment in np.ndarray format.
        Returns:
            navigation_pose: Navigation pose for the robot to navigate to the object.
        """
        goal = get_navigation_pose(P_table, P_object)
        return goal
        

    def _get_segmentation_map(
        self, obs: dict[str, Any], rgb: np.ndarray, box: list[float] = None
    ) -> np.ndarray:
        images = obs["robot0_robotview"]["images"]
        segmentation = images.get("segmentation")
        if segmentation is not None:
            if segmentation.ndim == 2:
                segmentation = segmentation[..., None]
            return segmentation.astype(np.int32, copy=False)

        print("Running SAM2 segmentation with box:", box)

        masks = self.sam2_seg_fn(rgb, box=box)
        if len(masks) == 0:
            raise RuntimeError("SAM2 returned no masks while attempting to segment scene.")

        if box is not None:
            # Just use mask with the highest score
            max_score = -1
            max_idx = -1
            for idx, entry in enumerate(masks, start=1):
                score = entry.get("score")
                if score is not None and score > max_score:
                    max_score = score
                    max_idx = idx
            masks = [masks[max_idx]]

        height, width = rgb.shape[:2]
        seg_map = np.zeros((height, width, 1), dtype=np.int32)
        for idx, entry in enumerate(masks, start=1):
            mask_obj = entry.get("mask") if isinstance(entry, dict) else None
            if mask_obj is None and hasattr(entry, "mask"):
                mask_obj = entry.mask
            if mask_obj is None:
                continue
            mask = np.asarray(mask_obj, dtype=bool)
            if mask.shape != (height, width):
                try:
                    mask = mask.reshape(height, width)
                except ValueError:
                    continue
            if mask.any():
                seg_map[mask, 0] = idx

        if seg_map.max() == 0:
            print("No masks found with box, Running SAM2 segmentation with global method")
            # Try again with global method (without box)
            masks = self.sam2_seg_fn(rgb)
            if len(masks) == 0:
                raise RuntimeError("SAM2 returned no masks while attempting to segment scene.")

            height, width = rgb.shape[:2]
            seg_map = np.zeros((height, width, 1), dtype=np.int32)
            for idx, entry in enumerate(masks, start=1):
                mask_obj = entry.get("mask") if isinstance(entry, dict) else None
                if mask_obj is None and hasattr(entry, "mask"):
                    mask_obj = entry.mask
                if mask_obj is None:
                    continue
                mask = np.asarray(mask_obj, dtype=bool)
                if mask.shape != (height, width):
                    try:
                        mask = mask.reshape(height, width)
                    except ValueError:
                        continue
                if mask.any():
                    seg_map[mask, 0] = idx

        if seg_map.max() == 0:
            raise RuntimeError("SAM2 masks were empty; cannot build segmentation map.")
        return seg_map

    def _save_segmentation_debug(self, segmentation: np.ndarray, path: pathlib.Path) -> None:
        denom = float(segmentation.max()) if segmentation.max() > 0 else 1.0
        vis = ((np.repeat(segmentation, 3, axis=2) / denom) * 255.0).astype(np.uint8)
        img = Image.fromarray(vis)

        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:  # pragma: no cover - Pillow always available in this repo
            img.save(path)
            return

        draw = ImageDraw.Draw(img)
        height, width = segmentation.shape[:2]
        font_size = max(int(min(height, width) * 0.04), 12)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        unique_vals = np.unique(segmentation)
        for val in unique_vals:
            if val <= 0:
                continue
            mask = segmentation[:, :, 0] == val if segmentation.ndim == 3 else segmentation == val
            if not np.any(mask):
                continue
            ys, xs = np.nonzero(mask)
            cy = float(ys.mean())
            cx = float(xs.mean())
            draw.text(
                (cx, cy),
                str(int(val)),
                fill=(255, 0, 0),
                anchor="mm",
                font=font,
            )

        img.save(path)

    def _compute_bbox_indices(
        self, box: list[float], shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        height, width = shape
        x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(box[2]), x1 + 1, width))
        y2 = int(np.clip(np.ceil(box[3]), y1 + 1, height))
        return x1, x2, y1, y2

    def _select_instance_from_box(
        self, segmentation: np.ndarray, box: list[float]
    ) -> tuple[int, np.ndarray]:
        height, width = segmentation.shape[:2]
        x1, x2, y1, y2 = self._compute_bbox_indices(box, (height, width))
        seg_crop = segmentation[y1:y2, x1:x2]
        unique_vals, counts = np.unique(seg_crop, return_counts=True)
        valid_mask = unique_vals > 0
        if not np.any(valid_mask):
            raise RuntimeError("No segmented instance overlaps detection bounding box.")
        unique_vals = unique_vals[valid_mask]
        counts = counts[valid_mask]
        queried_instance_idx = int(unique_vals[np.argmax(counts)])
        return queried_instance_idx, seg_crop
    
    def get_env_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the observation of the environment.
        Args:
            None
        Returns:
            rgb: RGB image of the environment in np.ndarray format. Shape: (H, W, 3), dtype uint8.
            depth: Depth image of the environment in np.ndarray format. Shape: (H, W), dtype float32.
        """
        obs = self._env.get_observation()
        
        rgb, depth = obs_get_rgb_depth(obs)
        depth = depth.cpu().numpy()
        rgb = rgb.cpu().numpy()[...,:3]

        return rgb, depth
    
    def get_sam3_mask(self, object_name: str) -> np.ndarray:
        """Get the mask of an object in the environment from a natural language description and current camera view.
        Args:
            object_name: The name of the object to get the mask of.
        Returns:
            mask sum: The sum of the mask of the object in the environment, indicating the number of pixels in the mask.
        """
        obs = self._env.get_observation()
        
        rgb, depth = obs_get_rgb_depth(obs)
        depth = depth.cpu().numpy()
        rgb = rgb.cpu().numpy()[...,:3]

        results = self.sam3_seg_fn(rgb, text_prompt=object_name)
        if len(results) == 0:
            raise ValueError("No sam3 detections")
        scores = [result["score"] for result in results]
        
        box = results[np.argmax(scores)]["box"]
        mask = results[np.argmax(scores)]["mask"]
        
        return mask.sum()

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description and current camera view.
        If the object is not found, return None for all return values.

        Args:
            object_name: The name of the object to get the pose of.
            return_bbox_extent:  Whether to return the extent of the oriented bounding box (oriented by quaternion_wxyz). Default is False.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) XYZ in meters (full side length, not half-length extent). If return_bbox_extent is False, returns None.
            o3d_points: point cloud of the object in np.ndarray format.
            obb: Open3D oriented bounding box of the object.
        """
        obs = self._env.get_observation()
        
        rgb, depth = obs_get_rgb_depth(obs)
        depth = depth.cpu().numpy()
        rgb = rgb.cpu().numpy()[...,:3]

        # Debug image saves TODO: Remove this eventually, or add a debug mode branch
        # save depth image with colormap
        # depth_img = depth_to_rgb(depth).astype(np.uint8)
        # depth_img_out = Image.fromarray(depth_img)
        # depth_img_out.save("depth_image.png")

        binary_map_nan_is_zero = (~np.isnan(depth)).astype(int)

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
                
            if np.max(scores) < self.sam3_score_threshold:
                return None, None, None, None, None
            
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

        obj_mask = mask & binary_map_nan_is_zero
        
        if self.debug:
            plt.imshow(obj_mask)
            plt.savefig(f"obj_mask_{object_name}.png")
        
        intrinsic_matrix = self.get_camera_intrinsics()
        T_world_cam = self.get_camera_pose()
        
        points = backproject_depth(np.array(obj_mask), np.array(depth), np.array(intrinsic_matrix), T_world_cam)   # (Nt, 3)
        
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)

        o3d_points, ind = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        obb = o3d_points.get_oriented_bounding_box()
        obb_tf_world = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R), translation=obb.center
        )

        # print(f"get_object_pose in {time.time() - start_time} seconds")
        print(f"Object position for {object_name}: {obb_tf_world.wxyz_xyz[-3:]}")
        print(f"Object quaternion wxyz for {object_name}: {obb_tf_world.wxyz_xyz[:4]}")
        
        if return_bbox_extent:
            print(f"Object extent for {object_name}: {obb.extent}")
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], obb.extent, np.asarray(o3d_points.points), obb
        else:
            return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], None, np.asarray(o3d_points.points), obb
        

    def sample_grasp_pose(self, object_name: str) -> tuple[list[np.ndarray], list[np.ndarray]] | tuple[None, None]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        If the object is not found or no grasp is found, return None for all return values.
        If the object is found, return a list of pregrasp and grasp poses for the object.
        The complete list of pregrasp and grasp poses are:
        - Simple pregrasp pose: The pose to execute before the grasp using a simple topdown grasp pose from the object's oriented bounding box.
        - Simple grasp pose: The pose to execute during the grasp using a simple topdown grasp pose from the object's oriented bounding box.
        - Pregrasp pose: The pose to execute before the grasp using the graspnet grasp pose.
        - Grasp pose: The pose to execute during the grasp using the graspnet grasp pose.
        - Pregrasp pose topdown: The pose to execute before the grasp using the graspnet grasp pose, but with the object's orientation.
        - Grasp pose topdown: The pose to execute during the grasp using the graspnet grasp pose, but with the object's orientation.
        - Pregrasp pose 90: The pose to execute before the grasp using the graspnet grasp pose, but with the object's orientation rotated 90 degrees around the z-axis.
        - Grasp pose 90: The pose to execute during the grasp using the graspnet grasp pose, but with the object's orientation rotated 90 degrees around the z-axis.
        If no graspnet detections are found, return the simple pregrasp and grasp poses.
        Args:
            object_name: The name of the object to sample a grasp pose for.
        Returns:
            if object is found:
                pregrasp_poses: List of pregrasp poses to execute, [pregrasp_pose_topdown, simple_pregrasp_pose, pregrasp_pose, pregrasp_pose_90] or [simple_pregrasp_pose] if no graspnet detections are found.
                grasp_poses: List of grasp poses to execute, [ grasp_pose_topdown, simple_grasp_pose, grasp_pose, grasp_pose_90] or [simple_grasp_pose] if no graspnet detections are found.
            if object is not found:
                return None, None
        """
        obs = self._env.get_observation()

        rgb, depth = obs_get_rgb_depth(obs)
        depth = depth.cpu().numpy()
        rgb = rgb.cpu().numpy()[...,:3]

        binary_map_nan_is_zero = (~np.isnan(depth)).astype(int)

        if self.use_sam3:
            results = self.sam3_seg_fn(rgb, text_prompt=object_name)
            if len(results) == 0:
                raise ValueError("No sam3 detections")
            scores = [result["score"] for result in results]
            if np.max(scores) < self.sam3_score_threshold:
                print(f"No sam3 detections for {object_name} and no grasp poses found")
                return None, None

            box = results[np.argmax(scores)]["box"]
            mask = results[np.argmax(scores)]["mask"]
            segmentation = mask[:, :, None]

            if self.debug:
                visualize_sam3_results(
                    Image.fromarray(rgb),
                    object_name,
                    results,
                    output_dir=pathlib.Path("."),
                    show=False,
                )
            idxs = np.where(segmentation.flatten() & binary_map_nan_is_zero.flatten().astype(bool))
            queried_instance_idx = 1
        

        obj_mask = mask & binary_map_nan_is_zero
        camera_intrinsics = self.get_camera_intrinsics()
        T_world_cam = self.get_camera_pose()
        
        points = backproject_depth(np.array(obj_mask), np.array(depth), np.array(camera_intrinsics), T_world_cam)   # (Nt, 3)

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)

        o3d_points, ind = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        obb = o3d_points.get_oriented_bounding_box()
        
        simple_pregrasp_pose, simple_grasp_pose = self.sample_grasp_pose_simple(object_name, obb)
        
        grasp_sample, grasp_scores, grasp_contact_pts = (
            self.grasp_net_plan_fn(
                depth,
                camera_intrinsics,
                segmentation[:, :, 0],
                queried_instance_idx,
            )
        )
        if len(grasp_sample) == 0:
            print(f"No graspnet detections for {object_name} using simple grasp poses")
            return [simple_pregrasp_pose], [simple_grasp_pose]
        
        grasp_sample_tf = vtf.SE3.from_matrix(
            grasp_sample[grasp_scores.argmax()]
        ) @ vtf.SE3.from_translation(np.array([0, 0, 0.05]))
        # grasp_sample_tf = vtf.SE3.from_matrix( grasp_sample[grasp_scores.argmax()]  )

        cam_extr = self.get_camera_pose()
        cam_extr_tf = vtf.SE3.from_matrix(cam_extr)
        grasp_sample_gl = convert_T_cam_cv_to_cam_gl(grasp_sample_tf.as_matrix())
        grasp_sample_tf_gl = vtf.SE3.from_matrix(grasp_sample_gl)
        
        grasp_sample_tf_world = cam_extr_tf @ grasp_sample_tf_gl
        # print(f"sample_grasp_pose in {time.time() - start_time} seconds")
        print(f"Grasp sample position for {object_name}: {grasp_sample_tf_world.wxyz_xyz[-3:]}")
        print(
            f"Grasp sample quaternion wxyz for {object_name}: {grasp_sample_tf_world.wxyz_xyz[:4]}"
        )
        
        grasp_pos = torch.tensor(grasp_sample_tf_world.wxyz_xyz[-3:].copy(), dtype=torch.float32)
        grasp_quat = torch.tensor(grasp_sample_tf_world.wxyz_xyz[:4][[1,2,3,0]].copy(), dtype=torch.float32)
        approach_dir = quat2mat(grasp_quat) @ torch.tensor([0.0, 0.0, -1.0])
        pregrasp_pos = grasp_pos - approach_dir * 0.1
        grasp_quat = quat_multiply(grasp_quat, torch.tensor([1.0, 0.0, 0.0, 0.0]))
        
        theta = np.pi / 2
        qz = np.array([0, 0, np.sin(theta/2), np.cos(theta/2)])
        grasp_quat_90 = quat_multiply(grasp_quat, torch.tensor(qz, dtype=torch.float32))
        
        pregrasp_pose = (pregrasp_pos.numpy(), grasp_quat.numpy())
        grasp_pose = (grasp_pos.numpy(), grasp_quat.numpy())
        
        pregrasp_pose_topdown = (pregrasp_pos.numpy(), np.array([1.0, 0.0, 0.0, 0.0]))
        grasp_pose_topdown = (grasp_pos.numpy(), np.array([1.0, 0.0, 0.0, 0.0]))
        
        pregrasp_pose_90 = (pregrasp_pos.numpy(), grasp_quat_90.numpy())
        grasp_pose_90 = (grasp_pos.numpy(), grasp_quat_90.numpy())

        if self.debug:
        # if True:
            self._viser_init_check()
            
            points, colors = depth_color_to_pointcloud_gl(
                        depth,
                        rgb,
                        camera_intrinsics,
                        T_world_cam=T_world_cam,
                    )

            self.viser_server.scene.add_point_cloud(
                "grasp_sample_point_cloud",
                points,
                colors,
                point_size=0.01,
                point_shape="square",
            )
            
            # self.viser_server.scene.add_frame(
            #     "robot0_robotview/simple_pregrasp",
            #     position=simple_pregrasp_pose[0],
            #     wxyz=simple_pregrasp_pose[1][[3,0,1,2]],
            #     axes_length=0.5,
            #     axes_radius=0.015,
            # )
            self.viser_server.scene.add_frame(
                "robot0_robotview/simple_grasp",
                position=simple_grasp_pose[0],
                wxyz=simple_grasp_pose[1][[3,0,1,2]],
                axes_length=0.5,
                axes_radius=0.015,
            )
            # self.viser_server.scene.add_frame(
            #     "robot0_robotview/pregrasp",
            #     position=pregrasp_pose[0],
            #     wxyz=pregrasp_pose[1][[3,0,1,2]],
            #     axes_length=0.5,
            #     axes_radius=0.015,
            # )
            # self.viser_server.scene.add_frame(
            #     "robot0_robotview/grasp",
            #     position=grasp_pose[0],
            #     wxyz=grasp_pose[1][[3,0,1,2]],
            #     axes_length=0.5,
            #     axes_radius=0.015,
            # )
            # self.viser_server.scene.add_frame(
            #     "robot0_robotview/pregrasp_topdown",
            #     position=pregrasp_pose_topdown[0],
            #     wxyz=pregrasp_pose_topdown[1][[3,0,1,2]],
            #     axes_length=0.5,
            #     axes_radius=0.015,
            # )
            self.viser_server.scene.add_frame(
                "robot0_robotview/grasp_topdown",
                position=grasp_pose_topdown[0],
                wxyz=grasp_pose_topdown[1][[3,0,1,2]],
                axes_length=0.5,
                axes_radius=0.015,
            )
            # self.viser_server.scene.add_frame(
            #     "robot0_robotview/pregrasp_90",
            #     position=grasp_pose_90[0],
            #     wxyz=grasp_pose_90[1][[3,0,1,2]],
            #     axes_length=0.5,
            #     axes_radius=0.015,
            # )
        # return [simple_pregrasp_pose,pregrasp_pose,pregrasp_pose_topdown,   pregrasp_pose_90], [simple_grasp_pose,grasp_pose, grasp_pose_topdown,   grasp_pose_90]
        return [simple_pregrasp_pose,pregrasp_pose_topdown], [simple_grasp_pose, grasp_pose_topdown]
        

    def sample_grasp_pose_simple(self, object_name: str, object_obb: o3d.geometry.OrientedBoundingBox) -> tuple[np.ndarray, np.ndarray]:
        """Sample a simple topdown grasp pose for an object in the environment from the object's oriented bounding box.
        The grasp pose is the object's center position and the object's orientation.
        Args:
            object_name: Name of the object to sample a grasp pose for.
            object_obb: Oriented bounding box of the object.
        Returns:
            pregrasp_pose: Pregrasp pose to execute, (position, quaternion_xyzw).
            grasp_pose: Grasp pose to execute, (position, quaternion_xyzw).
        """
        pregrasp_pose, grasp_pose = self._env._sample_grasp_pose(object_name, object_obb)
        
        return (pregrasp_pose[0].numpy(), pregrasp_pose[1].numpy()), (grasp_pose[0].numpy(), grasp_pose[1].numpy())
    
    
    def check_object_in_hand(self, arm=0) -> None:
        """
        Check if the grasp was successful by checking if there is an object in the hand. Note that this function may still return True if the wrong object is in the hand.
        Args:
            arm: Arm to check the grasp for. 0: left arm, 1: right arm.
        Returns:
            grasped_success: Whether the grasp was successful.
        """
        return self._env.check_object_in_hand(arm=arm)
        
    def grasp_object(self, pregrasp_pose: np.ndarray, grasp_pose: np.ndarray, object_name: str, arm=0) -> None:
        """
        Grasp an object in the environment.
        Args:
            pregrasp_pose: Pregrasp pose to execute, (position, quaternion_xyzw).
            grasp_pose: Grasp pose to execute, (position, quaternion_xyzw).
            object_name: Name of the object to grasp.
            arm: Arm to grasp the object for. 0: left arm, 1: right arm.
        Returns:
            None.
        """
        if pregrasp_pose is None or grasp_pose is None:
            print("Pregrasp or grasp pose not found, grasp failed")
            return False
        
        self.open_gripper(arm=arm)
        
        joints = self.solve_ik(grasp_pose[0]+np.array([0.0, 0.0, -0.02]), grasp_pose[1][[3,0,1,2]], arm=arm, offset_translation=np.array([0.0, 0.0, -0.02]))
        success = self.move_to_joint_positions(joints, max_steps=200, settle_steps=5)
        self._env._settle_robot()
        current_eef_pose = self.get_current_eef_pose(arm=arm)[0]
        if np.linalg.norm(current_eef_pose - grasp_pose[0]) > 0.03:
            diff = current_eef_pose - grasp_pose[0]
            diff[-1] = 0.08
            new_grasp_pose = grasp_pose[0] - diff
            joints = self.solve_ik(new_grasp_pose, grasp_pose[1][[3,0,1,2]], arm=arm, offset_translation=np.array([0.0, 0.0,0.0]))
            success = self.move_to_joint_positions(joints, max_steps=200, settle_steps=5)
            self._env._settle_robot()
      
                
        print("closing gripper")
        self.close_gripper(arm=arm)
       
        print("lifting arm")
        self._env._settle_robot()
        self.lift_arm(arm=arm)
        print("resetting torso")
        self.reset_torso()
        # self.write_video(f"grasp_{arm}_0_0_0_02")

    def execute_motion_plan(self, target_joint_positions, num_steps=20) -> None:
        """
        Execute a motion plan in the environment.
        Args:
            q_traj: Joint trajectory to execute.
        Returns:
            None.
        """
        cur_joint_positions = self.get_current_joint_positions()
        q_interp = np.linspace(cur_joint_positions, target_joint_positions, num_steps)
        self._env._execute_motion_plan(torch.from_numpy(q_interp))

    def get_robot_relative_eef_pose(self, arm=0) -> np.ndarray:
        """
        Get the relative end-effector pose of the robot.
        Returns:
            relative_eef_pose: Relative end-effector pose of the robot in robot base frame.
        """
        if arm == 0:
            arm = "left"
        elif arm == 1:
            arm = "right"
        return self._env.robot.get_relative_eef_pose(arm=arm)
    
    def lift_arm(self, arm=0) -> None:
        """
        Lift the arm in the environment.
        Args:
            arm: Arm to lift the arm for. 0: left arm, 1: right arm.
        Returns:
            None.
        """
        self._env._lift_arm(arm=arm)
        
    def reset_robot_joints(self) -> None:
        """Reset the robot joints to the initial joint positions.
        Returns:
            success: Whether the robot joints were reset successfully.
        """
        default_joints = np.array([-7.54267931e-01, -1.41683924e+00,  5.06645488e-03, -1.01875293e-03,
        9.54848947e-04, -1.16794765e-01, -2.27557393e-04, -3.49595532e-04,
       -1.72530690e-05,  1.09918790e-06,  1.79652070e-05,  1.03364218e-05,
       -1.34534719e-06,  2.18674671e-07,  2.28137324e-05,  1.80529587e-05,
       -1.05772679e-05, -2.68483768e-06,  1.21251214e-05,  1.19145689e-05,
        3.75604009e-06,  1.10797237e-06,  2.14736724e-06,  1.48261506e-06,
        2.99998261e-02,  2.99998168e-02,  2.99997535e-02,  2.99997460e-02], dtype=np.float32)
        current_joints = self.get_current_joint_positions().copy()
        current_joints[10:24] = default_joints[10:24]
        success = self.move_to_joint_positions(current_joints, max_steps=500)
        if not success:
            print("Failed to reset robot joints")
            return False
        return True
    
    def debug_pyroik(self):
        self._viser_init_check()
        
        obs = self._env.get_observation()

        rgb, depth = obs_get_rgb_depth(obs)
        depth = depth.cpu().numpy()
        rgb = rgb.cpu().numpy()[...,:3]
        camera_intrinsics = self.get_camera_intrinsics()
        T_world_cam = self.get_camera_pose()
            
        points, colors = depth_color_to_pointcloud_gl(
                    depth,
                    rgb,
                    camera_intrinsics,
                    T_world_cam=T_world_cam
                )
        self.viser_server.scene.add_point_cloud(
                "grasp_sample_point_cloud",
                points,
                colors,
                point_size=0.01,
                point_shape="square",
            )

        arm = 0
        cur_eef_pose = self.get_current_eef_pose(arm=arm)
        robot_pos, robot_quat, robot_yaw = self.get_robot_position()
        relative_eef_pose = self.get_robot_relative_eef_pose(arm=arm)
        
        self.viser_server.scene.add_frame(
                "robot_eef",
                position=relative_eef_pose[0].numpy() + robot_pos,
                wxyz=relative_eef_pose[1].numpy()[[3,0,1,2]],
                axes_length=0.5,
                axes_radius=0.015,
            )
        
        self.viser_server.scene.add_frame(
                "robot_eef_w",
                position=cur_eef_pose[0],
                wxyz=cur_eef_pose[1][[3,0,1,2]],
                axes_length=0.5,
                axes_radius=0.015,
            )
        
        cur_joints = self.get_current_joint_positions()
        cur_pyroki_joints = cur_joints[6:24][self.controller_joint_mapping]
        fk_eef = self._robot.forward_kinematics(cur_pyroki_joints)
        
        if arm == 0:
            gripper_index = 20
        elif arm == 1:
            gripper_index = 32
        gripper_pose = fk_eef[gripper_index]
        gripper_pose_roki = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=gripper_pose[:4]),
            translation=gripper_pose[4:],
        )
        self.viser_server.scene.add_frame(
                "robot_eef_fk_now",
                position=gripper_pose_roki.translation() + robot_pos,
                wxyz=gripper_pose_roki.rotation().wxyz,
                axes_length=0.5,
                axes_radius=0.015,
            )
        
        qy = np.array([0, 1.0, 0.0, 0.0]) 
        fk_eef_y = quat_multiply(torch.from_numpy(np.array(gripper_pose[:4]))[[1,2,3,0]], torch.tensor(qy, dtype=torch.float32))
        
        y_tra = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=np.array([0.0, 0, 1.0, 0.0 ])),
            translation=np.array([0.0, -0.0, -0.05]),
        )
        eff_y = gripper_pose_roki @ y_tra
        self.viser_server.scene.add_frame(
                "robot_eef_fk_now_y",
                position=eff_y.translation() + robot_pos,
                wxyz=eff_y.rotation().wxyz,
                axes_length=0.5,
                axes_radius=0.015,
            )
        
        solved_joints, joints = self.solve_ik(cur_eef_pose[0]+np.array([0.0, 0.0, 0.25]), cur_eef_pose[1][[3,0,1,2]], arm=arm)
                
        res  = self._robot.forward_kinematics(joints)
        gripper_pose = res[gripper_index]
        
        self.viser_server.scene.add_frame(
                "robot_eef_fk",
                position=gripper_pose[4:] + robot_pos,
                wxyz=gripper_pose[:4],
                axes_length=0.5,
                axes_radius=0.015,
            )
        
        success = self.move_to_joint_positions(solved_joints)
        
        cur_eef_pose_post = self.get_current_eef_pose(arm=arm)
        self.viser_server.scene.add_frame(
            "pyroik_ik_fk",
            position=cur_eef_pose_post[0],
            wxyz=cur_eef_pose_post[1][[3,0,1,2]],
            axes_length=0.5,
            axes_radius=0.015,
        )
        
        frames = self._env.get_video_frames()
        frame=frames['rgb']

        _write_video(frame, './', suffix='debug')

    def write_video(self, name: str):
        """Write a video of the environment.
        Args:
            frame: Frame to write.
            suffix: Suffix of the video file.
        """

        frames = self._env.get_video_frames()
        frame_rgb=frames['rgb']
        _write_video(frame_rgb, './', suffix=name)
        frame_depth=frames['ego']
        _write_video(frame_depth, './', suffix=name + '_ego')
        frame_left_wrist=frames['left_wrist']
        _write_video(frame_left_wrist, './', suffix=name + '_left_wrist')
        frame_right_wrist=frames['right_wrist']
        _write_video(frame_right_wrist, './', suffix=name + '_right_wrist')
    
    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        arm: int = 0,
        offset_translation: np.ndarray = np.array([0.02, 0.0, -0.05]),
    ) -> np.ndarray:
        """Solve inverse kinematics for the R1Pro right hand link.

        Args:
            position:
                Target position in world frame.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in world frame.
                Shape: (4,), [w, x, y, z], dtype float64.
            arm: int = 0, Arm to solve the IK for. 0: left arm, 1: right arm.
        Returns:
            joints:
                np.ndarray of shape (28,), dtype float64.
                Joint angles for the 28 DoF R1Pro. 
        """
        if arm == 0:
            target_link_name = "left_gripper_link"
        elif arm == 1:
            target_link_name = "right_gripper_link"
        
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        
        relative_eef_pose = self.convert_eef_world_pose_to_robot_base_pose((pos, quat_xyzw))
        pos = relative_eef_pose[0]
        target_eef_quat = relative_eef_pose[1] #xyzw
        
        # qy = np.array([0, 1.0, 0.0, 0.0]) 
        # target_quat_xyzw = quat_multiply(torch.from_numpy(target_eef_quat), torch.tensor(qy, dtype=torch.float32))
        # quat_xyzw = target_quat_xyzw.numpy()
        # quat_wxyz = quat_xyzw[[3,0,1,2]]
        
        # rot = SciRotation.from_quat(quat_xyzw)
        # offset_pos = pos + rot.apply(self._TCP_OFFSET)
        
        y_tra = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=np.array([0.0, 0, 1.0, 0.0 ])),
            translation=offset_translation,
        )
        y_tra_inv = y_tra.inverse()
        eef_pose_se3 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=target_eef_quat[[3,0,1,2]]),
            translation=pos,
        )
        eef_pose_se3_y = eef_pose_se3 @ y_tra_inv
        offset_pos = eef_pose_se3_y.translation()
        quat_wxyz = eef_pose_se3_y.rotation().wxyz
        
        
        try:
            current_q = self.get_current_joint_positions()[6:24][self.controller_joint_mapping]
            cfg = self._pks.solve_ik_rest(
                robot=self._robot,
                target_link_name=target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                rest_cost_weights=self._rest_cost_weights,
                initial_q=current_q,
            )
        except TimeoutError:
            raise  # don't swallow SIGALRM timeout
        except Exception as e:
            print(f"IK solve error: {e}")
        joints = np.asarray(cfg, dtype=np.float64)
        
        target_joint_positions = torch.from_numpy(self.get_current_joint_positions()).clone()
        robot_joint_mapped = joints[self.pyroki_joint_mapping] #now in robot controller joint order
        if arm == 0:
            joint_index = [0,1,2,3,4,6,8,10,12,14,16] # torso and left arm, ignore right arm
        else:
            joint_index = [0,1,2,3,5,7,9,11,13,15,17] # torso and right arm, ignore left arm
            
        controller_joint_index = np.array(joint_index) + 6 
        controller_joint_index = controller_joint_index.astype(np.int32).tolist()
        target_joint_positions[controller_joint_index] = torch.tensor(robot_joint_mapped[joint_index], dtype=torch.float32)
        return target_joint_positions.numpy() #, joints
    
    def navigate_to_pose(
        self, pose_2d
    ) -> None:
        """
        Navigate to a pose in the environment.
        Args:
            pose: Pose to navigate to, xy and yaw in world frame.
            if the pose is not reachable, interpolate between the robot pose and the goal pose to find a reachable pose.
        Returns:
            success: Whether the pose was navigated to successfully.
        """
        robot_pos, robot_quat, robot_yaw = self.get_robot_position()
        robot_pose = (robot_pos[0], robot_pos[1], robot_yaw)
        goal = pose_2d
        
        # success = self._env._navigate_to_pose(goal)
        # return success
        
        #interpolated goals between robot pose and goal
        num_waypoints = 5
        waypoints = np.linspace(goal, robot_pose, num_waypoints)
        for waypoint in waypoints:
            success = self._env._navigate_to_pose(waypoint)
            if success:
                print("Reached waypoint", waypoint)
                return True
        return False
        
  
    def move_hand(self, target_pose: tuple[np.ndarray, np.ndarray], arm=0) -> None:
        """
        Move the hand to a target pose in the environment. This function will ignore all obstacles except the robot itself.
        Args:
            target_pose: Target pose to move the hand to, a tuple of (position, quaternion (xyzw)).
            arm: Arm to move the hand for. 0: left arm, 1: right arm.
        Returns:
            success: Whether the hand was moved successfully.
        """
        # success = self._env._move_hand(target_pose, ignore_all_obstacles=True, skip_obstacle_update=True, ik_only=True, ik_world_collision_check=False)
        success = self._env._move_hand((torch.from_numpy(target_pose[0]), torch.from_numpy(target_pose[1])), arm=arm, ignore_all_obstacles=True, skip_obstacle_update=True)
        self._env._settle_robot()
        return success

    def open_gripper(self, arm=0) -> None:
        """Open gripper fully.
        Args:
            arm: Arm to open the gripper for. 0: left arm, 1: right arm.
        """
        self._env._open_close_gripper(arm=arm, max_steps=100, open=True)

    def close_gripper(self, arm=0) -> None:
        """Close gripper fully.

        Args:
            arm: Arm to close the gripper for. 0: left arm, 1: right arm.
        """
        self._env._open_close_gripper(arm=arm, max_steps=100, open=False)
    
    def get_current_joint_positions(self) -> np.ndarray:
        """
        Get the current joint positions in the environment.
        Returns:
            current_joint_positions: Current joint positions in the environment.
        """
        return self._env.get_joint_positions().numpy()
    
    def get_current_eef_pose(self, arm=0) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current end-effector pose in the environment.
        Args:
            arm: Arm to get the end-effector pose for. 0: left arm, 1: right arm.
        Returns:
            current_eef_pose: Current end-effector pose in the environment, a tuple of (position, quaternion (xyzw)).
        """
        eef_pose = self._env.get_robot_eef_pose(arm=arm)
        return (eef_pose[0].numpy(), eef_pose[1].numpy())
        
    def move_to_joint_positions(self, target_joint_positions, max_steps=20, settle_steps=10) -> None:
        """
        Move the robot to a target joint positions in the environment.
        Joint orders:
        (['base_footprint_x_joint', 'base_footprint_y_joint', 'base_footprint_z_joint', 'base_footprint_rx_joint', 'base_footprint_ry_joint', 'base_footprint_rz_joint', 'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2'])
        The first 6 joints are the base joints, the next 4 joints are the torso joints, the next 14 joints are the arm joints, the last 4 joints are the gripper joints. You should avoid moving the base joints directly.
        Args:
            target_joint_positions: Target joint positions to move the robot to.
            max_steps: Maximum number of steps to move the robot to the target joint positions.
            interpolation_steps: Number of intermediate joint positions to interpolate between the current and target joint positions.
        Returns:
            success: Whether the joint positions were reached successfully.
        """
        if isinstance(target_joint_positions, np.ndarray):
            target_joint_positions = torch.from_numpy(target_joint_positions)
        success = self._env._move_to_joint_positions(target_joint_positions, max_steps, settle_steps)
        return success

    def breakpoint_code_block(self) -> None:
        """Call this function to mark a significant checkpoint where you want to evaluate progress and potentially regenerate the remaining code.

        Args:
            None
        """
        return None
    
    
    def detect_object_sam3(self, object_name: str) -> None:
        """
        Detect an object in the environment.
        Args:
            object_name: Name of the object to detect.
        Returns:
            success: Whether the object was detected.
        """
        obs = self._env.get_observation()
        rgb, depth = obs_get_rgb_depth(obs)
        rgb = rgb.cpu().numpy()[...,:3]    
        results = self.sam3_seg_fn(rgb, text_prompt=object_name)
        if self.debug:
            visualize_sam3_results(
                Image.fromarray(rgb),
                object_name,
                results,
                output_dir=pathlib.Path("."),
                show=False,
            )
        if len(results) == 0:
            return False
        scores = [result["score"] for result in results]
        if np.max(scores) > self.sam3_score_threshold:
            # find the object
            return True
        return False
    
    def find_object_torso_rotate(self, object_name: str) -> None:
        """
        If failed to find the object with base rotation, rotate the torso to move cameras up and down until the object is found in the current field of view. Rotate the torso to move cameras up and down until the object is found in the current field of view.
        Args:
            object_name: Name of the object to find.
        Returns:
            success: Whether the object was found.
        """
        # torso 2 positive is going front
        # torso 1 positive is going front
        # to move back and look down, we need to move torso 1 back and torso 2 forward
        # torso_joint_limit = self._env.torso_joint_limits
        torso_joint_1 = self._env.controller.robot.joints["torso_joint1"]
        torso_joint_2 = self._env.controller.robot.joints["torso_joint2"]
        start_torso_joint_1= torso_joint_1.get_state()[0][0]
        start_torso_joint_2= torso_joint_2.get_state()[0][0]
    
        delta = 0.25
        sample_torso_joint_positions_1 = np.linspace(start_torso_joint_1, start_torso_joint_1 - delta*6, 6)
        sample_torso_joint_positions_2 = np.linspace(start_torso_joint_2, start_torso_joint_2 + delta*10, 6)

        for sample_torso_joint_position_1, sample_torso_joint_position_2 in zip(sample_torso_joint_positions_1, sample_torso_joint_positions_2):
            self._env.update_torso("torso_joint1", torch.tensor(sample_torso_joint_position_1))
            self._env.update_torso("torso_joint2", torch.tensor(sample_torso_joint_position_2))
            # self.save_current_observation(f"torso_joint_1_{sample_torso_joint_position_1}_torso_joint_2_{sample_torso_joint_position_2}")
            if self.detect_object_sam3(object_name):
                return True
            
        sample_torso_joint_positions_1 = np.linspace(start_torso_joint_1, start_torso_joint_1 + delta*6, 6)
        sample_torso_joint_positions_2 = np.linspace(start_torso_joint_2, start_torso_joint_2 - delta*10, 6)
        self._env.update_torso("torso_joint1", torch.tensor(start_torso_joint_1), 200)
        self._env.update_torso("torso_joint2", torch.tensor(start_torso_joint_2), 200)

        for sample_torso_joint_position_1, sample_torso_joint_position_2 in zip(sample_torso_joint_positions_1, sample_torso_joint_positions_2):
            self._env.update_torso("torso_joint1", torch.tensor(sample_torso_joint_position_1))
            self._env.update_torso("torso_joint2", torch.tensor(sample_torso_joint_position_2))
            # self.save_current_observation(f"torso_joint_1_{sample_torso_joint_position_1}_torso_joint_2_{sample_torso_joint_position_2}")
            
            if self.detect_object_sam3(object_name):
                return True
            
        return False    
    
    def reset_torso(self) -> None:
        """
        Reset the torso to the initial position.
        Args:
            None.
        Returns:
            None.
        """
        self._env.update_torso("torso_joint1", torch.tensor(0.0))
        self._env.update_torso("torso_joint2", torch.tensor(0.0))
    

    def find_object_base_rotate(self, object_name: str) -> None:
        """
        Rotate the robot base until the object is found in the current field of view.
        Args:
            object_name: Name of the object to find.
        Returns:
            success: Whether the object was found.
        """
        robot_pos, robot_quat = self._env.robot.get_position_orientation()
        cur_yaw = quat2yaw(robot_quat)
        delta_yaw = 0.5
        num_idx = 2*np.pi // delta_yaw
        obs = self._env.get_observation()
        for idx in range(int(num_idx)):
            
            robot_pos, robot_quat = self._env.robot.get_position_orientation()
            cur_yaw = quat2yaw(robot_quat)
            new_yaw = cur_yaw + delta_yaw if idx >0 else cur_yaw
            new_pose = (robot_pos[0], robot_pos[1], new_yaw)
            self._env._navigate_to_pose(new_pose)
            print("Navigating to pose", new_pose)
            if self.detect_object_sam3(object_name):
                print("Object found")
                return True
        return False
    
    # def find_object_base_rotate(self, object_name: str) -> None:
    #     """
    #     Rotate the robot base until the object is found in the current field of view.
    #     Args:
    #         object_name: Name of the object to find.
    #     Returns:
    #         success: Whether the object was found.
    #     """
    #     torso_joint_4 = self._env.controller.robot.joints["torso_joint4"]
    #     start_torso_joint_4= torso_joint_4.get_state()[0][0]

    #     delta = 0.5
    #     sample_torso_joint_positions_4 = np.linspace(start_torso_joint_4, start_torso_joint_4 + delta*6, 6)

    #     for sample_torso_joint_position_4 in sample_torso_joint_positions_4:
    #         self._env.update_torso("torso_joint4", torch.tensor(sample_torso_joint_position_4))
    #         if self.detect_object_sam3(object_name):
    #             self._env.update_torso("torso_joint4", torch.tensor(start_torso_joint_4), 200)
    #             return True
        
    #     sample_torso_joint_positions_4 = np.linspace(start_torso_joint_4 - delta*6, start_torso_joint_4,  6)
    #     for sample_torso_joint_position_4 in sample_torso_joint_positions_4:
    #         self._env.update_torso("torso_joint4", torch.tensor(sample_torso_joint_position_4))
    #         if self.detect_object_sam3(object_name):
    #             self._env.update_torso("torso_joint4", torch.tensor(start_torso_joint_4), 200)
    #             return True
            
    #     return False    
            
    def get_camera_pose(self) -> np.ndarray:
        """
        Get the camera pose in the environment.
        Returns:
            camera_pose: Camera pose in the environment.
        """
        obs = self._env.get_observation()
        all_keys = list(obs.keys())
        robot_key = [key for key in all_keys if 'robot' in key][0]
        ego_camera = self._env.robot.sensors[f'{robot_key}:zed_link:Camera:0']
        ego_camera_pos, ego_camera_quat = ego_camera.get_position_orientation()
        T_world_cam = pose_to_T_world_cam(np.array(ego_camera_pos), np.array(ego_camera_quat))
        return T_world_cam
    
    def get_camera_intrinsics(self) -> np.ndarray:
        """
        Get the camera intrinsics in the environment.
        Returns:
            camera_intrinsics: Camera intrinsics in the environment.
        """
        obs = self._env.get_observation()
        all_keys = list(obs.keys())
        robot_key = [key for key in all_keys if 'robot' in key][0]
        ego_camera = self._env.robot.sensors[f'{robot_key}:zed_link:Camera:0']
        side_camera = self._env.robot.sensors[f'{robot_key}:left_realsense_link:Camera:0']
        side_intrinsic_matrix = side_camera.intrinsic_matrix
        ego_intrinsic_matrix = ego_camera.intrinsic_matrix
        return ego_intrinsic_matrix.detach().numpy()
    
    def get_robot_position(self) -> np.ndarray:
        """
        Get the robot position in the environment.
        Returns:
            robot_position: Robot position in the environment.
            robot_quat: Robot quaternion in the environment in xyzw format.
            robot_yaw: Robot yaw in the environment.
        """
        robot_pos, robot_quat = self._env.robot.get_position_orientation()
        robot_yaw = quat2yaw(robot_quat)
        return np.array(robot_pos), np.array(robot_quat), np.array(robot_yaw)

    
    def convert_eef_world_pose_to_robot_base_pose(self, eef_world_pose: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the end-effector world pose to the robot base pose.
        Args:
            eef_world_pose: End-effector world pose, (position, quaternion).
        Returns:
            robot_base_pose: Robot base pose, (position, quaternion).
        """
        base_link_pose = self._env.robot.get_position_orientation()
        
        pose = relative_pose_transform(torch.from_numpy(eef_world_pose[0]), torch.from_numpy(eef_world_pose[1]), *base_link_pose)
        return pose[0].numpy(), pose[1].numpy()
    
    def save_current_observation(self, name) -> None:
        """
        Save the current observation in the environment.
        Args:
            None.
        Returns:
            None.
        """
        obs = self._env.get_observation()
        rgb, depth = obs_get_rgb_depth(obs)
        rgb = rgb.cpu().numpy()[...,:3]
        plt.imshow(rgb)
        plt.savefig(f'{name}_rgb.png')
        external_rgb = self._env.env.external_sensors['external_camera'].get_obs()[0]['rgb'][:,:,:3]
        plt.imshow(external_rgb)
        plt.savefig(f'{name}_external_rgb.png')


def _draw_boxes(
    rgb: np.ndarray, boxes: list[list[float]], labels: list[str], scores: list[float] | None = None
) -> Image.Image:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for b, lab in zip(boxes, labels, strict=False):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), lab, fill=(255, 0, 0))
    if scores is not None:
        for b, score in zip(boxes, scores, strict=False):
            x1, y1, x2, y2 = b
            draw.text((x1 + 100, max(0, y1 - 12)), f"{score:.2f}", fill=(255, 0, 0))
    return img
