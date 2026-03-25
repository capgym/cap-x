from typing import Any
import pathlib
import time
import viser.transforms as vtf
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation
from capx.integrations.vision.sam3 import init_sam3
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.motion.pyroki import init_pyroki

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    build_segmentation_map_from_sam2,
    close_gripper as _close_gripper,
    close_gripper_arm1 as _close_gripper_arm1,
    compute_bbox_indices,
    draw_boxes,
    open_gripper as _open_gripper,
    open_gripper_arm1 as _open_gripper_arm1,
    save_segmentation_debug,
    select_instance_from_box,
    transform_pose_arm0_to_arm1,
)
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb

# ------------------------------- Hammer Handover API ------------------------------
class FrankaHandoverApi(ApiBase):
    """Robot control API for two-arm hammer handover task.

    Functions:
      - get_object_pose(object_name: str, return_bbox_extent: bool = False) -> (position: np.ndarray, quaternion_wxyz: np.ndarray, bbox_extent: np.ndarray | None)
      - goto_pose_arm0(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - goto_pose_arm1(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper_arm0() -> None
      - open_gripper_arm1() -> None
      - close_gripper_arm0() -> None
      - close_gripper_arm1() -> None
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

        self._vtf = vtf
        self.cfg_0 = None
        self.cfg_1 = None

        print("init franka handover api")
        self.sam3_seg_fn = init_sam3()
        print("init sam3 seg fn")
        self.grasp_net_plan_fn = init_contact_graspnet()
        print("init grasp net plan fn")
        self.ik_solve_fn = init_pyroki()
        print("init pyroki server")

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            # "sample_grasp_pose": self.sample_grasp_pose,
            # "get_arm0_gripper_pose": self.get_arm0_gripper_pose,
            # "get_arm1_gripper_pose": self.get_arm1_gripper_pose,
            "goto_pose_arm0": self.goto_pose_arm0,
            "goto_pose_arm1": self.goto_pose_arm1,
            "open_gripper_arm0": self.open_gripper_arm0,
            "open_gripper_arm1": self.open_gripper_arm1,
            "close_gripper_arm0": self.close_gripper_arm0,
            "close_gripper_arm1": self.close_gripper_arm1,
        }
    
    def _save_segmentation_debug(self, segmentation: np.ndarray, path: pathlib.Path) -> None:
        save_segmentation_debug(segmentation, path)

    def _get_segmentation_map(
        self, obs: dict[str, Any], rgb: np.ndarray, box: list[float] = None
    ) -> np.ndarray:
        return build_segmentation_map_from_sam2(
            self.sam2_seg_fn, rgb, obs['agentview']["images"], box=box
        )

    def _compute_bbox_indices(
        self, box: list[float], shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        return compute_bbox_indices(box, shape)

    def _select_instance_from_box(
        self, segmentation: np.ndarray, box: list[float]
    ) -> tuple[int, np.ndarray]:
        return select_instance_from_box(segmentation, box)

    def get_object_pose(self, object_name: str, return_bbox_extent: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        Coordinates and quaternions are in robot0's base frame.
        The quaternion from get_object_pose may be unreliable, so disregard it.

        Args:
            object_name: The name of the object to get the pose of.
            return_bbox_extent: Whether to return the extent of the oriented bounding box (oriented by quaternion_wxyz). Default is False.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) XYZ in meters (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        start_time = time.time()
        obs = self._env.get_observation()

        rgb_imgs = obs_get_rgb(obs)
        assert len(rgb_imgs.keys()) > 0, "No RGB images in obs"
        camera_name = 'agentview'
        rgb = rgb_imgs[camera_name]

        # Use SAM3 text-based segmentation
        sam3_results = self.sam3_seg_fn(rgb, text_prompt=object_name)
        
        if len(sam3_results) == 0:
            raise RuntimeError("SAM3 returned no results for object: " + object_name)
        
        # Get the best result (highest score)
        best_result = max(sam3_results, key=lambda x: x.get("score", 0.0))
        mask = best_result["mask"]
        
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask, dtype=bool)
        
        depth = obs[camera_name]["images"]["depth"]
        
        # Debug saves
        depth_img = depth_to_rgb(depth[:, :, 0])
        Image.fromarray(depth_img).save("depth_image.jpg")
        
        # Create segmentation map from SAM3 mask
        height, width = rgb.shape[:2]
        if mask.shape != (height, width):
            mask = mask.reshape(height, width)
        
        segmentation = mask.astype(np.int32)[:, :, None] if mask.ndim == 2 else mask.astype(np.int32)
        queried_instance_idx = 1
        
        self._save_segmentation_debug(segmentation, pathlib.Path("segmentation_image.jpg"))

        binary_map_nan_is_zero = (~np.isnan(depth[:, :, 0])).astype(int)

        idxs = np.where(
            segmentation.flatten()[binary_map_nan_is_zero.flatten().astype(bool)]
            == queried_instance_idx
        )

        points = depth_to_pointcloud(depth[:, :, 0], obs[camera_name]["intrinsics"])[idxs]

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)

        obb = o3d_points.get_oriented_bounding_box()

        self._env.cube_center = obb.center
        self._env.cube_rot = obb.R

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs[camera_name]["pose"][3:]),
            translation=obs[camera_name]["pose"][:3],
        )
        obb_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R), translation=obb.center
        )
        # transform from camera frame to robot0 base frame
        obb_tf_robot0 = cam_extr_tf @ obb_tf

        print(f"get_object_pose in {time.time() - start_time} seconds")
        
        if return_bbox_extent:
            return obb_tf_robot0.wxyz_xyz[-3:], obb_tf_robot0.wxyz_xyz[:4], obb.extent
        else:
            return obb_tf_robot0.wxyz_xyz[-3:], obb_tf_robot0.wxyz_xyz[:4], None

    def get_arm0_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
         """Get the pose of the gripper for arm 0. Position is offset from the finger contact center.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
         obs = self._env.get_observation()
         if "robot0_cartesian_pos" not in obs:
             raise ValueError("Environment does not provide robot0_cartesian_pos. Make sure you're using a hammer handover environment.")
         return obs["robot0_cartesian_pos"][:3], obs["robot0_cartesian_pos"][3:7]
    
    def get_arm1_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
         """Get the pose of the gripper for arm 1. Position is offset from the finger contact center. Uses robot0's base frame. 

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
         obs = self._env.get_observation()
         if "robot1_cartesian_pos" not in obs:
             raise ValueError("Environment does not provide robot1_cartesian_pos. Make sure you're using a hammer handover environment.")         
         return obs["robot1_cartesian_pos"][:3], obs["robot1_cartesian_pos"][3:7]

    def goto_pose_arm0(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 0 (robot0).
        Position and quaternion are in robot0's base frame.
        There is no need to call a second goto_pose_arm0 with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters, in robot0's base frame.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        self.cfg_0 = None # TODO remove after debug

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if (z_approach != 0.0):  # If z_approach is not 0.0, approach the object from above by z_approach meters
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            self.cfg_0 = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                prev_cfg=self.cfg_0,
            )
            joints_z_offset = np.asarray(self.cfg_0[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking(joints_z_offset)

        self.cfg_0 = self.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            prev_cfg=self.cfg_0,
        )
        joints = np.asarray(self.cfg_0[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper_arm0(self) -> None:
        """Open gripper fully for Arm 0 (robot0).

        Args:
            None
        """
        _open_gripper(self._env, steps=40)

    def close_gripper_arm0(self) -> None:
        """Close gripper fully for Arm 0 (robot0).

        Args:
            None
        """
        _close_gripper(self._env, steps=60)

    def goto_pose_arm1(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 1 (robot1).
        Position and quaternion are in robot0's base frame.
        The function automatically transforms coordinates from robot0's base frame to robot1's base frame.
        There is no need to call a second goto_pose_arm1 with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters, in robot0's base frame (will be transformed to robot1's base frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        self.cfg_1 = None

        # Get base transforms from environment
        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(self._env, "base_link_wxyz_xyz_1"):
            raise RuntimeError("Environment does not provide base transforms. Make sure you're using a two-arm handover environment.")

        # Transform position and quaternion from robot0's base frame to robot1's base frame
        # Step 1: Transform from robot0 base frame to world frame
        pose_arm0_base = self._vtf.SE3.from_rotation_and_translation(
            rotation=self._vtf.SO3(wxyz=quaternion_wxyz),
            translation=position,
        )
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        pose_world = base0_transform @ pose_arm0_base

        # Step 2: Transform from world frame to robot1's base frame
        base1_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1)
        base1_transform_inv = base1_transform.inverse()
        pose_arm1_base = base1_transform_inv @ pose_world

        # Extract position and quaternion in robot1's base frame
        pos = np.asarray(pose_arm1_base.translation(), dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(pose_arm1_base.rotation().wxyz, dtype=np.float64).reshape(4)

        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if (z_approach != 0.0):  # If z_approach is not 0.0, approach the object from above by z_approach meters
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            self.cfg_1 = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                prev_cfg=self.cfg_1,
            )
            joints_z_offset = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking_arm1(joints_z_offset)

        self.cfg_1 = self.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            prev_cfg=self.cfg_1,
        )
        joints = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)

    def open_gripper_arm1(self) -> None:
        """Open gripper fully for Arm 1 (robot1).

        Args:
            None
        """
        _open_gripper_arm1(self._env, steps=40)

    def close_gripper_arm1(self) -> None:
        """Close gripper fully for Arm 1 (robot1).

        Args:
            None
        """
        _close_gripper_arm1(self._env, steps=60)

    def sample_grasp_pose(self, object_name: str, arm_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.
        Coordinates and quaternions are in robot0's base frame.

        Args:
            object_name: The name of the object to sample a grasp pose for.
            arm_name: The name of the arm ('arm0' or 'arm1') to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        start_time = time.time()
        obs = self._env.get_observation()

        rgb_imgs = obs_get_rgb(obs)
        assert len(rgb_imgs.keys()) > 0, "No RGB images in obs"
        camera_name = 'agentview'
        rgb = rgb_imgs[camera_name]

        # Use SAM3 text-based segmentation
        sam3_results = self.sam3_seg_fn(rgb, text_prompt=object_name)
        print(f"sam3 seg in {time.time() - start_time} seconds")

        if len(sam3_results) == 0:
            raise RuntimeError("SAM3 returned no results for object: " + object_name)
        
        # Get the best result (highest score)
        best_result = max(sam3_results, key=lambda x: x.get("score", 0.0))
        mask = best_result["mask"]
        
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask, dtype=bool)

        depth = obs[camera_name]["images"]["depth"]

        # Debug image saves
        depth_img = depth_to_rgb(depth[:, :, 0])
        depth_img_out = Image.fromarray(depth_img)
        depth_img_out.save("depth_image.jpg")

        # Create segmentation map from SAM3 mask
        height, width = rgb.shape[:2]
        if mask.shape != (height, width):
            mask = mask.reshape(height, width)
        
        segmentation = mask.astype(np.int32)[:, :, None] if mask.ndim == 2 else mask.astype(np.int32)
        queried_instance_idx = 1
        
        print(f"segmentation in {time.time() - start_time} seconds")

        self._save_segmentation_debug(segmentation, pathlib.Path("segmentation_image.jpg"))

        binary_map_nan_is_zero = (~np.isnan(depth[:, :, 0])).astype(int)

        idxs = np.where(
            segmentation.flatten()[binary_map_nan_is_zero.flatten().astype(bool)]
            == queried_instance_idx
        )
        points, color = depth_color_to_pointcloud(
            depth[:, :, 0], rgb, obs[camera_name]["intrinsics"]
        )

        self._env.cube_points = points[idxs]
        self._env.cube_color = color[idxs]
        
        print("segmentation shape:", segmentation.shape)
        print("segmentation[:,:,0] shape:", segmentation[:, :, 0].shape)
        self._env.grasp_sample, self._env.grasp_scores, self._env.grasp_contact_pts = (
            self.grasp_net_plan_fn(
                depth[:, :, 0],
                obs[camera_name]["intrinsics"],
                segmentation[:, :, 0],
                queried_instance_idx,
            )
        )

        if arm_name == 'arm0':
            arm_pos, _ = self.get_arm0_gripper_pose()
        elif arm_name == 'arm1':
            arm_pos, _ = self.get_arm1_gripper_pose()
        else:
            raise ValueError(f"Invalid arm_name: {arm_name}. Must be 'arm0' or 'arm1'.")

        best_grasp = None
        closest_distance = float('inf')
        for grasp in self._env.grasp_sample:
            current_grasp = vtf.SE3.from_matrix(grasp) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
            grasp_pos = current_grasp.translation()
            dist = np.linalg.norm(grasp_pos - arm_pos)

            if dist < closest_distance:
                closest_distance = dist
                best_grasp = current_grasp
        self._env.grasp_sample_tf = best_grasp

        self._env.grasp_sample_tf = vtf.SE3.from_matrix(
            self._env.grasp_sample[self._env.grasp_scores.argmax()]
        ) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs[camera_name]["pose"][3:]),
            translation=obs[camera_name]["pose"][:3],
        )
        grasp_sample_tf_robot0 = cam_extr_tf @ self._env.grasp_sample_tf
        
        # Store grasp pose for the specific arm for viser visualization
        if arm_name == "arm0":
            self._env.grasp_sample_tf_arm0 = grasp_sample_tf_robot0
        elif arm_name == "arm1":
            self._env.grasp_sample_tf_arm1 = grasp_sample_tf_robot0
        
        print(f"sample_grasp_pose in {time.time() - start_time} seconds")
        return grasp_sample_tf_robot0.wxyz_xyz[-3:], grasp_sample_tf_robot0.wxyz_xyz[:4]


def _draw_boxes(
    rgb: np.ndarray, boxes: list[list[float]], labels: list[str], scores: list[float] | None = None
) -> Image.Image:
    return draw_boxes(rgb, boxes, labels, scores)

