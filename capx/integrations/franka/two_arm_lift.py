import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki_context import get_pyroki_context
from capx.integrations.vision.sam2 import init_sam2
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)


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


class FrankaTwoArmLiftApi(ApiBase):
    """Robot control API for two-arm lift task using vision-based perception (Non-privileged).

    Functions:
      - get_handle0_pos() -> bbox_center: np.ndarray:
      - get_handle1_pos() -> bbox_center: np.ndarray:
      - get_arm0_gripper_pose() -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - get_arm1_gripper_pose() -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - update_handle0_pose_viser(position: np.ndarray, quaternion_wxyz: np.ndarray) -> None
      - update_handle1_pose_viser(position: np.ndarray, quaternion_wxyz: np.ndarray) -> None
      - update_handle0_bbox_center_viser(position: np.ndarray) -> None
      - update_handle1_bbox_center_viser(position: np.ndarray) -> None
      - goto_pose_arm0(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper_arm0() -> None
      - close_gripper_arm0() -> None
      - goto_pose_arm1(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper_arm1() -> None
      - close_gripper_arm1() -> None
      - goto_pose_both(position0: np.ndarray, quaternion_wxyz0: np.ndarray, position1: np.ndarray, quaternion_wxyz1: np.ndarray, z_approach: float = 0.0) -> None
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        import viser.transforms as vtf_mod  # type: ignore

        from capx.integrations.motion import pyroki_snippets as pks  # type: ignore

        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._target_link_name = ctx.target_link_name
        self._pks = pks
        self._vtf = vtf_mod
        self.cfg = None
        # For Arm 1 (robot1), use same robot model but different config
        self.cfg_1 = None

        self._debug = False

        # Vision initialization
        # NOTE: OWL-ViT is loaded locally per worker process (no server option available),
        # which can cause CUDA OOM with many workers. Consider reducing num_workers if OOM occurs.
        # SAM2 and Contact GraspNet use server-based inference and don't load models locally.
        print("init franka two arm lift api (vision)")
        self.owl_vit_det_fn = init_owlvit(device="cuda")
        print("init owlvit det fn")
        self.sam2_seg_fn = init_sam2(device="cuda")
        print("init sam2 seg fn")
        self.grasp_net_plan_fn = init_contact_graspnet(device="cuda")
        print("init grasp net plan fn")

    def functions(self) -> dict[str, Any]:
        return {
            # "get_pot_pose": self.get_pot_pose,
            "get_handle0_pos": self.get_handle0_pos,
            "get_handle1_pos": self.get_handle1_pos,
            "get_arm0_gripper_pose": self.get_arm0_gripper_pose,
            "get_arm1_gripper_pose": self.get_arm1_gripper_pose,
            # "update_handle0_pose_viser": self.update_handle0_pose_viser,
            # "update_handle1_pose_viser": self.update_handle1_pose_viser,
            # "update_handle0_bbox_center_viser": self.update_handle0_bbox_center_viser,
            # "update_handle1_bbox_center_viser": self.update_handle1_bbox_center_viser,
            "goto_pose_arm0": self.goto_pose_arm0,
            "open_gripper_arm0": self.open_gripper_arm0,
            "close_gripper_arm0": self.close_gripper_arm0,
            "goto_pose_arm1": self.goto_pose_arm1,
            "open_gripper_arm1": self.open_gripper_arm1,
            "close_gripper_arm1": self.close_gripper_arm1,
            "goto_pose_both": self.goto_pose_both,
        }

    # ----------------------- Helper Methods from FrankaControlApi -----------------------

    def _get_segmentation_map(
        self,
        obs: dict[str, Any],
        rgb: np.ndarray,
        box: list[float] = None,
        text_prompt: str = "object",
    ) -> np.ndarray:
        # Use agentview which is the main camera in RobosuiteTwoArmLiftEnv
        images = obs["agentview"]["images"]
        segmentation = images.get("segmentation")
        if segmentation is not None:
            if segmentation.ndim == 2:
                segmentation = segmentation[..., None]
            return segmentation.astype(np.int32, copy=False)

        print("Running SAM2 segmentation with box:", box)

        # SAM2 uses box parameter and max_masks
        max_masks = 10
        masks = self.sam2_seg_fn(rgb, box=box, max_masks=max_masks)

        if len(masks) == 0:
            raise RuntimeError("SAM2 returned no masks while attempting to segment scene.")

        if box is not None:
            # Just use mask with the highest score
            max_score = -1
            max_idx = -1
            for idx, entry in enumerate(masks):
                score = entry.get("score")
                if score is not None and score > max_score:
                    max_score = score
                    max_idx = idx
            if max_idx >= 0:
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
            masks = self.sam2_seg_fn(rgb, box=None, max_masks=max_masks)

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
            # Fallback: If no overlap found but segmentation map has only one instance,
            # use that instance directly (common when box prompt was used and mask is slightly offset)
            if segmentation.max() == 1:
                return 1, seg_crop
            raise RuntimeError("No segmented instance overlaps detection bounding box.")
        unique_vals = unique_vals[valid_mask]
        counts = counts[valid_mask]
        queried_instance_idx = int(unique_vals[np.argmax(counts)])
        return queried_instance_idx, seg_crop

    def _get_handle_pose_with_graspnet(
        self, object_name: str, index: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper to get handle pose using vision detection + GraspNet for orientation.

        Args:
            object_name: (str) Name of the object to get the pose for.
            index: (int) Index of the object to get the pose for.

        Returns:
            grasp_position: (3,) XYZ position of grasp pose in world coordinates
            grasp_quaternion_wxyz: (4,) WXYZ quaternion of grasp pose
            bbox_center_position: (3,) XYZ position of bounding box center in world coordinates
        """
        start_time = time.time()

        # Always get a fresh observation to avoid stale data from previous calls
        obs = self._env.get_observation()

        rbg_imgs = obs_get_rgb(obs)
        rgb_image = rbg_imgs["agentview"]
        rgb = rgb_image.copy()  # Make a copy to avoid any potential in-place modifications

        # OwlViT Detection - always use fresh RGB
        dets = self.owl_vit_det_fn(rgb, texts=[[object_name]])

        if self._debug:
            # save the rgb image
            raw_image = Image.fromarray(rgb_image.copy())
            raw_image.save(f"vision_det_original_{object_name}_{index}.jpg")

        if len(dets) == 0:
            raise ValueError(
                f"No detections for {object_name}; environment constraints or model mismatch"
            )

        # Sort detections by score descending
        dets.sort(key=lambda x: x["score"], reverse=True)

        candidate_limit = 5
        candidates = dets[:candidate_limit]

        if index >= len(candidates):
            raise ValueError(
                f"Requested index {index} for {object_name}, but only found {len(candidates)} candidates."
            )

        target_det = candidates[index]
        box = target_det["box"]

        if self._debug:
            img_out = _draw_boxes(rgb, [box], [target_det["label"]], scores=[target_det["score"]])
            img_out.save(f"vision_det_{object_name}_{index}.jpg")

        # Always get fresh depth data - ensure it's a numpy array copy
        depth_raw = obs["agentview"]["images"]["depth"]
        if isinstance(depth_raw, np.ndarray):
            depth = depth_raw.copy()
        else:
            depth = np.array(depth_raw)

        # Segmentation - always compute fresh segmentation
        # Pass object_name as text_prompt for SAM3
        segmentation = self._get_segmentation_map(obs, rgb, box=box, text_prompt=object_name)
        queried_instance_idx, seg_crop = self._select_instance_from_box(segmentation, box)

        # Calculate bbox center using actual 3D points from segmentation
        # This is more accurate than using a single pixel depth
        if depth.ndim == 3:
            depth_2d = depth[:, :, 0]
        else:
            depth_2d = depth

        # Get all points belonging to the segmented instance
        binary_map_nan_is_zero = (~np.isnan(depth_2d)).astype(int)
        seg_flat = (
            segmentation[:, :, 0].flatten() if segmentation.ndim == 3 else segmentation.flatten()
        )
        depth_flat = depth_2d.flatten()
        valid_mask = (binary_map_nan_is_zero.flatten().astype(bool)) & (
            seg_flat == queried_instance_idx
        )
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            raise ValueError(
                f"No valid depth points found for segmented instance {queried_instance_idx} of {object_name}"
            )

        # Convert all valid pixels to 3D points
        all_points_3d = depth_to_pointcloud(depth_2d, obs["agentview"]["intrinsics"])
        points_3d = all_points_3d[valid_indices]

        if len(points_3d) == 0:
            raise ValueError(f"No valid 3D points found for {object_name} at index {index}")

        # Compute center of 3D points (mean of all points)
        bbox_center_cam = np.mean(points_3d, axis=0)

        # Transform bbox center from camera to world coordinates
        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs["agentview"]["pose"][3:]),
            translation=obs["agentview"]["pose"][:3],
        )
        bbox_center_cam_tf = vtf.SE3.from_translation(bbox_center_cam)
        bbox_center_world_tf = cam_extr_tf @ bbox_center_cam_tf
        bbox_center_world = bbox_center_world_tf.translation()

        return None, None, bbox_center_world

    # ----------------------- Pose Getters -----------------------

    def get_pot_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the pose of the pot body using vision detection + GraspNet for orientation.

        Args:
            None
        Returns:
            position: (3,) XYZ position of grasp pose in world coordinates
            quaternion_wxyz: (4,) WXYZ quaternion of grasp pose
            bbox_center_position: (3,) XYZ position of bounding box center in world coordinates
        """
        # Use GraspNet to get pot orientation
        return self._get_handle_pose_with_graspnet("red box-like pot", index=0)

    def get_handle0_pos(self) -> np.ndarray:
        """Get the bounding box center position of handle 0 using vision detection.

        Args:
            None
        Returns:
            bbox_center: (3,) XYZ position of bounding box center in world coordinates
        """
        # Use vision detection to get handle bbox center
        _, _, bbox_center = self._get_handle_pose_with_graspnet("The green square frame", index=0)
        return bbox_center

    def get_handle1_pos(self) -> np.ndarray:
        """Get the bounding box center position of handle 1 using vision detection.

        Args:
            None
        Returns:
            bbox_center: (3,) XYZ position of bounding box center in world coordinates
        """
        # Use vision detection to get handle bbox center
        _, _, bbox_center = self._get_handle_pose_with_graspnet("The blue square", index=0)
        return bbox_center

    def get_arm0_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the gripper for arm 0.

        Args:
            None
        Returns:
            position: (3,) XYZ position in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()
        if "robot0_cartesian_pos" not in obs:
            raise ValueError("Environment does not provide robot0_cartesian_pos.")
        return obs["robot0_cartesian_pos"][:3], obs["robot0_cartesian_pos"][3:7]

    def get_arm1_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the gripper for arm 1.

        Args:
            None
        Returns:
            position: (3,) XYZ position in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()
        if "robot1_cartesian_pos" not in obs:
            raise ValueError("Environment does not provide robot1_cartesian_pos.")
        return obs["robot1_cartesian_pos"][:3], obs["robot1_cartesian_pos"][3:7]

    def update_handle0_pose_viser(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Update the handle0 pose visualization in viser.

        Args:
            position: (3,) XYZ position in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        Returns:
            None
        """
        if hasattr(self._env, "detected_poses"):
            self._env.detected_poses["handle0"] = (
                np.asarray(position, dtype=np.float64).reshape(3),
                np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4),
            )

    def update_handle1_pose_viser(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Update the handle1 pose visualization in viser.

        Args:
            position: (3,) XYZ position in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        Returns:
            None
        """
        if hasattr(self._env, "detected_poses"):
            self._env.detected_poses["handle1"] = (
                np.asarray(position, dtype=np.float64).reshape(3),
                np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4),
            )

    def update_handle0_bbox_center_viser(self, position: np.ndarray) -> None:
        """Update the handle0 bounding box center visualization in viser.

        Args:
            position: (3,) XYZ position in meters.
        Returns:
            None
        """
        if not hasattr(self._env, "detected_bbox_centers"):
            self._env.detected_bbox_centers = {}
        bbox_pos = np.asarray(position, dtype=np.float64).reshape(3)
        self._env.detected_bbox_centers["handle0"] = bbox_pos

        # Ensure viser handles are created and updated immediately
        try:
            if hasattr(self._env, "viser_server") and self._env.viser_server is not None:
                if hasattr(self._env, "_viser_init_check"):
                    self._env._viser_init_check()
                if (
                    hasattr(self._env, "det_handle0_bbox_center_handle")
                    and self._env.det_handle0_bbox_center_handle is not None
                ):
                    self._env.det_handle0_bbox_center_handle.position = bbox_pos
                    self._env.det_handle0_bbox_center_handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
                    print(f"Updated handle0 bbox center in viser: {bbox_pos}")
                else:
                    print("Warning: det_handle0_bbox_center_handle is None after _viser_init_check")
            else:
                print(
                    f"Warning: viser_server not available (hasattr: {hasattr(self._env, 'viser_server')})"
                )
        except Exception as e:
            print(f"Error updating handle0 bbox center in viser: {e}")
            import traceback

            traceback.print_exc()

    def update_handle1_bbox_center_viser(self, position: np.ndarray) -> None:
        """Update the handle1 bounding box center visualization in viser.

        Args:
            position: (3,) XYZ position in meters.
        Returns:
            None
        """
        if not hasattr(self._env, "detected_bbox_centers"):
            self._env.detected_bbox_centers = {}
        bbox_pos = np.asarray(position, dtype=np.float64).reshape(3)
        self._env.detected_bbox_centers["handle1"] = bbox_pos

        # Ensure viser handles are created and updated immediately
        try:
            if hasattr(self._env, "viser_server") and self._env.viser_server is not None:
                if hasattr(self._env, "_viser_init_check"):
                    self._env._viser_init_check()
                if (
                    hasattr(self._env, "det_handle1_bbox_center_handle")
                    and self._env.det_handle1_bbox_center_handle is not None
                ):
                    self._env.det_handle1_bbox_center_handle.position = bbox_pos
                    self._env.det_handle1_bbox_center_handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
                    print(f"Updated handle1 bbox center in viser: {bbox_pos}")
                else:
                    print("Warning: det_handle1_bbox_center_handle is None after _viser_init_check")
            else:
                print(
                    f"Warning: viser_server not available (hasattr: {hasattr(self._env, 'viser_server')})"
                )
        except Exception as e:
            print(f"Error updating handle1 bbox center in viser: {e}")
            import traceback

            traceback.print_exc()

    # ----------------------- Motion Control Methods (Same as Privileged) -----------------------

    def goto_pose_arm0(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 0 (robot0)
        Args:
            position: (3,) XYZ position in meters for arm 0.
            quaternion_wxyz: (4,) WXYZ unit quaternion for arm 0.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
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

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None:
            self.cfg = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
        else:
            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper_arm0(self) -> None:
        """Open gripper fully for Arm 0 (robot0).
        Args:
            None
        Returns:
            None
        """
        self._env._set_gripper(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper_arm0(self) -> None:
        """Close gripper fully for Arm 0 (robot0).
        Args:
            None
        Returns:
            None
        """
        self._env._set_gripper(0.0)
        for _ in range(60):
            self._env._step_once()

    def goto_pose_arm1(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 1 (robot1).
        Args:
            position: (3,) XYZ position in meters for arm 1.
            quaternion_wxyz: (4,) WXYZ unit quaternion for arm 1.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        if not hasattr(self._env, "move_to_joints_blocking_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")

        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(
            self._env, "base_link_wxyz_xyz_1"
        ):
            raise RuntimeError("Environment does not provide base transforms.")

        pose_arm0_base = self._vtf.SE3.from_rotation_and_translation(
            rotation=self._vtf.SO3(wxyz=quaternion_wxyz),
            translation=position,
        )
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        pose_world = base0_transform @ pose_arm0_base

        base1_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1)
        base1_transform_inv = base1_transform.inverse()
        pose_arm1_base = base1_transform_inv @ pose_world

        pos = np.asarray(pose_arm1_base.translation(), dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(pose_arm1_base.rotation().wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg_1 is None:
                self.cfg_1 = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
            else:
                self.cfg_1 = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg_1,
                )
            joints_z_offset = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking_arm1(joints_z_offset)

        if self.cfg_1 is None:
            self.cfg_1 = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
        else:
            self.cfg_1 = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg_1,
            )
        joints = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)

    def open_gripper_arm1(self) -> None:
        """Open gripper fully for Arm 1 (robot1).
        Args:
            None
        Returns:
            None
        """
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper_arm1(self) -> None:
        """Close gripper fully for Arm 1 (robot1).
        Args:
            None
        Returns:
            None
        """
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(0.0)
        for _ in range(60):
            self._env._step_once()

    def goto_pose_both(
        self,
        position0: np.ndarray,
        quaternion_wxyz0: np.ndarray,
        position1: np.ndarray,
        quaternion_wxyz1: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Go to pose using Inverse Kinematics for moving both arms simultaneously. Positions and quaternions are in robot0's base frame.
        Args:
            position0: (3,) XYZ position in meters for arm 0.
            quaternion_wxyz0: (4,) WXYZ unit quaternion for arm 0.
            position1: (3,) XYZ position in meters for arm 1.
            quaternion_wxyz1: (4,) WXYZ unit quaternion for arm 1.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        if not hasattr(self._env, "move_to_joints_blocking_both"):
            raise RuntimeError("Environment does not support simultaneous control")

        # Robot 0 setup
        pos0 = np.asarray(position0, dtype=np.float64).reshape(3)
        quat_wxyz0 = np.asarray(quaternion_wxyz0, dtype=np.float64).reshape(4)
        quat_xyzw0 = np.array(
            [quat_wxyz0[1], quat_wxyz0[2], quat_wxyz0[3], quat_wxyz0[0]], dtype=np.float64
        )
        rot0 = SciRotation.from_quat(quat_xyzw0)
        offset_pos0 = pos0 + rot0.apply(self._TCP_OFFSET)

        # Robot 1 setup
        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(
            self._env, "base_link_wxyz_xyz_1"
        ):
            raise RuntimeError("Environment does not provide base transforms.")

        pose_arm1_base_in_0 = self._vtf.SE3.from_rotation_and_translation(
            rotation=self._vtf.SO3(wxyz=quaternion_wxyz1),
            translation=position1,
        )
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        pose_world1 = base0_transform @ pose_arm1_base_in_0

        base1_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1)
        base1_transform_inv = base1_transform.inverse()
        pose_arm1_base = base1_transform_inv @ pose_world1

        pos1 = np.asarray(pose_arm1_base.translation(), dtype=np.float64).reshape(3)
        quat_wxyz1 = np.asarray(pose_arm1_base.rotation().wxyz, dtype=np.float64).reshape(4)
        quat_xyzw1 = np.array(
            [quat_wxyz1[1], quat_wxyz1[2], quat_wxyz1[3], quat_wxyz1[0]], dtype=np.float64
        )
        rot1 = SciRotation.from_quat(quat_xyzw1)
        offset_pos1 = pos1 + rot1.apply(self._TCP_OFFSET)

        # Approach Phase
        if z_approach != 0.0:
            z_offset_pos0 = offset_pos0 + rot0.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos0,
                    target_wxyz=quat_wxyz0,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos0,
                    target_wxyz=quat_wxyz0,
                    prev_cfg=self.cfg,
                )
            joints0_approach = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

            z_offset_pos1 = offset_pos1 + rot1.apply(np.array([0, 0, -z_approach]))

            if self.cfg_1 is None:
                self.cfg_1 = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos1,
                    target_wxyz=quat_wxyz1,
                )
            else:
                self.cfg_1 = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos1,
                    target_wxyz=quat_wxyz1,
                    prev_cfg=self.cfg_1,
                )
            joints1_approach = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking_both(joints0_approach, joints1_approach)

        # Target Phase
        if self.cfg is None:
            self.cfg = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos0,
                target_wxyz=quat_wxyz0,
            )
        else:
            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos0,
                target_wxyz=quat_wxyz0,
                prev_cfg=self.cfg,
            )
        joints0 = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

        if self.cfg_1 is None:
            self.cfg_1 = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos1,
                target_wxyz=quat_wxyz1,
            )
        else:
            self.cfg_1 = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos1,
                target_wxyz=quat_wxyz1,
                prev_cfg=self.cfg_1,
            )
        joints1 = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)

        self._env.move_to_joints_blocking_both(joints0, joints1)
