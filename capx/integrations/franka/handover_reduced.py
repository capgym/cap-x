from typing import Any

import numpy as np
import PIL
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki import init_pyroki
from capx.integrations.vision.sam2 import init_sam2


# ------------------------------- Handover API ------------------------------
class FrankaHandoverApiReduced(ApiBase):
    """
    Robot control helpers for two-arm Franka handover task.
    All coordinates are in robot0's (arm0) base frame.
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, env: BaseEnv, tcp_offset: list[float] | None = [0.0, 0.0, -0.107]) -> None:
        super().__init__(env)
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        print("init franka handover api reduced")
        self.owl_vit_det_fn = init_owlvit(device="cuda")
        print("init owlvit det fn")
        self.sam2_seg_fn = init_sam2()
        print("init sam2 seg fn")
        self.ik_solve_fn = init_pyroki()
        self.cfg_0 = None
        self.cfg_1 = None

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "detect_object_owlvit": self.detect_object_owlvit,
            "segment_sam2": self.segment_sam2,
            "move_to_joints_arm0": self.move_to_joints_arm0,
            "move_to_joints_arm1": self.move_to_joints_arm1,
            "solve_ik_arm0": self.solve_ik_arm0,
            "solve_ik_arm1": self.solve_ik_arm1,
            "open_gripper_arm0": self.open_gripper_arm0,
            "close_gripper_arm0": self.close_gripper_arm0,
            "open_gripper_arm1": self.open_gripper_arm1,
            "close_gripper_arm1": self.close_gripper_arm1,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W, 1), dtype float32.
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_cartesian_pos"]: (7,) array with [x, y, z, w, x, y, z] for arm0 gripper pose in robot0 base frame.
                - ["robot1_cartesian_pos"]: (7,) array with [x, y, z, w, x, y, z] for arm1 gripper pose in robot0 base frame.
        """
        return self._env.get_observation()

    # --------------------------------------------------------------------- #
    # Vision models: OWL-ViT detection + SAM2 segmentation
    # --------------------------------------------------------------------- #
    def detect_object_owlvit(
        self,
        rgb: np.ndarray,
        text: str,
    ) -> list[dict[str, Any]]:
        """Run OWL-ViT open-vocabulary detection on a single RGB image.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
                This should typically come from:
                    rgb = obs["agentview"]["images"]["rgb"]
            text:
                Natural language text query for OWL-ViT.

        Returns:
            detections:
                A list of dictionaries, one per detected box. Each dict typically
                contains:

                  - "box":   [x1, y1, x2, y2] in pixel coordinates (float)
                  - "label": str, the text label that matched best
                  - "score": float, confidence score in [0, 1]

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]  # (H, W, 3)
            >>> dets = detect_object_owlvit(rgb, text="hammer")
            >>> if dets:
            ...     best = max(dets, key=lambda d: d["score"])
            ...     print(best["box"], best["label"], best["score"])
        """
        return self.owl_vit_det_fn(rgb, texts=[[text]])

    def segment_sam2(
        self,
        rgb: np.ndarray,
        box: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run SAM2 segmentation on an RGB image, optionally conditioned on a box.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            box:
                Optional bounding box [x1, y1, x2, y2] in pixel coordinates, float.
                If provided, SAM2 will segment primarily within this region.
                If None, SAM2 runs in global mode over the whole image.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool or uint8,
                              where True/1 means the pixel belongs to the instance.
                  - "score": float confidence score (if provided by SAM2).

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> dets = detect_object_owlvit(rgb, text="hammer")
            >>> best = max(dets, key=lambda d: d["score"])
            >>> box = best["box"]
            >>> masks = segment_sam2(rgb, box=box)
        """
        return self.sam2_seg_fn(rgb, box=box)

    # --------------------------------------------------------------------- #
    # IK / motion primitives
    # --------------------------------------------------------------------- #
    def solve_ik_arm0(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve inverse kinematics for arm0 (robot0) panda_hand link.
        
        All coordinates are in robot0's base frame.

        Args:
            position:
                Target position in robot0's base frame.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in robot0's base frame.
                Shape: (4,), [w, x, y, z], dtype float64.

        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles for the 7 DoF Franka arm.

        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity, wxyz
            >>> joints = solve_ik_arm0(target_pos, target_quat)
            >>> move_to_joints_arm0(joints)
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        cfg = self.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
        )
        joints = np.asarray(cfg[:-1], dtype=np.float64).reshape(7)
        return joints

    def solve_ik_arm1(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve inverse kinematics for arm1 (robot1) panda_hand link.
        
        All coordinates are in robot0's base frame. The function automatically
        transforms coordinates from robot0's base frame to robot1's base frame.

        Args:
            position:
                Target position in robot0's base frame (will be transformed to robot1's base frame).
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in robot0's base frame.
                Shape: (4,), [w, x, y, z], dtype float64.

        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles for the 7 DoF Franka arm.

        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])  # in robot0 frame
            >>> target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity, wxyz
            >>> joints = solve_ik_arm1(target_pos, target_quat)
            >>> move_to_joints_arm1(joints)
        """
        # Get base transforms from environment
        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(self._env, "base_link_wxyz_xyz_1"):
            raise RuntimeError("Environment does not provide base transforms. Make sure you're using a two-arm handover environment.")

        # Transform position and quaternion from robot0's base frame to robot1's base frame
        # Step 1: Transform from robot0 base frame to world frame
        pose_arm0_base = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=quaternion_wxyz),
            translation=position,
        )
        base0_transform = vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        pose_world = base0_transform @ pose_arm0_base

        # Step 2: Transform from world frame to robot1's base frame
        base1_transform = vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1)
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

        cfg = self.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
        )
        joints = np.asarray(cfg[:-1], dtype=np.float64).reshape(7)
        return joints

    def move_to_joints_arm0(self, joints: np.ndarray) -> None:
        """Move arm0 (robot0) to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm.
                Shape: (7,), dtype float64.

        Returns:
            None

        Example:
            >>> joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
            >>> move_to_joints_arm0(joints)
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm0(joints)

    def move_to_joints_arm1(self, joints: np.ndarray) -> None:
        """Move arm1 (robot1) to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm.
                Shape: (7,), dtype float64.

        Returns:
            None

        Example:
            >>> joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
            >>> move_to_joints_arm1(joints)
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)

    def open_gripper_arm0(self) -> None:
        """Open gripper fully for arm0 (robot0).

        Args:
            None
        """
        self._env._set_gripper(1.0)
        for _ in range(30):
            self._env._step_once()

    def close_gripper_arm0(self) -> None:
        """Close gripper fully for arm0 (robot0).

        Args:
            None
        """
        self._env._set_gripper(0.0)
        for _ in range(30):
            self._env._step_once()

    def open_gripper_arm1(self) -> None:
        """Open gripper fully for arm1 (robot1).

        Args:
            None
        """
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(1.0)
        for _ in range(30):
            self._env._step_once()

    def close_gripper_arm1(self) -> None:
        """Close gripper fully for arm1 (robot1).

        Args:
            None
        """
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(0.0)
        for _ in range(30):
            self._env._step_once()
