from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase

class FrankaHandoverPrivilegedApi(ApiBase):
    """Robot control API for two-arm hammer handover task.

    Functions:
      - get_hammer_pose() -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
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
        # Lazy-import to keep startup light
        import viser.transforms as vtf  # type: ignore

        from capx.integrations.motion.pyroki import init_pyroki  # type: ignore

        self._vtf = vtf
        self.cfg_0 = None
        self.cfg_1 = None
        self.ik_solve_fn = init_pyroki()

    def functions(self) -> dict[str, Any]:
        return {
            "get_hammer_pose": self.get_hammer_pose,
            # "get_arm0_gripper_pose": self.get_arm0_gripper_pose,
            # "get_arm1_gripper_pose": self.get_arm1_gripper_pose,
            "goto_pose_arm0": self.goto_pose_arm0,
            "goto_pose_arm1": self.goto_pose_arm1,
            "open_gripper_arm0": self.open_gripper_arm0,
            "open_gripper_arm1": self.open_gripper_arm1,
            "close_gripper_arm0": self.close_gripper_arm0,
            "close_gripper_arm1": self.close_gripper_arm1,
        }
    
    def get_arm0_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the gripper for arm 0.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()
        if "robot0_cartesian_pos" not in obs:
            raise ValueError(
                "Environment does not provide robot0_cartesian_pos. Make sure you're using a hammer handover environment."
            )
        return obs["robot0_cartesian_pos"][:3], obs["robot0_cartesian_pos"][3:7]

    def get_arm1_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the gripper for arm 1.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()
        if "robot1_cartesian_pos" not in obs:
            raise ValueError(
                "Environment does not provide robot1_cartesian_pos. Make sure you're using a hammer handover environment."
            )

        return obs["robot1_cartesian_pos"][:3], obs["robot1_cartesian_pos"][3:7]

    def get_hammer_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the middle of the hammer handle. 
        The quaternion output may be unreliable.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()
        if "hammer_poses" not in obs:
            raise ValueError(
                "Environment does not provide hammer_poses. Make sure you're using a hammer handover environment."
            )
        return obs["hammer_poses"]["handle"][:3], obs["hammer_poses"]["handle"][3:]

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
        
        self.cfg_0 = None

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        
        if hasattr(self._env, "viser_server") and self._env.viser_server is not None:
            try:
                self._env._viser_init_check()
                if not hasattr(self._env, "target_frame_handle_arm0") or self._env.target_frame_handle_arm0 is None:
                    self._env.target_frame_handle_arm0 = self._env.viser_server.scene.add_frame(
                        "/target_arm0", axes_length=0.1, axes_radius=0.003
                    )
                self._env.target_frame_handle_arm0.position = pos
                self._env.target_frame_handle_arm0.wxyz = quat_wxyz
            except Exception:
                pass
        
        # Align with legacy env: apply TCP offset in end-effector frame
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
        self._env._set_gripper(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper_arm0(self) -> None:
        """Close gripper fully for Arm 0 (robot0).

        Args:
            None
        """
        self._env._set_gripper(0.0)
        for _ in range(60):
            self._env._step_once()

    def goto_pose_arm1(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 1 (robot1).
        Position and quaternion are in robot0's base frame (same as returned by get_hammer_pose).
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

        if not hasattr(self._env, "move_to_joints_blocking_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")

        # Get base transforms from environment
        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(
            self._env, "base_link_wxyz_xyz_1"
        ):
            raise RuntimeError(
                "Environment does not provide base transforms. Make sure you're using a two-arm handover environment."
            )

        # Transform position and quaternion from robot0's base frame to robot1's base frame
        # Step 1: Transform from robot0 base frame to world frame
        pose_arm0_base = self._vtf.SE3.from_rotation_and_translation(
            rotation=self._vtf.SO3(wxyz=quaternion_wxyz),
            translation=position,
        )
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        pose_world = base0_transform @ pose_arm0_base

        if hasattr(self._env, "viser_server") and self._env.viser_server is not None:
            try:
                self._env._viser_init_check()
                if not hasattr(self._env, "target_frame_handle_arm1") or self._env.target_frame_handle_arm1 is None:
                    self._env.target_frame_handle_arm1 = self._env.viser_server.scene.add_frame(
                        "/target_arm1", axes_length=0.1, axes_radius=0.003
                    )
                self._env.target_frame_handle_arm1.position = pose_arm0_base.translation()
                self._env.target_frame_handle_arm1.wxyz = pose_arm0_base.rotation().wxyz
            except Exception:
                pass

        # Step 2: Transform from world frame to robot1's base frame
        base1_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1)
        base1_transform_inv = base1_transform.inverse()
        pose_arm1_base = base1_transform_inv @ pose_world

        # Extract position and quaternion in robot1's base frame
        pos = np.asarray(pose_arm1_base.translation(), dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(pose_arm1_base.rotation().wxyz, dtype=np.float64).reshape(4)

        # Align with legacy env: apply TCP offset in end-effector frame
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
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper_arm1(self) -> None:
        """Close gripper fully for Arm 1 (robot1).

        Args:
            None
        """
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(0.0)
        for _ in range(60):
            self._env._step_once()
