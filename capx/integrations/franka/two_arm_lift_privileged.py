from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase


# ------------------------------- Two Arm Lift API ------------------------------
class FrankaTwoArmLiftPrivilegedApi(ApiBase):
    """Robot control API for two-arm lift task.

    Functions:
      - get_handle0_pose() -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - get_handle1_pose() -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - get_arm0_gripper_pose() -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - get_arm1_gripper_pose() -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
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
        import viser.transforms as vtf  # type: ignore

        from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._target_link_name = ctx.target_link_name
        self._pks = pks
        self._vtf = vtf
        self.cfg = None
        # For Arm 1 (robot1), use same robot model but different config
        self.cfg_1 = None

    def functions(self) -> dict[str, Any]:
        return {
            # "get_pot_pose": self.get_pot_pose,
            "get_handle0_pos": self.get_handle0_pos,
            "get_handle1_pos": self.get_handle1_pos,
            "get_arm0_gripper_pose": self.get_arm0_gripper_pose,
            "get_arm1_gripper_pose": self.get_arm1_gripper_pose,
            "goto_pose_arm0": self.goto_pose_arm0,
            "open_gripper_arm0": self.open_gripper_arm0,
            "close_gripper_arm0": self.close_gripper_arm0,
            "goto_pose_arm1": self.goto_pose_arm1,
            "open_gripper_arm1": self.open_gripper_arm1,
            "close_gripper_arm1": self.close_gripper_arm1,
            "goto_pose_both": self.goto_pose_both,
        }

    def get_pot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the pot body.

        Args:
            None
        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()
        if "pot_poses" not in obs:
            raise ValueError(
                "Environment does not provide pot_poses. Make sure you're using a two-arm lift environment."
            )
        return obs["pot_poses"]["pot"][:3], obs["pot_poses"]["pot"][3:]

    def get_handle0_pos(self) -> np.ndarray:
        """Get the bounding box center position of handle 0 using vision detection.

        Args:
            None
        Returns:
            bbox_center: (3,) XYZ position of bounding box center in world coordinates
        """
        obs = self._env.get_observation()
        if "pot_poses" not in obs:
            raise ValueError(
                "Environment does not provide pot_poses. Make sure you're using a two-arm lift environment."
            )
        return obs["pot_poses"]["handle0"][:3]  # , obs["pot_poses"]["handle0"][3:]

    def get_handle1_pos(self) -> np.ndarray:
        """Get the bounding box center position of handle 1 using vision detection.

        Args:
            None
        Returns:
            bbox_center: (3,) XYZ position of bounding box center in world coordinates
        """
        obs = self._env.get_observation()
        if "pot_poses" not in obs:
            raise ValueError(
                "Environment does not provide pot_poses. Make sure you're using a two-arm lift environment."
            )
        return obs["pot_poses"]["handle1"][:3]  # , obs["pot_poses"]["handle1"][3:]

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
