from typing import Any

import numpy as np
import viser.transforms as vtf

from capx.integrations.franka.nut_assembly_privileged import (
    FrankaControlNutAssemblyPrivilegedApi,
)
from capx.integrations.motion.pyroki import init_pyroki


class FrankaTwoArmPegInHolePrivilegedApi(FrankaControlNutAssemblyPrivilegedApi):
    """Exact-pose two-arm control API for RoboSuite TwoArmPegInHole."""

    def __init__(self, env) -> None:
        self._env = env
        self.ik_solve_fn = init_pyroki()
        self.cfg_0: np.ndarray | None = None
        self.cfg_1: np.ndarray | None = None

    def functions(self) -> dict[str, Any]:
        return {
            "compose_pose": self.compose_pose,
            "get_object_pose": self.get_object_pose,
            "relative_pose": self.relative_pose,
            "goto_pose_arm0": self.goto_pose_arm0,
            "goto_pose_arm1": self.goto_pose_arm1,
            "goto_home_joint_positions": self.goto_home_joint_positions,
        }

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return peg, hole, insertion target, or arm EEF pose in arm-0 base frame.

        Args:
            object_name: Description containing peg, hole, hole target, arm0, or arm1.

        Returns:
            Position XYZ and quaternion WXYZ.
        """
        normalized = object_name.lower().replace(" ", "")
        if "target" in normalized and "hole" in normalized:
            key = "hole_target"
        elif "arm0" in normalized or "robot0" in normalized:
            key = "arm0_eef"
        elif "arm1" in normalized or "robot1" in normalized:
            key = "arm1_eef"
        elif "peg" in normalized:
            key = "peg"
        elif "hole" in normalized:
            key = "hole"
        else:
            raise ValueError(f"Invalid peg-in-hole object name: {object_name}")
        pose = self._env.get_observation()["peg_in_hole_poses"][key]
        return pose[:3], pose[3:]

    def goto_pose_arm0(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> None:
        """Move arm 0 to a panda-hand pose expressed in arm-0 base coordinates."""
        self.cfg_0 = self.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quaternion_wxyz, position]),
            prev_cfg=self.cfg_0,
        )
        self._env.move_to_joints_blocking(np.asarray(self.cfg_0[:-1], dtype=np.float64))

    def goto_pose_arm1(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> None:
        """Move arm 1 to a panda-hand pose expressed in arm-0 base coordinates."""
        target_in_arm0 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=np.asarray(quaternion_wxyz, dtype=np.float64)),
            translation=np.asarray(position, dtype=np.float64),
        )
        target_world = vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0) @ target_in_arm0
        target_in_arm1 = (
            vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1).inverse() @ target_world
        )
        self.cfg_1 = self.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate(
                [target_in_arm1.rotation().wxyz, target_in_arm1.translation()]
            ),
            prev_cfg=self.cfg_1,
        )
        self._env.move_to_joints_blocking_arm1(
            np.asarray(self.cfg_1[:-1], dtype=np.float64)
        )

    def goto_home_joint_positions(self) -> None:
        """Return both arms to their reset joint configurations."""
        self._env.move_to_joints_blocking(self._env.home_joint_position_0)
        self._env.move_to_joints_blocking_arm1(self._env.home_joint_position_1)
        self.cfg_0 = None
        self.cfg_1 = None


__all__ = ["FrankaTwoArmPegInHolePrivilegedApi"]
