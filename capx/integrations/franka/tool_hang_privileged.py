from typing import Any

import numpy as np

from capx.integrations.franka.nut_assembly_privileged import (
    FrankaControlNutAssemblyPrivilegedApi,
)


class FrankaControlToolHangPrivilegedApi(FrankaControlNutAssemblyPrivilegedApi):
    """Exact-pose control API for RoboSuite ToolHang."""

    def functions(self) -> dict[str, Any]:
        return {
            "compose_pose": self.compose_pose,
            "get_object_pose": self.get_object_pose,
            "relative_pose": self.relative_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "goto_home_joint_position": self.goto_home_joint_position,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def _pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        pose = self._env.get_observation()["tool_hang_poses"][key]
        return pose[:3], pose[3:]

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return an exact task-object pose in the robot base frame.

        Args:
            object_name: One of stand, stand mount, frame, frame grip, frame tip,
                frame hang site, tool, tool grip, or tool hole.

        Returns:
            Position XYZ and quaternion WXYZ.
        """
        normalized = object_name.lower()
        if "hole" in normalized and ("tool" in normalized or "wrench" in normalized):
            return self._pose("tool_hole")
        if "grip" in normalized and ("tool" in normalized or "wrench" in normalized):
            return self._pose("tool_grip")
        aliases = (
            (("stand", "mount"), "stand_mount"),
            (("frame", "grip"), "frame_grip"),
            (("frame", "tip"), "frame_tip"),
            (("frame", "hang"), "frame_hang"),
        )
        for words, key in aliases:
            if all(word in normalized for word in words):
                return self._pose(key)
        if "stand" in normalized:
            return self._pose("stand")
        if "frame" in normalized:
            return self._pose("frame")
        if "tool" in normalized or "wrench" in normalized:
            return self._pose("tool")
        raise ValueError(f"Invalid ToolHang object name: {object_name}")

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the documented frame-grip or tool-grip grasp pose.

        Args:
            object_name: Description containing frame or tool / wrench.

        Returns:
            Position XYZ and quaternion WXYZ in the robot base frame.
        """
        normalized = object_name.lower()
        if "frame" in normalized:
            return self._pose("frame_grip")
        if "tool" in normalized or "wrench" in normalized:
            return self._pose("tool_grip")
        raise ValueError(f"Invalid ToolHang grasp object name: {object_name}")


__all__ = ["FrankaControlToolHangPrivilegedApi"]
