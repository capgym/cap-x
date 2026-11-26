"""Privileged high-level control API for TRON2 in CaP-X."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from capx.integrations.base_api import ApiBase
from capx.integrations.tron2.official_client import HORIZONTAL_GRASP_QUAT_WXYZ


class Tron2ControlPrivilegedApi(ApiBase):
    """High-level TRON2 dual-arm skill API backed by official Tron2_controller."""

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "home_pose": self.home_pose,
            "get_red_cube_pos": self.get_red_cube_pos,
            "get_left_ee_pose": self.get_left_ee_pose,
            "goto_pose_left": self.goto_pose_left,
            "goto_pose_right": self.goto_pose_right,
            "open_gripper_left": self.open_gripper_left,
            "close_gripper_left": self.close_gripper_left,
            "open_gripper_right": self.open_gripper_right,
            "close_gripper_right": self.close_gripper_right,
        }

    def home_pose(self) -> None:
        """Move both arms to the safe raised ready pose and open both grippers."""
        self._env.move_home()

    def get_red_cube_pos(self) -> np.ndarray:
        """Get the red cube's 3D center position [x, y, z] (meters) in base_Link frame.

        Returns:
            np.ndarray of shape (3,): [x, y, z] in meters relative to base_Link.
        """
        obs = self._env.get_observation()
        return np.asarray(obs["red_cube_pos_base"], dtype=np.float64)

    def get_left_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the left gripper fingertip (TCP) pose in base_Link frame.

        Returns:
            Tuple of (position (3,), quaternion_wxyz (4,)).
        """
        pose = self._env.client.get_move_pose()
        return pose.left_pos.copy(), pose.left_quat.copy()

    def goto_pose_left(
        self,
        pos: list[float] | np.ndarray,
        quat_wxyz: list[float] | np.ndarray | None = None,
        duration_sec: float = 2.0,
    ) -> bool:
        """Move the left gripper fingertip (TCP) to a target pose in base_Link frame via official MoveP.

        Args:
            pos: Target [x, y, z] position in meters relative to base_Link.
            quat_wxyz: Target orientation [w, x, y, z]. Defaults to horizontal forward grasp.
            duration_sec: Planned motion duration in seconds.
        """
        quat = HORIZONTAL_GRASP_QUAT_WXYZ if quat_wxyz is None else quat_wxyz
        response = self._env.move_arms_pose(
            left_pos=pos, left_quat_wxyz=quat, duration_sec=duration_sec, report_joint_limit=True,
        )
        return response.get("result") == "success"

    def goto_pose_right(
        self,
        pos: list[float] | np.ndarray,
        quat_wxyz: list[float] | np.ndarray | None = None,
        duration_sec: float = 2.0,
    ) -> bool:
        """Move the right gripper fingertip (TCP) to a target pose in base_Link frame via official MoveP.

        Args:
            pos: Target [x, y, z] position in meters relative to base_Link.
            quat_wxyz: Target orientation [w, x, y, z]. Defaults to horizontal forward grasp.
            duration_sec: Planned motion duration in seconds.
        """
        quat = HORIZONTAL_GRASP_QUAT_WXYZ if quat_wxyz is None else quat_wxyz
        response = self._env.move_arms_pose(
            right_pos=pos, right_quat_wxyz=quat, duration_sec=duration_sec, report_joint_limit=True,
        )
        return response.get("result") == "success"

    def open_gripper_left(self) -> None:
        """Open the left parallel gripper fully (~7.5 cm)."""
        self._env.set_gripper("left", 1.0)

    def close_gripper_left(self) -> None:
        """Close the left parallel gripper smoothly to clamp an object."""
        self._env.set_gripper("left", 0.0)

    def open_gripper_right(self) -> None:
        """Open the right parallel gripper fully (~7.5 cm)."""
        self._env.set_gripper("right", 1.0)

    def close_gripper_right(self) -> None:
        """Close the right parallel gripper smoothly to clamp an object."""
        self._env.set_gripper("right", 0.0)
