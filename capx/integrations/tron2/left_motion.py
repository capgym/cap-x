"""Shared TRON2 left-arm TCP movement for visual and privileged tasks."""

from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np

from capx.integrations.base_api import ApiBase

from .grasp_feasibility import cgn_approach_waypoints, plan_left_tcp_motion
from .servoj import move_left_servoj_path


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray | list[float]) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


class Tron2LeftMotionApi(ApiBase):
    """Execute left-arm TCP goals through the selected motion backend."""

    _motion_log_name = "TRON2"

    def get_left_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the left gripper fingertip (TCP) pose in base_Link frame.

        Returns:
            Tuple of (position (3,), quaternion_wxyz (4,)).
        """
        pose = self._env.client.get_move_pose()
        return pose.left_pos.copy(), pose.left_quat.copy()

    def _verify_left_tcp(self, target: np.ndarray, quat: np.ndarray, stage: str, backend: str) -> bool:
        pose = self._env.client.get_move_pose()
        distance = np.linalg.norm(pose.left_pos - target)
        angle = np.degrees(2 * np.arccos(np.clip(abs(np.dot(quat, pose.left_quat)), 0, 1)))
        reached = bool(distance <= 0.005 and angle <= 3)
        print(f"[{self._motion_log_name}] {stage} measured TCP error {distance * 1000:.2f} mm / {angle:.2f} deg "
              f"actual_pos={pose.left_pos.tolist()} actual_quat_wxyz={pose.left_quat.tolist()} "
              f"target_pos={target.tolist()} backend={backend} reached={reached}")
        return reached

    def _move_left_checked(
        self, stage: str, target: np.ndarray, quat: np.ndarray, rotation: np.ndarray, duration_sec: float, backend: str,
    ) -> bool:
        with self._env.motion_feedback(stage):
            if backend == "movep":
                result = self._env.move_arms_pose(
                    left_pos=target, left_quat_wxyz=quat, duration_sec=duration_sec, stage=stage,
                    report_motion_rejection=True,
                )
                sent = result.get("result") == "success"
            else:
                path, reason = plan_left_tcp_motion(self._env, target, rotation)
                if path is None:
                    print(f"[{self._motion_log_name}] WARNING: {stage} local planning rejected: {reason}", file=sys.stderr)
                    self._env.record_rejected_motion()
                    self._verify_left_tcp(target, quat, stage, backend)
                    return False
                if backend == "servoj":
                    stream = move_left_servoj_path(self._env, path, duration_sec)
                    print(f"[{self._motion_log_name}] SERVOJ_STREAM stage={stage} {json.dumps(stream, ensure_ascii=False)}")
                    sent = stream["completed"]
                else:
                    sent = self._env.move_left_joint_path(path, duration_sec)
            reached = self._verify_left_tcp(target, quat, stage, backend)
        if not sent:
            print(f"[{self._motion_log_name}] WARNING: {stage} {backend} execution did not complete.",
                  file=sys.stderr)
        elif not reached:
            print(f"[{self._motion_log_name}] WARNING: {stage} finished dispatch without reaching the requested TCP.",
                  file=sys.stderr)
        return sent and reached

    def goto_pose_left(
        self,
        pos: list[float] | np.ndarray,
        quat_wxyz: list[float] | np.ndarray | None = None,
        approach_offset: float = 0.0,
        duration_sec: float = 2.0,
        *,
        stage: str = "free-motion",
    ) -> bool:
        """Move the left TCP to a position and orientation in the base_Link frame.

        Args:
            pos: Target [x, y, z] in meters.
            quat_wxyz: Target orientation [w, x, y, z]. Omit to keep the measured orientation.
            approach_offset: Optional backoff in meters along the tool's +Z axis. A positive
                offset moves above the pre-grasp, to the pre-grasp, then to the target.
            duration_sec: Requested duration in seconds per stage.
            stage: Log label for a direct move, such as "lift".

        Returns:
            True when the motion reaches the target. Planning or controller rejection,
            or a missed endpoint, returns False and leaves the arm at its resulting pose.
        """
        backend = os.environ.get("CAPX_TRON2_LEFT_MOTION", "movej").lower()
        if backend not in ("movej", "movep", "servoj"):
            raise ValueError("CAPX_TRON2_LEFT_MOTION must be movej, movep, or servoj")
        target = np.asarray(pos, dtype=float).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError("The requested TCP position must be finite")
        approach_offset = float(approach_offset)
        if not np.isfinite(approach_offset) or approach_offset < 0:
            raise ValueError("Approach offset must be finite and nonnegative")
        duration_sec = float(duration_sec)
        if not np.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("Motion duration must be finite and positive")
        if not isinstance(stage, str) or not stage.strip() or not stage.isprintable():
            raise ValueError("Stage must be a nonempty, printable label")
        if quat_wxyz is None:
            quat_wxyz = self.get_left_ee_pose()[1]
        quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
        scale = np.max(np.abs(quat))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("The requested orientation must be a finite, nonzero WXYZ quaternion")
        quat = quat / scale
        quat /= np.linalg.norm(quat)
        rotation = _quat_wxyz_to_rotmat(quat)
        if approach_offset == 0:
            return self._move_left_checked(stage, target, quat, rotation, duration_sec, backend)

        with np.errstate(over="ignore"):
            waypoints = cgn_approach_waypoints(target, rotation, approach_offset)
        if not np.isfinite(waypoints).all():
            raise ValueError("Approach offset generates a nonfinite waypoint")
        for phase, waypoint in zip(("above", "pre-grasp", "contact"), waypoints, strict=True):
            if not self._move_left_checked(phase, waypoint, quat, rotation, duration_sec, backend):
                return False
        return True
