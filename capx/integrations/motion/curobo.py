from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from capx.utils.serve_utils import post_with_retries

DEFAULT_URL = "http://127.0.0.1:8117"


def init_curobo(
    server_url: str = DEFAULT_URL,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Return an IK solver callable that forwards requests to a cuRobo server.

    Same interface as ``init_pyroki()`` — returns a function with signature::

        ik_solve_fn(target_pose_wxyz_xyz, prev_cfg=None) -> joint_positions

    Args:
        server_url: Base URL of the cuRobo FastAPI server.

    Returns:
        IK solver function.
    """
    server_url = server_url.rstrip("/")

    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        payload = {
            "target_pose_wxyz_xyz": target_pose_wxyz_xyz.tolist(),
            "prev_cfg": prev_cfg.tolist() if prev_cfg is not None else None,
        }
        data = post_with_retries(f"{server_url}/ik", payload)
        return np.asarray(data["joint_positions"], dtype=np.float32)

    return ik_solve_fn


def init_curobo_trajopt(
    server_url: str = DEFAULT_URL,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a trajectory-optimisation callable that forwards to a cuRobo server.

    Same interface as ``init_pyroki_trajopt()`` — returns a function with signature::

        trajopt_plan_fn(start_pose_wxyz_xyz, end_pose_wxyz_xyz) -> waypoints

    Args:
        server_url: Base URL of the cuRobo FastAPI server.

    Returns:
        Trajectory planning function.
    """
    server_url = server_url.rstrip("/")

    def trajopt_plan_fn(
        start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray
    ) -> np.ndarray:
        payload = {
            "start_pose_wxyz_xyz": start_pose_wxyz_xyz.tolist(),
            "end_pose_wxyz_xyz": end_pose_wxyz_xyz.tolist(),
        }
        data = post_with_retries(f"{server_url}/plan", payload)
        return np.asarray(data["waypoints"], dtype=np.float32)

    return trajopt_plan_fn
