from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

try:
    import pyroki as pk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pk = None  # type: ignore[assignment]

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_robot_description = None  # type: ignore[assignment]

"""Per-process cached PyRoKI context to avoid repeated JIT recompiles.

This module constructs and memoizes the heavy PyRoKI objects (robot model and
collision model). Reusing the same Python objects across environment instances
within a worker process prevents JAX/XLA from retracing and recompiling the IK
solver on every env spawn, which can exhaust memory.
"""


@dataclass(frozen=True)
class PyrokiContext:
    """Holds shared PyRoKI objects that should be reused within a process."""

    robot: Any
    robot_coll: Any
    target_link_name: str


@lru_cache(maxsize=8)
def get_pyroki_context(
    robot_urdf_or_name: str = "panda_description",
    *,
    target_link_name: str = "panda_hand",
) -> PyrokiContext:
    """Return a cached PyRoKI context for the given robot and target link.

    Args:
        robot_urdf_or_name: Filesystem path to a URDF, or a name that
            `robot_descriptions` can resolve (e.g., "panda_description").
        target_link_name: End-effector link name.

    Returns:
        PyrokiContext with constructed `pk.Robot` and `pk.collision.RobotCollision`.

    Notes:
        - The returned objects are cached per-process; callers must not mutate
          them in ways that affect JAX-pytree structure.
    """
    if pk is None:
        raise RuntimeError("pyroki not installed; install with robotics extras")
    if load_robot_description is None:
        raise RuntimeError("robot_descriptions not available; install robotics extras")

    if os.path.exists(robot_urdf_or_name):
        import yourdfpy as urdfpy  # type: ignore

        urdf = urdfpy.URDF.load(robot_urdf_or_name)
    else:
        urdf = load_robot_description(robot_urdf_or_name)

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    return PyrokiContext(robot=robot, robot_coll=robot_coll, target_link_name=target_link_name)


__all__ = ["PyrokiContext", "get_pyroki_context"]
