from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pyroki as pk  # type: ignore
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation, Slerp


def _install_lightweight_integration_namespaces() -> None:
    """Load motion helpers without executing the full API registration graph."""
    capx_root = Path(__file__).resolve().parents[1]
    namespaces = {
        "capx.integrations": capx_root / "integrations",
        "capx.integrations.motion": capx_root / "integrations" / "motion",
    }
    for name, path in namespaces.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module


_install_lightweight_integration_namespaces()

import capx.integrations.motion.pyroki_snippets as pks
from capx.integrations.motion.pyroki_context import get_pyroki_context


def slerp_quaternions(
    q_start: np.ndarray,  # wxyz format
    q_end: np.ndarray,  # wxyz format
    num_steps: int,
) -> np.ndarray:
    """SLERP interpolation between two quaternions.

    Args:
        q_start: Start quaternion in wxyz format
        q_end: End quaternion in wxyz format
        num_steps: Number of interpolation steps

    Returns:
        Array of shape (num_steps, 4) with interpolated quaternions in wxyz format
    """
    # scipy uses xyzw format, so convert
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])

    # Create slerp interpolator
    key_rots = Rotation.concatenate([r_start, r_end])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)

    # Convert back to wxyz format
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.column_stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
    )
    return quats_wxyz


def plan_trajectory_linear_ik(
    robot: pk.Robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    num_waypoints: int = 25,
    use_prev_cfg: bool = True,
    jump_threshold: float = 0.5,
) -> np.ndarray:
    """Plan a trajectory by linear interpolation + IK at each waypoint.

    Args:
        robot: PyRoKi robot model
        target_link_name: Name of the end-effector link
        start_pos: Start position (3,)
        start_wxyz: Start orientation quaternion in wxyz format (4,)
        end_pos: End position (3,)
        end_wxyz: End orientation quaternion in wxyz format (4,)
        num_waypoints: Number of waypoints in the trajectory
        use_prev_cfg: If True, use previous IK solution to bias next solve
        jump_threshold: Max allowed joint change (radians) between waypoints before warning

    Returns:
        Array of shape (num_waypoints, num_joints) with joint configurations
    """
    # Linear interpolation for positions
    positions = np.linspace(start_pos, end_pos, num_waypoints)

    # SLERP for orientations
    orientations = slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    # Solve IK for each waypoint
    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations)):
        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(15):
                cfg = pks.solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                print(
                    f"Error: {np.linalg.norm(cfg - prev_cfg)}",
                    np.allclose(cfg, prev_cfg, atol=1e-3),
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
        else:
            cfg = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.array(cfg)

        # Check for large joint jumps
        if prev_cfg is not None:
            joint_diff = np.abs(cfg - prev_cfg)
            large_jumps = np.where(joint_diff > jump_threshold)[0]
            if len(large_jumps) > 0:
                for joint_idx in large_jumps:
                    jump_warnings.append(
                        f"  Waypoint {i}: joint {joint_idx} jumped {np.degrees(joint_diff[joint_idx]):.1f}° "
                        f"({joint_diff[joint_idx]:.3f} rad)"
                    )

        trajectory.append(cfg)
        prev_cfg = cfg

    # Print warnings summary
    if jump_warnings:
        print(
            f"\nWARNING: {len(jump_warnings)} large joint jump(s) detected (threshold: {np.degrees(jump_threshold):.1f}°):"
        )
        for warning in jump_warnings:
            print(warning)
        print()

    return np.array(trajectory)


# =====================================================
# Logging and FastAPI app
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroki_server")

app = FastAPI()

_ROBOT = None
_ROBOT_COLL = None
_TARGET_LINK = None
_FK_POSITION_TOLERANCE = 0.03
_FK_ROTATION_TOLERANCE = 0.75


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking CPU-bound function without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# =====================================================
# Pydantic Models
# =====================================================


class IkRequest(BaseModel):
    """IK request with optional prev_cfg for vel-cost IK."""

    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    prev_cfg: list[float] | None = None  # optional


class IkResponse(BaseModel):
    joint_positions: list[float]


class ObstacleEntry(BaseModel):
    type: str
    point: list[float] | None = None
    normal: list[float] | None = None
    center: list[float] | None = None
    radius: float | None = None
    position: list[float] | None = None
    height: float | None = None
    extent: list[float] | None = None  # for box obstacles


class PlanRequest(BaseModel):
    start_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    end_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    obstacles: list[dict] | None = None  # optional list of ObstacleEntry
    timesteps: int = 20
    dt: float = 0.02


class PlanResponse(BaseModel):
    waypoints: list[list[float]]
    dt: float


# =====================================================
# INTERNAL HELPERS
# =====================================================


def _build_world_coll(obstacles: list[dict[str, Any]] | None):
    if obstacles is None:
        return []
    world = []
    for obj in obstacles:
        t = obj.get("type")
        if t == "halfspace":
            p = np.array(obj["point"], dtype=np.float64)
            n = np.array(obj["normal"], dtype=np.float64)
            world.append(pk.collision.HalfSpace.from_point_and_normal(p, n))
        elif t == "sphere":
            c = np.array(obj["center"], dtype=np.float64)
            r = float(obj["radius"])
            world.append(pk.collision.Sphere.from_center_and_radius(c, np.array([r])))
        elif t == "capsule":
            pos = np.array(obj["position"], dtype=np.float64)
            rad = float(obj["radius"])
            h = float(obj["height"])
            world.append(
                pk.collision.Capsule.from_radius_height(
                    position=pos,
                    radius=np.array([rad]),
                    height=np.array([h]),
                )
            )
        elif t == "box":
            extent = np.array(obj["extent"], dtype=np.float64)
            pos = np.array(obj.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            world.append(
                pk.collision.Box.from_extent(
                    extent=extent,
                    position=pos,
                )
            )
        else:
            logger.warning(f"Unknown obstacle type '{t}', ignoring")
    return world


# =====================================================
# SERVER INIT — Load Pyroki Only Once
# =====================================================
def set_min_distance_from_limits(urdf: yourdfpy.URDF, min_distance_from_limits: float = 0.15) -> yourdfpy.URDF:
    """
    Set the minimum distance from limits for the robot.
    min_distance_from_limits: float in radians
    """
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower = joint.limit.lower + min_distance_from_limits
                joint.limit.upper = joint.limit.upper - min_distance_from_limits
    return urdf


def init_pyroki_server(
    robot_urdf_name: str = "panda_description",
    target_link_name: str = "panda_hand",
    root_link_name: str | None = None,
    fk_position_tolerance: float = 0.03,
    fk_rotation_tolerance: float = 0.75,
):
    global _ROBOT, _ROBOT_COLL, _TARGET_LINK, _FK_POSITION_TOLERANCE, _FK_ROTATION_TOLERANCE

    logger.info(f"Loading robot URDF '{robot_urdf_name}' with Pyroki...")
    ctx = get_pyroki_context(
        robot_urdf_name,
        target_link_name=target_link_name,
        root_link_name=root_link_name,
    )
    _ROBOT = ctx.robot
    _ROBOT_COLL = ctx.robot_coll
    _TARGET_LINK = target_link_name
    _FK_POSITION_TOLERANCE = float(fk_position_tolerance)
    _FK_ROTATION_TOLERANCE = float(fk_rotation_tolerance)

    logger.info(
        "PyRoki robot ready: root_link=%s target_link=%s actuated_joints=%s",
        root_link_name or "<urdf-root>",
        target_link_name,
        tuple(_ROBOT.joints.actuated_names),
    )
    logger.info(
        "PyRoki FK validation tolerances: position=%.4fm rotation=%.4frad",
        _FK_POSITION_TOLERANCE,
        _FK_ROTATION_TOLERANCE,
    )
    logger.info("PyRoki loaded and ready!")


# =====================================================
# ROUTES
# =====================================================



def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> Rotation:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid quaternion for FK residual check: {quat.tolist()}")
    quat = quat / norm
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])


def _ik_fk_residual(joints: list[float], target_pose_wxyz_xyz: np.ndarray) -> tuple[float, float]:
    target_link_index = _ROBOT.links.names.index(_TARGET_LINK)
    fk_poses = np.asarray(_ROBOT.forward_kinematics(np.asarray(joints, dtype=np.float64)))
    actual_pose = np.asarray(fk_poses[target_link_index], dtype=np.float64).reshape(7)

    position_error = float(np.linalg.norm(actual_pose[-3:] - target_pose_wxyz_xyz[-3:]))
    target_rot = _quat_wxyz_to_rotation(target_pose_wxyz_xyz[:4])
    actual_rot = _quat_wxyz_to_rotation(actual_pose[:4])
    rotation_error = float((target_rot.inv() * actual_rot).magnitude())
    return position_error, rotation_error


def _validate_ik_solution(joints: list[float], target_pose_wxyz_xyz: np.ndarray) -> tuple[float, float]:
    position_error, rotation_error = _ik_fk_residual(joints, target_pose_wxyz_xyz)
    if (
        position_error > _FK_POSITION_TOLERANCE
        or rotation_error > _FK_ROTATION_TOLERANCE
    ):
        raise ValueError(
            "PyRoKi FK residual too high: "
            f"position={position_error:.4f}m (tol {_FK_POSITION_TOLERANCE:.4f}m), "
            f"rotation={rotation_error:.4f}rad (tol {_FK_ROTATION_TOLERANCE:.4f}rad)."
        )
    return position_error, rotation_error

def _do_solve_ik(target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None) -> list[float]:
    """Blocking IK solve (CPU-bound)."""
    if prev_cfg is None:
        q = pks.solve_ik(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
        )
    else:
        q = pks.solve_ik_vel_cost(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
            prev_cfg=prev_cfg,
            initial_cfg=prev_cfg,
        )
    return list(map(float, q))


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")

    target_pose_wxyz_xyz = np.array(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev_cfg = np.array(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None

    try:
        joints = await _run_in_thread(_do_solve_ik, target_pose_wxyz_xyz, prev_cfg)
        position_error, rotation_error = _validate_ik_solution(joints, target_pose_wxyz_xyz)
        logger.info(
            "IK FK residual: position=%.4fm rotation=%.4frad prev_cfg=%s",
            position_error,
            rotation_error,
            "yes" if prev_cfg is not None else "no",
        )
    except Exception as e:
        logger.exception("IK failed")
        raise HTTPException(500, f"IK solve failed: {e}")

    return IkResponse(joint_positions=joints)


def _do_plan_motion(req: PlanRequest) -> PlanResponse:
    """Blocking motion planning (CPU-bound)."""
    start_pose = np.array(req.start_pose_wxyz_xyz, dtype=np.float64)
    end_pose = np.array(req.end_pose_wxyz_xyz, dtype=np.float64)

    start_wxyz = start_pose[:4]
    start_position = start_pose[4:]
    end_wxyz = end_pose[:4]
    end_position = end_pose[4:]

    timesteps = req.timesteps
    dt = req.dt

    sol_traj = plan_trajectory_linear_ik(
        robot=_ROBOT,
        target_link_name=_TARGET_LINK,
        start_pos=start_position,
        start_wxyz=start_wxyz,
        end_pos=end_position,
        end_wxyz=end_wxyz,
        num_waypoints=timesteps,
    )
    sol_traj = np.asarray(sol_traj)

    return PlanResponse(
        waypoints=sol_traj.tolist(),
        dt=float(dt),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_motion(req: PlanRequest):
    if _ROBOT is None or _ROBOT_COLL is None:
        raise HTTPException(503, "Pyroki not initialized")

    try:
        return await _run_in_thread(_do_plan_motion, req)
    except Exception as e:
        logger.exception("Planning failure")
        raise HTTPException(500, f"Motion planning failed: {e}")


# =====================================================
# ENTRYPOINT
# =====================================================


def main(
    robot: str = "panda_description",
    target_link: str = "panda_hand",
    root_link: str | None = None,
    fk_position_tolerance: float = 0.03,
    fk_rotation_tolerance: float = 0.75,
    port: int = 8116,
    host: str = "127.0.0.1",
):
    init_pyroki_server(
        robot_urdf_name=robot,
        target_link_name=target_link,
        root_link_name=root_link,
        fk_position_tolerance=fk_position_tolerance,
        fk_rotation_tolerance=fk_rotation_tolerance,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
