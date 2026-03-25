"""cuRobo GPU-accelerated motion planning server.

Exposes IK and collision-aware motion planning for Franka Panda via FastAPI.
Uses NVIDIA cuRobo for GPU-accelerated trajectory optimization.

Usage:
    uv run --no-sync --active python -m capx.serving.launch_curobo_server --port 8117
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("curobo_server")

app = FastAPI(title="cuRobo Motion Planning Server")

_IK_SOLVER = None
_MOTION_GEN = None
_TENSOR_ARGS = None

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# =====================================================
# Pydantic Models
# =====================================================


class IkRequest(BaseModel):
    """IK request: solve for joint config given target pose."""

    target_pose_wxyz_xyz: list[float]  # length 7: [qw, qx, qy, qz, x, y, z]
    prev_cfg: list[float] | None = None  # optional initial joint config (7 DOF)


class IkResponse(BaseModel):
    joint_positions: list[float]


class CuboidObstacle(BaseModel):
    name: str
    dims: list[float]  # [x, y, z] dimensions
    pose: list[float]  # [x, y, z, qw, qx, qy, qz]


class PlanRequest(BaseModel):
    """Plan trajectory between two poses (matches pyroki client interface)."""

    start_pose_wxyz_xyz: list[float]  # length 7: [qw, qx, qy, qz, x, y, z]
    end_pose_wxyz_xyz: list[float]  # length 7: [qw, qx, qy, qz, x, y, z]
    obstacles: list[CuboidObstacle] | None = None
    max_attempts: int = 10
    timeout: float = 30.0
    enable_graph: bool = True


class PlanResponse(BaseModel):
    waypoints: list[list[float]]


class MotionPlanRequest(BaseModel):
    """Motion plan from start joint config to goal pose (joint-space start)."""

    start_joint_positions: list[float]  # 7 DOF joint config
    goal_pose_wxyz_xyz: list[float]  # length 7: [qw, qx, qy, qz, x, y, z]
    obstacles: list[CuboidObstacle] | None = None
    max_attempts: int = 10
    timeout: float = 30.0
    enable_graph: bool = True


class MotionPlanResponse(BaseModel):
    waypoints: list[list[float]]
    success: bool
    plan_time_ms: float
    path_length: float
    num_waypoints: int
    status: str | None = None


class HealthResponse(BaseModel):
    status: str
    gpu: str
    curobo_version: str


# =====================================================
# Initialization
# =====================================================


def _build_table_world():
    from curobo.geom.types import WorldConfig

    return WorldConfig.from_dict(
        {
            "cuboid": {
                "table": {
                    "dims": [2.0, 2.0, 0.1],
                    "pose": [0.0, 0.0, -0.05, 1, 0, 0, 0],
                }
            }
        }
    )


def init_curobo(
    robot_config: str = "franka.yml",
    ee_link: str = "panda_hand",
    use_cuda_graph: bool = True,
):
    global _IK_SOLVER, _MOTION_GEN, _TENSOR_ARGS

    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    _TENSOR_ARGS = TensorDeviceType()
    logger.info(f"Loading cuRobo with robot config '{robot_config}'...")

    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_config))["robot_cfg"]
    robot_data["kinematics"]["ee_link"] = ee_link
    robot_cfg = RobotConfig.from_dict(robot_data, tensor_args=_TENSOR_ARGS)

    world_cfg = _build_table_world()

    # IK solver
    logger.info("Initializing IK solver...")
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_seeds=32,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=_TENSOR_ARGS,
        use_cuda_graph=use_cuda_graph,
    )
    _IK_SOLVER = IKSolver(ik_config)
    logger.info("IK solver ready.")

    # Motion planner
    logger.info("Initializing motion planner...")
    mg_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=use_cuda_graph,
        collision_cache={"obb": 20},
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        num_graph_seeds=4,
        interpolation_dt=0.02,
        trajopt_dt=0.25,
        trajopt_tsteps=32,
        interpolation_steps=2000,
        collision_activation_distance=0.02,
    )
    _MOTION_GEN = MotionGen(mg_config)
    logger.info("Warming up motion planner (may take a minute)...")
    _MOTION_GEN.warmup(enable_graph=True, warmup_js_trajopt=False)
    logger.info("cuRobo motion planner ready!")


# =====================================================
# Blocking handlers (run in thread pool)
# =====================================================


def _do_solve_ik(req: IkRequest) -> IkResponse:
    from curobo.types.math import Pose

    pose = req.target_pose_wxyz_xyz
    goal = Pose(
        position=_TENSOR_ARGS.to_device([pose[4:]]),
        quaternion=_TENSOR_ARGS.to_device([pose[:4]]),
    )

    if req.prev_cfg is not None:
        from curobo.types.state import JointState

        seed = JointState.from_position(_TENSOR_ARGS.to_device([req.prev_cfg]))
        result = _IK_SOLVER.solve_single(goal, seed_config=seed)
    else:
        result = _IK_SOLVER.solve_single(goal)

    torch.cuda.synchronize()

    if result.success.item():
        joints = result.solution[result.success].squeeze().cpu().numpy().flatten().tolist()
    else:
        joints = result.solution.squeeze().cpu().numpy().flatten().tolist()

    return IkResponse(joint_positions=joints)


def _do_plan(req: PlanRequest) -> PlanResponse:
    from curobo.geom.types import WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    # Update world if obstacles provided
    if req.obstacles:
        world_dict = {"cuboid": {}}
        for obs in req.obstacles:
            world_dict["cuboid"][obs.name] = {"dims": obs.dims, "pose": obs.pose}
        _MOTION_GEN.world_coll_checker.clear_cache()
        _MOTION_GEN.update_world(WorldConfig.from_dict(world_dict))

    # First solve IK for start pose to get joint config
    start_pose = req.start_pose_wxyz_xyz
    start_goal = Pose(
        position=_TENSOR_ARGS.to_device([start_pose[4:]]),
        quaternion=_TENSOR_ARGS.to_device([start_pose[:4]]),
    )
    start_ik = _IK_SOLVER.solve_single(start_goal)
    if not start_ik.success.item():
        logger.warning("IK failed for start pose, using best attempt")
    start_q = start_ik.solution.squeeze()
    start_state = JointState.from_position(start_q.unsqueeze(0))

    # Goal pose
    end_pose = req.end_pose_wxyz_xyz
    goal_pose = Pose(
        position=_TENSOR_ARGS.to_device([end_pose[4:]]),
        quaternion=_TENSOR_ARGS.to_device([end_pose[:4]]),
    )

    plan_config = MotionGenPlanConfig(
        max_attempts=req.max_attempts,
        enable_graph=req.enable_graph,
        enable_graph_attempt=1 if req.enable_graph else None,
        timeout=req.timeout,
        enable_finetune_trajopt=True,
    )

    _MOTION_GEN.reset(reset_seed=False)

    result = _MOTION_GEN.plan_single(start_state, goal_pose, plan_config)
    torch.cuda.synchronize()

    if result.success.item():
        traj = result.get_interpolated_plan()
        waypoints = traj.position.cpu().numpy().tolist()
    else:
        logger.warning("Motion plan failed, returning empty waypoints")
        waypoints = []

    return PlanResponse(waypoints=waypoints)


def _do_motion_plan(req: MotionPlanRequest) -> MotionPlanResponse:
    from curobo.geom.types import WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    if req.obstacles:
        world_dict = {"cuboid": {}}
        for obs in req.obstacles:
            world_dict["cuboid"][obs.name] = {"dims": obs.dims, "pose": obs.pose}
        _MOTION_GEN.world_coll_checker.clear_cache()
        _MOTION_GEN.update_world(WorldConfig.from_dict(world_dict))

    start_state = JointState.from_position(
        _TENSOR_ARGS.to_device([req.start_joint_positions])
    )

    pose = req.goal_pose_wxyz_xyz
    goal_pose = Pose(
        position=_TENSOR_ARGS.to_device([pose[4:]]),
        quaternion=_TENSOR_ARGS.to_device([pose[:4]]),
    )

    plan_config = MotionGenPlanConfig(
        max_attempts=req.max_attempts,
        enable_graph=req.enable_graph,
        enable_graph_attempt=1 if req.enable_graph else None,
        timeout=req.timeout,
        enable_finetune_trajopt=True,
    )

    _MOTION_GEN.reset(reset_seed=False)

    st = time.time()
    result = _MOTION_GEN.plan_single(start_state, goal_pose, plan_config)
    torch.cuda.synchronize()
    plan_ms = (time.time() - st) * 1000

    success = result.success.item()
    status = str(result.status) if hasattr(result, "status") else None

    if success:
        traj = result.get_interpolated_plan()
        wp_np = traj.position.cpu().numpy()
        waypoints = wp_np.tolist()
        diffs = np.diff(wp_np, axis=0)
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        waypoints = []
        path_length = 0.0

    return MotionPlanResponse(
        waypoints=waypoints,
        success=success,
        plan_time_ms=plan_ms,
        path_length=path_length,
        num_waypoints=len(waypoints),
        status=status,
    )


# =====================================================
# Routes
# =====================================================


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest):
    if _IK_SOLVER is None:
        raise HTTPException(503, "cuRobo not initialized")
    if len(req.target_pose_wxyz_xyz) != 7:
        raise HTTPException(400, "target_pose_wxyz_xyz must have exactly 7 elements")
    try:
        return await _run_in_thread(_do_solve_ik, req)
    except Exception as e:
        logger.exception("IK solve failed")
        raise HTTPException(500, f"IK solve failed: {e}")


@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    if _MOTION_GEN is None or _IK_SOLVER is None:
        raise HTTPException(503, "cuRobo not initialized")
    if len(req.start_pose_wxyz_xyz) != 7:
        raise HTTPException(400, "start_pose_wxyz_xyz must have 7 elements")
    if len(req.end_pose_wxyz_xyz) != 7:
        raise HTTPException(400, "end_pose_wxyz_xyz must have 7 elements")
    try:
        return await _run_in_thread(_do_plan, req)
    except Exception as e:
        logger.exception("Motion planning failed")
        raise HTTPException(500, f"Motion planning failed: {e}")


@app.post("/motion_plan", response_model=MotionPlanResponse)
async def motion_plan(req: MotionPlanRequest):
    if _MOTION_GEN is None:
        raise HTTPException(503, "cuRobo not initialized")
    if len(req.start_joint_positions) != 7:
        raise HTTPException(400, "start_joint_positions must have 7 elements")
    if len(req.goal_pose_wxyz_xyz) != 7:
        raise HTTPException(400, "goal_pose_wxyz_xyz must have 7 elements")
    try:
        return await _run_in_thread(_do_motion_plan, req)
    except Exception as e:
        logger.exception("Motion planning failed")
        raise HTTPException(500, f"Motion planning failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    import curobo

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    ready = _IK_SOLVER is not None and _MOTION_GEN is not None
    return HealthResponse(
        status="ready" if ready else "initializing",
        gpu=gpu,
        curobo_version=curobo.__version__,
    )


# =====================================================
# Entrypoint
# =====================================================


def main(
    robot_config: str = "franka.yml",
    ee_link: str = "panda_hand",
    port: int = 8117,
    host: str = "127.0.0.1",
    no_cuda_graph: bool = False,
):
    init_curobo(
        robot_config=robot_config,
        ee_link=ee_link,
        use_cuda_graph=not no_cuda_graph,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="cuRobo Motion Planning Server")
    parser.add_argument("--robot-config", default="franka.yml")
    parser.add_argument("--ee-link", default="panda_hand")
    parser.add_argument("--port", type=int, default=8117)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    main(
        robot_config=args.robot_config,
        ee_link=args.ee_link,
        port=args.port,
        host=args.host,
        no_cuda_graph=args.no_cuda_graph,
    )
