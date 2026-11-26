"""Single-world TRON2 tabletop simulation driven by the official controller.

High-level commands use the controller's WebSocket API. The same MuJoCo world
applies MROS motor commands, returns feedback, and simulates gripper contacts.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.tron2.motor_sim import MotorTransport, Tron2MotorSim
from capx.integrations.tron2.official_client import (
    DEFAULT_HOME_Q16,
    PROTOCOL_JOINT_NAMES,
    Tron2JointLimitRejection,
    Tron2MotionRejection,
    Tron2OfficialClient,
)

_LOG = logging.getLogger(__name__)
_DEFAULT_XML = (
    Path(__file__).resolve().parents[2]
    / "assets/tron2/DACH_TRON2A/xml/scene_table_cube.xml"
)

_R_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])
_TARGET_LIFT_M = 0.10

GRIPPER_OPEN_RAD = 0.72
GRIPPER_CLOSE_RAD = 0.0


class _TableContactFeedback:
    """Summarize solver-measured left-arm contact with the table during one motion."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.table_id = model.body("table").id
        root_id = model.body("proximal_pitch_L_Link").id
        self.arm_bodies = {root_id}
        for body in range(root_id + 1, model.nbody):
            if model.body_parentid[body] in self.arm_bodies:
                self.arm_bodies.add(body)
        self.force = np.empty(6)
        self.summary: dict[int, tuple[float, int]] | None = None
        self.loaded = False

    def sample(self, data: mujoco.MjData) -> None:
        if self.summary is None:
            return
        body_loads: dict[int, float] = {}
        for index, contact in enumerate(data.contact):
            if contact.efc_address < 0 or contact.geom1 < 0 or contact.geom2 < 0:
                continue
            a, b = self.model.geom_bodyid[contact.geom]
            if a != self.table_id and b != self.table_id:
                continue
            body = b if a == self.table_id else a
            if body in self.arm_bodies:
                mujoco.mj_contactForce(self.model, data, index, self.force)
                if self.force[0] > 0:
                    body_loads[body] = body_loads.get(body, 0.0) + float(self.force[0])
        self.loaded = bool(body_loads)
        for body, load in body_loads.items():
            peak_load, loaded_steps = self.summary.get(body, (0.0, 0))
            self.summary[body] = max(peak_load, load), loaded_steps + 1


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, rot.reshape(9))
    return q


class Tron2CubeLiftLowLevel(BaseEnv):
    """Simulate the controller's motors and tabletop in one MuJoCo world."""

    def __init__(
        self,
        *,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
        xml_path: str | Path | None = None,
        controller_host: str = "127.0.0.1",
        controller_port: int = 5000,
        accid: str = "DACH_TRON2A_001",
        primary_camera: str = "robot0_robotview",
        render_width: int = 640,
        render_height: int = 480,
        max_steps: int = 1000,
        seed: int | None = None,
        motor_command_timeout_sec: float = 0.0,
        motor_transport: MotorTransport | None = None,
    ) -> None:
        super().__init__()
        if controller_host != "127.0.0.1":
            raise ValueError("The TRON2 motor simulation requires a loopback controller")
        self.privileged = privileged
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self.max_steps = max_steps

        self.xml_path = Path(xml_path) if xml_path is not None else _DEFAULT_XML
        self.primary_camera = primary_camera
        self.save_camera_name = primary_camera
        self._render_width = int(render_width)
        self._render_height = int(render_height)
        self._rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # The motor thread owns the only stepped state. Task reads and renders
        # use a snapshot so the renderer never blocks the physics step.
        self._motor_data = mujoco.MjData(self.model)
        self.data = mujoco.MjData(self.model)

        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self._cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._cube_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_jnt_id])

        # Build joint-name -> (qpos_adr, dof_adr, act_id) table
        jnt_map: dict[str, tuple[int, int, int]] = {}
        for act_id in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[act_id, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if name:
                jnt_map[name] = (
                    int(self.model.jnt_qposadr[jnt_id]),
                    int(self.model.jnt_dofadr[jnt_id]),
                    act_id,
                )

        def _bind_gripper(jnt_name: str, kp: float, kd: float) -> tuple[int, int, int, float, float, float, float]:
            qpos_adr, dof_adr, act_id = jnt_map[jnt_name]
            lo, hi = self.model.actuator_ctrlrange[act_id]
            return (qpos_adr, dof_adr, act_id, kp, kd, float(lo), float(hi))

        self._grip_base_l = _bind_gripper("grasper_base_L_Joint", 12.0, 0.5)
        self._grip_base_r = _bind_gripper("grasper_base_R_Joint", 12.0, 0.5)
        self._grip_drv_l = _bind_gripper("grasper_L_drive_Joint", 4.0, 0.15)
        self._grip_drv_r = _bind_gripper("grasper_R_drive_Joint", 4.0, 0.15)

        # Official Tron2_controller WebSocket client (lazy-connected on reset/first command)
        self.client = Tron2OfficialClient(
            host=controller_host,
            port=controller_port,
            accid=accid,
        )
        self._connected = False

        # Control & episode state
        self._grip_l = GRIPPER_OPEN_RAD
        self._grip_r = GRIPPER_OPEN_RAD
        self._cube_init_world = np.array([0.52, 0.15, 0.875], dtype=np.float64)
        self._step_count = 0
        self._record_ticks = 0

        # Offscreen renderer (created lazily when needed)
        self._renderer: mujoco.Renderer | None = None
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._record_stride = 2  # approximately 20 frames per second during motion

        self._joint_qpos = np.array([self.model.joint(name).qposadr[0] for name in PROTOCOL_JOINT_NAMES])
        self._motor_data.qpos[self._joint_qpos] = DEFAULT_HOME_Q16
        self._reset_cube(self._motor_data)
        mujoco.mj_forward(self.model, self._motor_data)
        self._motor = Tron2MotorSim(
            self.model, self._motor_data,
            command_timeout_sec=motor_command_timeout_sec, transport=motor_transport,
        )
        self._physics_lock = threading.Lock()
        self._physics_stop = threading.Event()
        self._physics_thread: threading.Thread | None = None
        self._physics_error: Exception | None = None
        self._table_contact_feedback = _TableContactFeedback(self.model)
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_buffer = np.empty(mujoco.mj_stateSize(self.model, self._state_spec))

    # ----------------------- Controller & Physics -----------------------

    def start_motor_loop(self) -> None:
        """Publish initial motor state before the controller connects."""
        self._check_physics()
        if self._physics_thread is not None:
            return
        self._motor.start()
        self._physics_stop.clear()
        self._physics_thread = threading.Thread(target=self._physics_loop, name="tron2-motor-sim", daemon=True)
        self._physics_thread.start()
        atexit.register(self.close)

    def _check_physics(self) -> None:
        if self._physics_error is not None:
            raise RuntimeError("TRON2 motor simulation stopped") from self._physics_error

    def _physics_loop(self) -> None:
        dt = self.model.opt.timestep
        started = deadline = time.monotonic()
        waiting_for_control = True
        _LOG.info("TRON2 is parked at its initial pose while waiting for arm position control")
        try:
            while not self._physics_stop.is_set():
                if waiting_for_control and self._motor.has_arm_position_control():
                    waiting_for_control = False
                    _LOG.info("TRON2 arm position control enabled. Starting tabletop physics.")
                with self._physics_lock:
                    if waiting_for_control:
                        # Hold the initialized pose above the table until the
                        # controller starts sending arm position commands.
                        self._motor.publish_state_if_due(clock_time_sec=time.monotonic() - started)
                    else:
                        self._motor.apply_control()
                        self._apply_gripper_control(self._motor_data)
                        mujoco.mj_step(self.model, self._motor_data)
                        self._table_contact_feedback.sample(self._motor_data)
                        self._motor.publish_state_if_due()
                deadline += dt
                now = time.monotonic()
                if now - deadline > 0.05:
                    deadline = now
                if deadline > now:
                    self._physics_stop.wait(deadline - now)
        except Exception as exc:
            self._physics_error = exc
            self._physics_stop.set()

    def _ensure_connected(self) -> None:
        self.start_motor_loop()
        if not self._connected:
            self.client.connect()
            self._connected = True

    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self._render_height,
                width=self._render_width,
            )
        return self._renderer

    def _observation_data(self) -> mujoco.MjData:
        self._check_physics()
        with self._physics_lock:
            mujoco.mj_getState(self.model, self._motor_data, self._state_buffer, self._state_spec)
        mujoco.mj_setState(self.model, self.data, self._state_buffer, self._state_spec)
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def _read_arm_joints(self) -> np.ndarray:
        self._check_physics()
        with self._physics_lock:
            return self._motor_data.qpos[self._joint_qpos]

    def _apply_gripper_control(self, data: mujoco.MjData) -> None:
        for (qpos_adr, dof_adr, act_id, kp, kd, lo, hi), target in (
            (self._grip_base_l, 0.0),
            (self._grip_base_r, 0.0),
            (self._grip_drv_l, self._grip_l),
            (self._grip_drv_r, self._grip_r),
        ):
            tau = kp * (target - data.qpos[qpos_adr]) - kd * data.qvel[dof_adr] + data.qfrc_bias[dof_adr]
            data.ctrl[act_id] = float(np.clip(tau, lo, hi))

    def _wait_and_record(self, duration_sec: float) -> None:
        """Wait for the independent motor loop and record observation snapshots."""
        self._ensure_connected()
        end = time.monotonic() + float(duration_sec)
        while (remaining := end - time.monotonic()) > 0:
            self._check_physics()
            start = time.monotonic()
            self._record_ticks += 1
            if self._record_frames and self._record_ticks % self._record_stride == 0:
                self._record_frame()
            self._physics_stop.wait(max(0.0, min(remaining, 0.025) - (time.monotonic() - start)))
        self._check_physics()

    # ----------------------- High-Level Motion Primitives -----------------------

    @contextmanager
    def motion_feedback(self, stage: str) -> Iterator[None]:
        """Report table contact once per link after the action phase, outside the physics thread."""
        tracker = self._table_contact_feedback
        summary: dict[int, tuple[float, int]] = {}
        with self._physics_lock:
            previous, previous_loaded = tracker.summary, tracker.loaded
            tracker.summary = summary
            tracker.loaded = False
        try:
            yield
        finally:
            with self._physics_lock:
                tracker.summary = previous
                tracker.loaded = previous_loaded
            for body, (peak, steps) in sorted(summary.items()):
                print(f"[TRON2] TABLE_CONTACT stage={stage} link={self.model.body(body).name} "
                      f"peak_normal_force_n={peak:.3f} loaded_duration_s={steps * self.model.opt.timestep:.3f}")

    def move_home(self, duration_sec: float = 2.5) -> dict[str, Any]:
        """Move both arms to official safe home pose (MoveJ) and open grippers."""
        self._ensure_connected()
        self._grip_l = GRIPPER_OPEN_RAD
        self._grip_r = GRIPPER_OPEN_RAD
        try:
            resp = self.client.move_j(DEFAULT_HOME_Q16[:14], duration_sec=duration_sec)
        except Tron2JointLimitRejection as error:
            self._record_joint_limit_rejection(error)
            return error.response
        self._wait_and_record(duration_sec + 0.5)
        return resp

    def move_arms_pose(
        self,
        *,
        left_pos: np.ndarray | list[float] | None = None,
        left_quat_wxyz: np.ndarray | list[float] | None = None,
        right_pos: np.ndarray | list[float] | None = None,
        right_quat_wxyz: np.ndarray | list[float] | None = None,
        duration_sec: float = 2.0,
        report_joint_limit: bool = True,
        report_motion_rejection: bool = False,
        stage: str = "pose",
    ) -> dict[str, Any]:
        """Move either arm via MoveP and optionally return valid controller rejections."""
        self._ensure_connected()
        cur_pose = self.client.get_move_pose()
        l_pos = np.asarray(left_pos, dtype=np.float64) if left_pos is not None else cur_pose.left_pos
        l_quat = np.asarray(left_quat_wxyz, dtype=np.float64) if left_quat_wxyz is not None else cur_pose.left_quat
        r_pos = np.asarray(right_pos, dtype=np.float64) if right_pos is not None else cur_pose.right_pos
        r_quat = np.asarray(right_quat_wxyz, dtype=np.float64) if right_quat_wxyz is not None else cur_pose.right_quat

        try:
            resp = self.client.move_p(l_pos, l_quat, r_pos, r_quat, duration_sec=duration_sec)
        except Tron2MotionRejection as error:
            joint_limit = isinstance(error, Tron2JointLimitRejection)
            if not (report_motion_rejection or (report_joint_limit and joint_limit)):
                raise
            self._log_motion_rejection(error, stage)
            self.record_rejected_motion()
            actual = self.client.get_move_pose()
            for side, requested in (
                ("left", left_pos is not None or left_quat_wxyz is not None),
                ("right", right_pos is not None or right_quat_wxyz is not None),
            ):
                if requested:
                    pos = getattr(actual, f"{side}_pos")
                    quat = getattr(actual, f"{side}_quat")
                    displacement = 1000 * np.linalg.norm(pos - getattr(cur_pose, f"{side}_pos"))
                    print(f"[TRON2] TCP_AFTER_REJECTION stage={stage} arm={side} base_link_pos_m={pos.tolist()} "
                          f"quat_wxyz={quat.tolist()} displacement_since_request_mm={displacement:.3f}")
            return error.response
        self._wait_and_record(duration_sec + 0.5)
        self._step_count += 1
        return resp

    def _log_motion_rejection(self, error: Tron2MotionRejection, stage: str) -> None:
        context = f"stage={stage} result={error.response['result']}"
        event = "CONTROLLER_MOTION_REJECTED"
        if isinstance(error, Tron2JointLimitRejection):
            event = "CONTROLLER_JOINT_LIMIT"
            if error.joint_index is not None and 0 <= error.joint_index < len(PROTOCOL_JOINT_NAMES):
                context += f" joint_index={error.joint_index} joint={PROTOCOL_JOINT_NAMES[error.joint_index]}"
            if error.planned_point is not None:
                context += f" planned_point={error.planned_point}"
        print(f"[TRON2] {event} {context}")
        print(error.response["message"])

    def _record_joint_limit_rejection(self, error: Tron2JointLimitRejection, stage: str = "joint-motion") -> None:
        self._log_motion_rejection(error, stage)
        self.record_rejected_motion()

    def record_rejected_motion(self) -> None:
        """Capture the held simulation state after a motion request fails."""
        self._wait_and_record(0.35)
        self._step_count += 1

    def move_left_joint_path(self, path: np.ndarray, duration_sec: float) -> bool:
        """Follow a joint path. Stop on controller rejection or a timeout against the table."""
        self._ensure_connected()
        for waypoint, left in enumerate(path[1:], start=1):
            q = self._read_arm_joints()
            target = np.r_[left, q[7:14]]
            duration = max(0.65, duration_sec / (len(path) - 1), np.max(np.abs(left - q[:7])) / 0.45)
            try:
                self.client.move_j(target, duration_sec=duration)
            except Tron2JointLimitRejection as error:
                self._record_joint_limit_rejection(error)
                return False
            self._wait_and_record(duration + 0.35)
            deadline = time.monotonic() + 3.0
            while np.max(np.abs((measured := self._read_arm_joints()[:7]) - left)) > 0.015:
                if time.monotonic() >= deadline:
                    joint = int(np.argmax(np.abs(measured - left)))
                    with self._physics_lock:
                        loaded = self._table_contact_feedback.loaded
                    detail = (f"waypoint={waypoint}/{len(path) - 1} joint={PROTOCOL_JOINT_NAMES[joint]} "
                              f"target_rad={left[joint]:.4f} measured_rad={measured[joint]:.4f}")
                    if loaded:
                        print(f"[TRON2] WAYPOINT_NOT_REACHED {detail} stopped_under_table_load=True")
                        self._step_count += 1
                        return False
                    print(f"[TRON2] WAYPOINT_SETTLE_WARNING {detail} continuing_from_measured_pose=True")
                    break
                self._wait_and_record(0.12)
        self._step_count += 1
        return True

    def set_gripper(self, side: str, open_fraction: float, duration_sec: float = 1.0) -> None:
        """Smoothly open (1.0) or close (0.0) the specified gripper ('left' or 'right')."""
        frac = float(np.clip(open_fraction, 0.0, 1.0))
        target = GRIPPER_CLOSE_RAD + frac * (GRIPPER_OPEN_RAD - GRIPPER_CLOSE_RAD)
        start = self._grip_l if side == "left" else self._grip_r
        steps = max(1, int(round(duration_sec / 0.04)))
        for alpha in np.linspace(0.0, 1.0, steps):
            val = float((1.0 - alpha) * start + alpha * target)
            if side == "left":
                self._grip_l = val
            else:
                self._grip_r = val
            self._wait_and_record(0.04)
        self._wait_and_record(0.4)

    # ----------------------- Camera & Observation Helpers -----------------------

    def _get_camera_obs(self, cam_name: str, *, data: mujoco.MjData | None = None) -> dict[str, Any]:
        if data is None:
            data = self._observation_data()
        renderer = self._ensure_renderer()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        fovy_rad = np.deg2rad(self.model.cam_fovy[cam_id])
        fy = 0.5 * self._render_height / np.tan(0.5 * fovy_rad)
        K = np.array(
            [[fy, 0.0, self._render_width / 2.0], [0.0, fy, self._render_height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        t_world_base = np.eye(4)
        t_world_base[:3, :3] = data.xmat[self._base_id].reshape(3, 3)
        t_world_base[:3, 3] = data.xpos[self._base_id]

        t_world_cam_cv = np.eye(4)
        t_world_cam_cv[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ _R_MJ_TO_CV
        t_world_cam_cv[:3, 3] = data.cam_xpos[cam_id]

        pose_mat = np.linalg.inv(t_world_base) @ t_world_cam_cv
        quat_wxyz = _rotmat_to_quat_wxyz(pose_mat[:3, :3])
        pose_7d = np.concatenate([pose_mat[:3, 3], quat_wxyz])

        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render().astype(np.float32)[:, :, None].copy()
        renderer.disable_depth_rendering()

        return {
            "images": {"rgb": rgb, "depth": depth},
            "intrinsics": K,
            "pose_mat": pose_mat,
            "pose": pose_7d,
        }

    # ----------------------- BaseEnv Required Methods -----------------------

    def _reset_cube(self, data: mujoco.MjData) -> None:
        start = self._cube_qpos_adr
        data.qpos[start:start + 7] = [0.52, 0.15, 0.875, 1.0, 0.0, 0.0, 0.0]
        dof = self.model.jnt_dofadr[self._cube_jnt_id]
        data.qvel[dof:dof + 6] = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = self._record_ticks = 0
        # Move the arms through the controller and reset only the free cube.
        self.move_home()
        with self._physics_lock:
            self._reset_cube(self._motor_data)
            mujoco.mj_forward(self.model, self._motor_data)
        self._wait_and_record(0.15)
        self._cube_init_world = self._observation_data().xpos[self._cube_id].copy()
        return self.get_observation(), {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = self.get_observation()
        reward = float(self.compute_reward())
        terminated = bool(self.task_completed())
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def get_observation(self) -> dict[str, Any]:
        data = self._observation_data()
        obs: dict[str, Any] = {
            "robot_joint_pos": data.qpos[self._joint_qpos],
            "left_gripper_open": float(self._grip_l / GRIPPER_OPEN_RAD),
            "right_gripper_open": float(self._grip_r / GRIPPER_OPEN_RAD),
        }
        if self.privileged:
            cube_pos_world = data.xpos[self._cube_id]
            obs["red_cube_pos_base"] = cube_pos_world - data.xpos[self._base_id]
            obs["red_cube_pos_world"] = cube_pos_world.copy()
        if not self.privileged or self.enable_render:
            cam_obs = self._get_camera_obs(self.primary_camera, data=data)
            obs[self.primary_camera] = cam_obs
            if self.primary_camera != "robot0_robotview":
                obs["robot0_robotview"] = cam_obs
        return obs

    def compute_reward(self) -> float:
        cube_z = self._observation_data().xpos[self._cube_id, 2]
        if cube_z >= self._cube_init_world[2] + _TARGET_LIFT_M:
            return 1.0
        dz = cube_z - self._cube_init_world[2]
        return float(np.clip(dz / _TARGET_LIFT_M, 0.0, 0.99))

    def task_completed(self) -> bool:
        target_z = self._cube_init_world[2] + _TARGET_LIFT_M
        return bool(self._observation_data().xpos[self._cube_id, 2] >= target_z)

    # ----------------------- Video Capture (CaP-X Compatible) -----------------------

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [f.copy() for f in self._frame_buffer[start:end]]

    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        renderer = self._ensure_renderer()
        renderer.disable_depth_rendering()
        renderer.update_scene(self._observation_data(), camera=self.save_camera_name)
        self._frame_buffer.append(renderer.render().copy())

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        renderer = self._ensure_renderer()
        renderer.disable_depth_rendering()
        renderer.update_scene(self._observation_data(), camera=self.save_camera_name)
        return renderer.render().copy()

    def close(self) -> None:
        atexit.unregister(self.close)
        try:
            self._physics_stop.set()
            if self._physics_thread is not None:
                self._physics_thread.join(timeout=3.0)
                if self._physics_thread.is_alive():
                    raise RuntimeError("TRON2 motor simulation thread did not stop")
                self._physics_thread = None
            self._motor.close()
        finally:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._connected:
                self.client.close()
                self._connected = False
