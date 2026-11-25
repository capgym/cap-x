"""Low-level MuJoCo + official Tron2_controller environment for CaP-X.

Architecture (Direction 3 — simulation aligned with real TRON2 hardware):
- Arm motion (16 axes) is planned & controlled by official LimX Tron2_controller
  over WebSocket (MoveJ / MoveP on ws://<host>:5000).
- Tabletop physics, object contact, 4-bar parallel gripper actuation, and RGB-D
  camera rendering run locally in MuJoCo using `scene_table_cube.xml`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.tron2.official_client import (
    DEFAULT_HOME_Q16,
    HORIZONTAL_GRASP_QUAT_WXYZ,
    PROTOCOL_JOINT_NAMES,
    Tron2OfficialClient,
)

_DEFAULT_XML = (
    Path(__file__).resolve().parents[2]
    / "assets/tron2/DACH_TRON2A/xml/scene_table_cube.xml"
)

# Local PD gains to track official 16-axis q_ref inside the tabletop MuJoCo scene
_ARM_GAINS: tuple[tuple[float, float], ...] = (
    (400.0, 30.0), (400.0, 30.0), (250.0, 18.0), (300.0, 22.0),
    (120.0, 8.0), (120.0, 8.0), (120.0, 8.0),
    (400.0, 30.0), (400.0, 30.0), (250.0, 18.0), (300.0, 22.0),
    (120.0, 8.0), (120.0, 8.0), (120.0, 8.0),
    (80.0, 6.0), (80.0, 6.0),
)

_R_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])

GRIPPER_OPEN_RAD = 0.72
GRIPPER_CLOSE_RAD = 0.0
DEFAULT_RIGHT_PARK_POS = np.array([0.46, -0.32, -0.20], dtype=np.float64)
DEFAULT_LEFT_PARK_POS = np.array([0.46, 0.32, -0.20], dtype=np.float64)


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, rot.reshape(9))
    return q


class Tron2CubeLiftLowLevel(BaseEnv):
    """CaP-X BaseEnv wrapping TRON2 tabletop cube lifting with official Tron2_controller."""

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
    ) -> None:
        super().__init__()
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

        # Load tabletop MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
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

        self._arm_bindings: list[tuple[int, int, int, int, float, float, float, float]] = []
        for idx, name in enumerate(PROTOCOL_JOINT_NAMES):
            qpos_adr, dof_adr, act_id = jnt_map[name]
            kp, kd = _ARM_GAINS[idx]
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self._arm_bindings.append((idx, qpos_adr, dof_adr, act_id, kp, kd, float(lo), float(hi)))

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
        self._q_ref = np.array(DEFAULT_HOME_Q16, dtype=np.float64)
        self._grip_l = GRIPPER_OPEN_RAD
        self._grip_r = GRIPPER_OPEN_RAD
        self._last_left_pose = (DEFAULT_LEFT_PARK_POS.copy(), np.array(HORIZONTAL_GRASP_QUAT_WXYZ, dtype=np.float64))
        self._last_right_pose = (DEFAULT_RIGHT_PARK_POS.copy(), np.array(HORIZONTAL_GRASP_QUAT_WXYZ, dtype=np.float64))
        self._cube_init_world = np.array([0.52, 0.15, 0.875], dtype=np.float64)
        self._step_count = 0
        self._poll_count = 0

        # Offscreen renderer (created lazily when needed)
        self._renderer: mujoco.Renderer | None = None
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._record_stride = 2  # record every 2 poll cycles (~20 FPS at 40 Hz polling)

    # ----------------------- Connection & Physics Sync -----------------------

    def _ensure_connected(self) -> None:
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

    def _apply_pd_and_step(self, n_substeps: int = 25) -> None:
        """Advance local MuJoCo physics by n_substeps (default 25 * 1ms = 25ms)."""
        for _ in range(n_substeps):
            for idx, qpos_adr, dof_adr, act_id, kp, kd, lo, hi in self._arm_bindings:
                tau = (
                    kp * (self._q_ref[idx] - self.data.qpos[qpos_adr])
                    + kd * (0.0 - self.data.qvel[dof_adr])
                    + self.data.qfrc_bias[dof_adr]
                )
                self.data.ctrl[act_id] = float(np.clip(tau, lo, hi))

            for (qpos_adr, dof_adr, act_id, kp, kd, lo, hi), target in (
                (self._grip_base_l, 0.0),
                (self._grip_base_r, 0.0),
                (self._grip_drv_l, self._grip_l),
                (self._grip_drv_r, self._grip_r),
            ):
                tau = (
                    kp * (target - self.data.qpos[qpos_adr])
                    + kd * (0.0 - self.data.qvel[dof_adr])
                    + self.data.qfrc_bias[dof_adr]
                )
                self.data.ctrl[act_id] = float(np.clip(tau, lo, hi))

            mujoco.mj_step(self.model, self.data)

    def sync_from_controller(self, duration_sec: float, poll_dt: float = 0.025) -> None:
        """Stream official 16-axis joint state and step tabletop physics in 1:1 time."""
        self._ensure_connected()
        substeps = max(1, int(round(poll_dt / self.model.opt.timestep)))
        end_t = time.perf_counter() + float(duration_sec)
        while time.perf_counter() < end_t:
            t0 = time.perf_counter()
            self._q_ref = self.client.get_joint_state()
            self._apply_pd_and_step(substeps)
            self._poll_count += 1
            if self._record_frames and self._poll_count % self._record_stride == 0:
                self._record_frame()
            elapsed = time.perf_counter() - t0
            if elapsed < poll_dt:
                time.sleep(poll_dt - elapsed)

    # ----------------------- High-Level Motion Primitives -----------------------

    def move_home(self, duration_sec: float = 2.5) -> dict[str, Any]:
        """Move both arms to official safe home pose (MoveJ) and open grippers."""
        self._ensure_connected()
        self._grip_l = GRIPPER_OPEN_RAD
        self._grip_r = GRIPPER_OPEN_RAD
        resp = self.client.move_j(DEFAULT_HOME_Q16[:14], duration_sec=duration_sec)
        self.sync_from_controller(duration_sec + 0.5)
        pose = self.client.get_move_pose()
        self._last_left_pose = (pose.left_pos.copy(), pose.left_quat.copy())
        self._last_right_pose = (pose.right_pos.copy(), pose.right_quat.copy())
        return resp

    def move_arms_pose(
        self,
        *,
        left_pos: np.ndarray | list[float] | None = None,
        left_quat_wxyz: np.ndarray | list[float] | None = None,
        right_pos: np.ndarray | list[float] | None = None,
        right_quat_wxyz: np.ndarray | list[float] | None = None,
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        """Move left, right, or both arms via official Tron2_controller MoveP."""
        self._ensure_connected()
        cur_pose = self.client.get_move_pose()
        l_pos = np.asarray(left_pos, dtype=np.float64) if left_pos is not None else cur_pose.left_pos
        l_quat = np.asarray(left_quat_wxyz, dtype=np.float64) if left_quat_wxyz is not None else cur_pose.left_quat
        r_pos = np.asarray(right_pos, dtype=np.float64) if right_pos is not None else cur_pose.right_pos
        r_quat = np.asarray(right_quat_wxyz, dtype=np.float64) if right_quat_wxyz is not None else cur_pose.right_quat

        resp = self.client.move_p(l_pos, l_quat, r_pos, r_quat, duration_sec=duration_sec)
        self.sync_from_controller(duration_sec + 0.5)
        self._last_left_pose = (l_pos.copy(), l_quat.copy())
        self._last_right_pose = (r_pos.copy(), r_quat.copy())
        self._step_count += 1
        return resp

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
            self.sync_from_controller(0.04)
        self.sync_from_controller(0.4)

    # ----------------------- Camera & Observation Helpers -----------------------

    def _get_camera_obs(self, cam_name: str) -> dict[str, Any]:
        renderer = self._ensure_renderer()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        fovy_rad = np.deg2rad(self.model.cam_fovy[cam_id])
        fy = 0.5 * self._render_height / np.tan(0.5 * fovy_rad)
        K = np.array(
            [[fy, 0.0, self._render_width / 2.0], [0.0, fy, self._render_height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        t_world_base = np.eye(4)
        t_world_base[:3, :3] = self.data.xmat[self._base_id].reshape(3, 3)
        t_world_base[:3, 3] = self.data.xpos[self._base_id]

        t_world_cam_cv = np.eye(4)
        t_world_cam_cv[:3, :3] = self.data.cam_xmat[cam_id].reshape(3, 3) @ _R_MJ_TO_CV
        t_world_cam_cv[:3, 3] = self.data.cam_xpos[cam_id]

        pose_mat = np.linalg.inv(t_world_base) @ t_world_cam_cv
        quat_wxyz = _rotmat_to_quat_wxyz(pose_mat[:3, :3])
        pose_7d = np.concatenate([pose_mat[:3, 3], quat_wxyz])

        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=cam_name)
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=cam_name)
        depth = renderer.render().astype(np.float32)[:, :, None].copy()
        renderer.disable_depth_rendering()

        return {
            "images": {"rgb": rgb, "depth": depth},
            "intrinsics": K,
            "pose_mat": pose_mat,
            "pose": pose_7d,
        }

    # ----------------------- BaseEnv Required Methods -----------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._ensure_connected()
        self._step_count = 0
        self._poll_count = 0

        # 1) 先让 4090w 官方控制器抬臂到位（避免从垂手姿态在桌板底下硬顶桌底）
        self._grip_l = GRIPPER_OPEN_RAD
        self._grip_r = GRIPPER_OPEN_RAD
        self.client.move_j(DEFAULT_HOME_Q16[:14], duration_sec=2.5)
        time.sleep(3.0)
        q_home = self.client.get_joint_state()

        # 2) 再重置本地桌面场景，直接以“抬臂在桌面上方”的干净姿态开局
        mujoco.mj_resetData(self.model, self.data)
        self._q_ref = q_home.copy()
        for idx, qpos_adr, *_ in self._arm_bindings:
            self.data.qpos[qpos_adr] = q_home[idx]
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = np.array([0.52, 0.15, 0.875])
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.model.jnt_dofadr[self._cube_jnt_id] : self.model.jnt_dofadr[self._cube_jnt_id] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._apply_pd_and_step(150)
        self._cube_init_world = self.data.xpos[self._cube_id].copy()

        pose = self.client.get_move_pose()
        self._last_left_pose = (pose.left_pos.copy(), pose.left_quat.copy())
        self._last_right_pose = (pose.right_pos.copy(), pose.right_quat.copy())
        return self.get_observation(), {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = self.get_observation()
        reward = float(self.compute_reward())
        terminated = bool(self.task_completed())
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def get_observation(self) -> dict[str, Any]:
        cube_pos_base = self.data.xpos[self._cube_id] - self.data.xpos[self._base_id]
        obs: dict[str, Any] = {
            "robot_joint_pos": self._q_ref.copy(),
            "left_gripper_open": float(self._grip_l / GRIPPER_OPEN_RAD),
            "right_gripper_open": float(self._grip_r / GRIPPER_OPEN_RAD),
            "red_cube_pos_base": cube_pos_base.copy(),
            "red_cube_pos_world": self.data.xpos[self._cube_id].copy(),
        }
        if not self.privileged or self.enable_render:
            cam_obs = self._get_camera_obs(self.primary_camera)
            obs[self.primary_camera] = cam_obs
            if self.primary_camera != "robot0_robotview":
                obs["robot0_robotview"] = cam_obs
        return obs

    def compute_reward(self) -> float:
        dz = float(self.data.xpos[self._cube_id][2] - self._cube_init_world[2])
        if dz >= 0.08:
            return 1.0
        return float(np.clip(dz / 0.12, 0.0, 0.99))

    def task_completed(self) -> bool:
        dz = float(self.data.xpos[self._cube_id][2] - self._cube_init_world[2])
        return dz >= 0.08

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
        renderer.update_scene(self.data, camera=self.save_camera_name)
        self._frame_buffer.append(renderer.render().copy())

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        renderer = self._ensure_renderer()
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=self.save_camera_name)
        return renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._connected:
            self.client.close()
            self._connected = False
