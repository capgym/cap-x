from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.g1.sdk import (
    G1_NUM_ARM_JOINTS,
    G1_NUM_DEX3_HAND_JOINTS,
    G1ArmSdkBridge,
    G1Dex3HandBridge,
    as_g1_arm_joints,
    as_g1_dex3_hand_joints,
    dex3_grasp_joints,
)
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.msgpack_server_client_utils import MsgpackNumpyServer
from capx.utils.video_utils import resize_with_pad


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _key_get(mapping: dict[Any, Any] | None, key: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(key.encode(), default)


def _start_msgpack_server_in_background(server: MsgpackNumpyServer) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())

    thread = threading.Thread(target=run_loop, daemon=True, name="g1-msgpack-observation")
    thread.start()
    return loop, thread


class G1RealLowLevel(BaseEnv):
    """Real Unitree G1 right-arm low-level env for CaP-X code execution."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
        sdk_bridge: G1ArmSdkBridge | None = None,
        hand_bridge: G1Dex3HandBridge | None = None,
        dry_run: bool | None = None,
        network_interface: str | None = None,
        domain_id: int = 0,
        enable_dex3: bool = False,
        dex3_hand_side: str = "right",
        dex3_hand_cmd_topic: str | None = None,
        dex3_hand_state_topic: str | None = None,
        observation_server_host: str = "0.0.0.0",
        observation_server_port: int | None = None,
        wait_for_observation_on_reset: bool = False,
        action_publish_period: float = 0.02,
        joint_interpolation_steps: int = 50,
        hold_arm_after_move: bool = True,
        arm_hold_publish_period: float | None = None,
        default_joint_positions: list[float] | None = None,
        pregrasp_side_joints: list[float] | None = None,
        pregrasp_side_before_first_pose: bool = False,
        write_timeout: float | None = 2.0,
        sam3_mask_popup: bool | None = None,
        grasp_debug_visualization: bool | None = None,
        grasp_debug_output_dir: str | None = None,
        grasp_debug_approach_distance: float = 0.1,
    ) -> None:
        super().__init__()
        self.seed_value = seed
        self.privileged = privileged
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self.dry_run = _env_bool("CAPX_G1_DRY_RUN", True) if dry_run is None else bool(dry_run)
        self.wait_for_observation_on_reset = wait_for_observation_on_reset
        self._action_publish_period = float(action_publish_period)
        self._joint_interpolation_steps = max(0, int(joint_interpolation_steps))
        self._hold_arm_after_move = bool(hold_arm_after_move)
        self._arm_hold_publish_period = (
            max(float(arm_hold_publish_period), 0.001)
            if arm_hold_publish_period is not None
            else max(self._action_publish_period, 0.02)
        )
        self._arm_hold_target: np.ndarray | None = None
        self._arm_hold_paused = False
        self._arm_hold_stop = threading.Event()
        self._arm_hold_thread: threading.Thread | None = None
        self._arm_hold_lock = threading.Lock()
        self._arm_hold_last_error: str | None = None
        self._pregrasp_side_joints = (
            None if pregrasp_side_joints is None else as_g1_arm_joints(pregrasp_side_joints).copy()
        )
        self._pregrasp_side_before_first_pose = bool(pregrasp_side_before_first_pose)
        self._pregrasp_side_done = False
        self._gripper_fraction = 1.0
        self._enable_dex3 = bool(enable_dex3 or hand_bridge is not None)
        self._current_hand_joints = np.zeros(G1_NUM_DEX3_HAND_JOINTS, dtype=np.float64)
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._last_recorded_observation_id: int | None = None
        self.sam3_mask_popup = _env_bool(
            "CAPX_SAM3_MASK_POPUP",
            False if sam3_mask_popup is None else bool(sam3_mask_popup),
        )
        self.grasp_debug_visualization = _env_bool(
            "CAPX_GRASP_DEBUG_VISUALIZATION",
            False if grasp_debug_visualization is None else bool(grasp_debug_visualization),
        )
        self.grasp_debug_output_dir = (
            "outputs/g1_grasp_bottle/grasp_debug"
            if grasp_debug_output_dir is None
            else str(grasp_debug_output_dir)
        )
        self.grasp_debug_approach_distance = float(grasp_debug_approach_distance)

        if default_joint_positions is None:
            self._current_joints = np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)
        else:
            self._current_joints = as_g1_arm_joints(default_joint_positions).copy()

        self.sdk_bridge = sdk_bridge or G1ArmSdkBridge(
            network_interface=network_interface,
            domain_id=domain_id,
            dry_run=self.dry_run,
            write_timeout=write_timeout,
        )
        self.hand_bridge = hand_bridge
        if self._enable_dex3 and self.hand_bridge is None:
            self.hand_bridge = G1Dex3HandBridge(
                network_interface=network_interface,
                domain_id=domain_id,
                dry_run=self.dry_run,
                hand_side=dex3_hand_side,
                publisher_topic=dex3_hand_cmd_topic,
                state_topic=dex3_hand_state_topic,
                write_timeout=write_timeout,
            )
        print(
            f"[g1-real] dry_run={self.dry_run} "
            f"network_interface={getattr(self.sdk_bridge, 'network_interface', '<custom>')}"
        )
        if not self.dry_run:
            self.sdk_bridge.connect()
            if self.hand_bridge is not None:
                self.hand_bridge.connect()

        self.low_level_server: MsgpackNumpyServer | None = None
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._server_thread: threading.Thread | None = None
        if observation_server_port is not None:
            self.low_level_server = MsgpackNumpyServer(
                host=observation_server_host,
                port=int(observation_server_port),
            )
            self._server_loop, self._server_thread = _start_msgpack_server_in_background(
                self.low_level_server
            )

        self.latest_action: dict[str, Any] = {}
        self.obs: dict[str, Any] = {
            "robot_joint_pos": self._current_joints.copy(),
            "gripper_type": "dex3" if self.hand_bridge is not None else "rubber_hand",
            "gripper_fraction": self._gripper_fraction,
        }
        if self.hand_bridge is not None:
            self.obs["gripper_joint_pos"] = self._current_hand_joints.copy()

    def _convert_msgpack_observation(self, msg: dict[Any, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        formatted = _key_get(msg, "robot0_robotview")
        if formatted is not None:
            obs["robot0_robotview"] = formatted

        camera = _key_get(msg, "camera_top")
        if camera is not None:
            robotview: dict[str, Any] = {"images": {}}
            images = _key_get(camera, "images", {})

            rgb = _key_get(images, "left_rgb")
            if rgb is None:
                rgb = _key_get(images, "rgb")
            if rgb is not None:
                robotview["images"]["rgb"] = np.asarray(rgb)

            depth = _key_get(camera, "depth_data")
            if depth is None:
                depth = _key_get(images, "depth")
            if depth is not None:
                depth_arr = np.asarray(depth)
                if depth_arr.ndim == 2:
                    depth_arr = depth_arr[:, :, None]
                robotview["images"]["depth"] = depth_arr

            intrinsics = _key_get(camera, "intrinsics")
            left_intrinsics = _key_get(intrinsics, "left", {}) if intrinsics is not None else {}
            matrix = _key_get(left_intrinsics, "intrinsics_matrix")
            if matrix is None:
                matrix = _key_get(camera, "intrinsics_matrix")
            if matrix is not None:
                robotview["intrinsics"] = np.asarray(matrix)

            pose = _key_get(camera, "pose")
            if pose is not None:
                robotview["pose"] = np.asarray(pose)

            pose_mat = _key_get(camera, "pose_mat")
            if pose_mat is not None:
                robotview["pose_mat"] = np.asarray(pose_mat)

            if robotview["images"] or "intrinsics" in robotview:
                obs["robot0_robotview"] = robotview

        for key in ("cube_center", "cube_half_size", "target_object_name"):
            value = _key_get(msg, key)
            if value is not None:
                obs[key] = value

        return obs

    def _update_from_network(self) -> None:
        if self.low_level_server is None or self.low_level_server.latest_observation is None:
            return

        raw_observation = self.low_level_server.latest_observation
        new_obs = self._convert_msgpack_observation(raw_observation)
        for key, value in new_obs.items():
            self.obs[key] = value

        observation_id = id(raw_observation)
        if self._record_frames and observation_id != self._last_recorded_observation_id:
            before_count = len(self._frame_buffer)
            self._record_frame()
            if len(self._frame_buffer) > before_count:
                self._last_recorded_observation_id = observation_id

    def _update_from_sdk(self) -> None:
        self._current_joints = self.sdk_bridge.get_arm_joint_positions().copy()
        self.obs["robot_joint_pos"] = self._current_joints.copy()
        if self.hand_bridge is not None:
            self._current_hand_joints = self.hand_bridge.get_hand_joint_positions().copy()
            self.obs["gripper_type"] = "dex3"
            self.obs["gripper_joint_pos"] = self._current_hand_joints.copy()
        else:
            self.obs["gripper_type"] = "rubber_hand"
        self.obs["gripper_fraction"] = self._gripper_fraction

    def _publish_action_metadata(self, target: np.ndarray) -> None:
        g1_payload = {
            "joint_pos": target.astype(np.float32).tolist(),
            "arm_joint_pos": target.astype(np.float32).tolist(),
            "gripper_type": "dex3" if self.hand_bridge is not None else "rubber_hand",
            "gripper_fraction": float(self._gripper_fraction),
        }
        if self.hand_bridge is not None:
            g1_payload["hand_joint_pos"] = self._current_hand_joints.astype(np.float32).tolist()
        command = {
            "timestamp": time.time(),
            "g1": g1_payload,
        }
        self.latest_action = command
        if self.low_level_server is not None:
            self.low_level_server.latest_action = command

    def _publish_joints_or_raise(self, target: np.ndarray) -> None:
        ok = self.sdk_bridge.publish_joints(target)
        if ok is False:
            raise RuntimeError(
                "rt/lowcmd publish failed. Check that the G1 is in debug/low-level mode, "
                "the network interface is correct, and a robot DDS subscriber is matched."
            )

    def _ensure_arm_hold_thread(self) -> None:
        if self.dry_run or not self._hold_arm_after_move:
            return
        with self._arm_hold_lock:
            if self._arm_hold_thread is not None and self._arm_hold_thread.is_alive():
                return
            self._arm_hold_stop.clear()
            thread = threading.Thread(
                target=self._arm_hold_loop,
                daemon=True,
                name="g1-arm-hold-lowcmd",
            )
            self._arm_hold_thread = thread
            thread.start()

    def _set_arm_hold_paused(self, paused: bool) -> None:
        if self.dry_run or not self._hold_arm_after_move:
            return
        with self._arm_hold_lock:
            self._arm_hold_paused = bool(paused)

    def _set_arm_hold_target(self, target: np.ndarray, *, publish_now: bool = True) -> None:
        if self.dry_run or not self._hold_arm_after_move:
            return
        target_arr = as_g1_arm_joints(target).copy()
        with self._arm_hold_lock:
            self._arm_hold_target = target_arr
            self._arm_hold_last_error = None
        if publish_now:
            self._publish_arm_hold_once(target_arr)
        self._ensure_arm_hold_thread()

    def _current_arm_hold_target(self) -> np.ndarray | None:
        if self.dry_run or not self._hold_arm_after_move:
            return None
        with self._arm_hold_lock:
            return None if self._arm_hold_target is None else self._arm_hold_target.copy()

    def _publish_arm_hold_once(self, target: np.ndarray) -> None:
        if self.dry_run or not self._hold_arm_after_move:
            return
        target_arr = as_g1_arm_joints(target).copy()
        self._publish_joints_or_raise(target_arr)
        self._publish_action_metadata(target_arr)

    def _refresh_arm_hold_now(self) -> None:
        if self.dry_run or not self._hold_arm_after_move:
            return
        target = self._current_arm_hold_target()
        if target is None:
            target = self._current_joints.copy()
        self._set_arm_hold_target(target, publish_now=True)

    def _arm_hold_loop(self) -> None:
        while not self._arm_hold_stop.wait(self._arm_hold_publish_period):
            with self._arm_hold_lock:
                paused = self._arm_hold_paused
                target = None if self._arm_hold_target is None else self._arm_hold_target.copy()
            if paused or target is None:
                continue
            try:
                ok = self.sdk_bridge.publish_joints(target)
                if ok is False:
                    message = "rt/lowcmd hold publish failed"
                    with self._arm_hold_lock:
                        if self._arm_hold_last_error != message:
                            print(f"[g1-real] {message}")
                        self._arm_hold_last_error = message
                    continue
                self._publish_action_metadata(target)
            except Exception as exc:
                message = str(exc)
                with self._arm_hold_lock:
                    if self._arm_hold_last_error != message:
                        print(f"[g1-real] arm hold publish error: {message}")
                    self._arm_hold_last_error = message

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed_value = seed

        if self.wait_for_observation_on_reset:
            while self.low_level_server is not None and self.low_level_server.latest_observation is None:
                print("Waiting for observation from G1 real environment...")
                time.sleep(1.0)

        self._pregrasp_side_done = False
        self._update_from_network()
        self._update_from_sdk()
        return self.obs.copy(), {}

    def move_to_pregrasp_side_pose_once(self, *, force: bool = False) -> bool:
        """Move the right arm to a configured side-lift waypoint before Cartesian grasping."""

        if self._pregrasp_side_joints is None:
            return False
        if not force and not self._pregrasp_side_before_first_pose:
            return False
        if not force and self._pregrasp_side_done:
            return False

        target_str = np.array2string(self._pregrasp_side_joints, precision=4, suppress_small=True)
        print(f"[g1-real] moving to pregrasp side pose target={target_str}")
        self.move_to_joints_blocking(self._pregrasp_side_joints)
        self._pregrasp_side_done = True
        return True

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action is not None:
            if isinstance(action, dict) and "joint_pos" in action:
                self.move_to_joints_blocking(action["joint_pos"])
            else:
                arr = np.asarray(action)
                if arr.size == G1_NUM_ARM_JOINTS:
                    self.move_to_joints_blocking(arr)

        obs = self.get_observation()
        return obs, 0.0, False, False, {}

    def move_to_joints_blocking(
        self,
        joints: Any,
        *,
        tolerance: float = 0.05,
        max_steps: int = 350,
    ) -> None:
        target = as_g1_arm_joints(joints).copy()
        target_str = np.array2string(target, precision=4, suppress_small=True)
        print(f"[g1-real] move_to_joints target={target_str} dry_run={self.dry_run}")

        if self.dry_run:
            self.sdk_bridge.publish_joints(target)
            self._publish_action_metadata(target)
            self._current_joints = target.copy()
            self.obs["robot_joint_pos"] = self._current_joints.copy()
            return

        wait_for_low_state = getattr(self.sdk_bridge, "wait_for_low_state", None)
        if callable(wait_for_low_state) and not wait_for_low_state(timeout_s=3.0):
            raise RuntimeError(
                "Did not receive G1 rt/lowstate within 3 seconds; refusing to report motion complete."
            )

        self._update_from_network()
        self._update_from_sdk()
        start_joints = self._current_joints.copy()
        delta = target - start_joints
        start_str = np.array2string(start_joints, precision=4, suppress_small=True)
        delta_str = np.array2string(delta, precision=4, suppress_small=True)
        print(
            f"[g1-real] current={start_str} delta={delta_str} "
            f"delta_norm={np.linalg.norm(delta):.4f}"
        )
        should_hold_target = False
        try:
            if self._joint_interpolation_steps > 0 and np.linalg.norm(self._current_joints - target) >= tolerance:
                waypoints = np.linspace(
                    self._current_joints,
                    target,
                    self._joint_interpolation_steps + 1,
                    dtype=np.float64,
                )[1:]
                for waypoint in waypoints:
                    self._update_from_network()
                    self._publish_joints_or_raise(waypoint)
                    self._set_arm_hold_target(waypoint, publish_now=False)
                    self._publish_action_metadata(waypoint)
                    time.sleep(max(self._action_publish_period, 0.0))

            last_publish = -float("inf")
            reached = False
            for _ in range(max_steps):
                self._update_from_network()
                self._update_from_sdk()
                if np.linalg.norm(self._current_joints - target) < tolerance:
                    reached = True
                    break

                now = time.time()
                if now - last_publish >= self._action_publish_period:
                    self._publish_joints_or_raise(target)
                    self._set_arm_hold_target(target, publish_now=False)
                    self._publish_action_metadata(target)
                    last_publish = now
                time.sleep(0.01)

            should_hold_target = True
        finally:
            if should_hold_target:
                self._set_arm_hold_target(target)

        final_str = np.array2string(self._current_joints, precision=4, suppress_small=True)
        if not reached:
            err = float(np.linalg.norm(self._current_joints - target))
            print(
                f"[g1-real] move_to_joints reached timeout with joint error {err:.4f}; "
                f"final={final_str}."
            )
        else:
            print(f"[g1-real] move_to_joints reached target; final={final_str}.")

    @staticmethod
    def _densify_joint_trajectory(
        start: np.ndarray,
        trajectory: np.ndarray,
        *,
        max_joint_step: float,
    ) -> np.ndarray:
        start_arr = as_g1_arm_joints(start).copy()
        dense: list[np.ndarray] = []
        previous = start_arr
        for waypoint in trajectory:
            target = as_g1_arm_joints(waypoint).copy()
            delta = target - previous
            steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) / float(max_joint_step))))
            for alpha in np.linspace(1.0 / steps, 1.0, steps):
                dense.append(previous + alpha * delta)
            previous = target
        return np.asarray(dense, dtype=np.float64)

    @staticmethod
    def _stream_targets_with_final_hold(
        trajectory: np.ndarray,
        period: float,
        hold_final_seconds: float,
    ) -> list[np.ndarray]:
        targets = [as_g1_arm_joints(waypoint).copy() for waypoint in trajectory]
        if not targets:
            return []
        hold_seconds = max(float(hold_final_seconds), 0.0)
        if hold_seconds > 0.0 and period > 0.0:
            repeats = max(1, int(np.ceil(hold_seconds / period)))
            targets.extend([targets[-1].copy() for _ in range(repeats)])
        return targets

    def execute_joint_trajectory_streaming(
        self,
        trajectory: Any,
        *,
        dt: float | None = None,
        max_joint_step: float | None = None,
        hold_final_seconds: float = 0.0,
    ) -> None:
        """Stream a precomputed right-arm joint trajectory without per-waypoint blocking.

        Args:
            trajectory: Array-like shape (N, 7), in G1 right-arm joint order.
            dt: Optional publish period in seconds. If None, uses the environment
                action publish period.
            max_joint_step: Optional maximum absolute joint delta per published
                command, in radians. When set, the trajectory is upsampled from
                the current arm state to avoid large open-loop target jumps.
            hold_final_seconds: Seconds to keep re-publishing the final target
                before returning control to the hold thread.

        Notes:
            This is intended for short Cartesian approach trajectories that have
            already been converted to IK waypoints. It continuously publishes each
            waypoint and updates the arm hold target as it goes, avoiding the
            stop/restart behavior of calling move_to_joints_blocking repeatedly.
        """

        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        if traj.ndim != 2 or traj.shape[1] != G1_NUM_ARM_JOINTS:
            raise ValueError(
                f"Expected G1 joint trajectory shape (N, {G1_NUM_ARM_JOINTS}), got {traj.shape}."
            )
        if traj.shape[0] == 0:
            return

        if max_joint_step is not None:
            max_step = float(max_joint_step)
            if max_step <= 0.0:
                raise ValueError("max_joint_step must be positive when provided.")
            start = self._current_joints.copy() if self.dry_run else self.sdk_bridge.get_arm_joint_positions().copy()
            traj = self._densify_joint_trajectory(start, traj, max_joint_step=max_step)

        period = self._action_publish_period if dt is None else max(float(dt), 0.0)
        print(f"[g1-real] stream_joint_trajectory points={traj.shape[0]} dt={period:.4f} dry_run={self.dry_run}")

        if self.dry_run:
            for target in self._stream_targets_with_final_hold(traj, period, hold_final_seconds):
                self.sdk_bridge.publish_joints(target)
                self._publish_action_metadata(target)
                self._current_joints = target.copy()
                self.obs["robot_joint_pos"] = self._current_joints.copy()
                if period > 0.0:
                    time.sleep(period)
            return

        wait_for_low_state = getattr(self.sdk_bridge, "wait_for_low_state", None)
        if callable(wait_for_low_state) and not wait_for_low_state(timeout_s=3.0):
            raise RuntimeError(
                "Did not receive G1 rt/lowstate within 3 seconds; refusing to stream trajectory."
            )

        completed = False
        try:
            for target in self._stream_targets_with_final_hold(traj, period, hold_final_seconds):
                self._publish_joints_or_raise(target)
                self._set_arm_hold_target(target, publish_now=False)
                self._publish_action_metadata(target)
                self._current_joints = target.copy()
                self.obs["robot_joint_pos"] = self._current_joints.copy()
                if period > 0.0:
                    time.sleep(period)
            completed = True
        finally:
            if completed:
                self._set_arm_hold_target(traj[-1])

    def move_to_arm_joints_blocking(
        self,
        arm: str,
        joints: Any,
        *,
        tolerance: float = 0.05,
        max_steps: int = 350,
    ) -> None:
        arm_key = arm.lower()
        if arm_key not in {"right", "single", "arm"}:
            raise ValueError("Only the G1 right arm is supported by this single-arm env.")
        self.move_to_joints_blocking(joints, tolerance=tolerance, max_steps=max_steps)

    def get_current_arm_joints(self) -> np.ndarray:
        self._update_from_network()
        self._update_from_sdk()
        return self._current_joints.copy()

    def move_hand_to_joints_blocking(self, joints: Any) -> None:
        target = as_g1_dex3_hand_joints(joints).copy()
        target_str = np.array2string(target, precision=4, suppress_small=True)
        print(f"[g1-real] move_hand target={target_str} dry_run={self.dry_run}")

        self._refresh_arm_hold_now()

        if self.hand_bridge is None:
            self._current_hand_joints = target.copy()
            self.obs["gripper_type"] = "rubber_hand"
            self.obs["gripper_joint_pos"] = self._current_hand_joints.copy()
            return

        if not self.dry_run:
            wait_for_hand_state = getattr(self.hand_bridge, "wait_for_hand_state", None)
            if callable(wait_for_hand_state) and not wait_for_hand_state(timeout_s=3.0):
                raise RuntimeError(
                    "Did not receive G1 Dex3 hand state within 3 seconds; refusing to report hand motion complete."
                )

        ok = self.hand_bridge.publish_hand_joints(target)
        if ok is False:
            raise RuntimeError(
                "Dex3 hand publish failed. Check the right-hand DDS topic, network interface, "
                "and that the hand controller is running."
            )
        self._current_hand_joints = target.copy()
        self.obs["gripper_type"] = "dex3"
        self.obs["gripper_joint_pos"] = self._current_hand_joints.copy()
        self.obs["gripper_fraction"] = self._gripper_fraction
        self._refresh_arm_hold_now()

    def open_dex3_hand(self) -> None:
        self._gripper_fraction = 1.0
        self.move_hand_to_joints_blocking(dex3_grasp_joints(trigger=0.0, squeeze=0.0))

    def close_dex3_hand(self) -> None:
        self._gripper_fraction = 0.0
        self.move_hand_to_joints_blocking(dex3_grasp_joints(trigger=1.0, squeeze=1.0))

    def close_dex3_index_pinch(self) -> None:
        self._gripper_fraction = 0.0
        self.move_hand_to_joints_blocking(dex3_grasp_joints(trigger=1.0, squeeze=0.0))

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        if self.hand_bridge is not None:
            if self._gripper_fraction >= 0.5:
                self.open_dex3_hand()
            else:
                self.close_dex3_hand()
            return
        self.obs["gripper_type"] = "rubber_hand"
        self.obs["gripper_fraction"] = self._gripper_fraction

    def _step_once(self) -> None:
        self.sdk_bridge.publish_joints(self._current_joints)
        self._publish_action_metadata(self._current_joints)
        if self._record_frames:
            self._record_frame()

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def get_observation(self) -> dict[str, Any]:
        self._update_from_network()
        self._update_from_sdk()
        return self.obs

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
            self._last_recorded_observation_id = None
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def _record_frame(self) -> None:
        rgbs = obs_get_rgb(self.obs)
        if not rgbs:
            return
        frame = resize_with_pad(list(rgbs.values())[0], 480, 640)
        self._frame_buffer.append(frame)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        rgbs = obs_get_rgb(self.get_observation())
        if rgbs:
            return list(rgbs.values())[0]
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        self._arm_hold_stop.set()
        hold_thread = self._arm_hold_thread
        if hold_thread is not None and hold_thread.is_alive():
            hold_thread.join(timeout=1.0)
        self.sdk_bridge.close()
        if self.hand_bridge is not None:
            self.hand_bridge.close()
