from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import select
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from capx.integrations.g1.sdk import (
    DEFAULT_G1_NETWORK_INTERFACE,
    G1_ARM_MOTOR_INDICES_BY_SIDE,
    G1_DUAL_ARM_NUM_JOINTS,
    G1_LOWCMD_CONTROLLED_MOTOR_INDICES,
    G1_NUM_ARM_JOINTS,
    G1_ARM_DUAL_CFG_SLICE_BY_SIDE,
    as_g1_arm_joints,
    normalize_g1_arm_side,
)

DEFAULT_G1_GATEWAY_LCM_URL = "udpm://239.255.76.67:7667?ttl=255"
DEFAULT_G1_GATEWAY_ARM_CHANNEL = "arm_action"
DEFAULT_G1_GATEWAY_BODY_STATE_CHANNEL = "body_control_data"
DEFAULT_G1_GATEWAY_IMU_CHANNEL = "state_estimator_data"
DEFAULT_OPENHOMIE_LCM_TYPES_DIR = (
    Path("/home/peilab/development/OpenHomie")
    / "HomieDeploy"
    / "g1_gym_deploy"
    / "lcm_types"
)
G1_GATEWAY_ARM_MOTOR_SLICE = slice(15, 29)


class G1GatewayUnavailableError(RuntimeError):
    """Raised when the LCM runtime or generated Gateway message types are unavailable."""


def _resolve_lcm_types_dir(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        return Path(path).expanduser()

    env_path = os.environ.get("CAPX_G1_GATEWAY_LCM_TYPES_DIR") or os.environ.get(
        "OPENHOMIE_LCM_TYPES_DIR"
    )
    if env_path:
        return Path(env_path).expanduser()

    openhomie_root = os.environ.get("OPENHOMIE_ROOT_DIR")
    if openhomie_root:
        return (
            Path(openhomie_root).expanduser()
            / "HomieDeploy"
            / "g1_gym_deploy"
            / "lcm_types"
        )

    return DEFAULT_OPENHOMIE_LCM_TYPES_DIR


def _load_generated_lcm_type(
    lcm_types_dir: Path,
    module_name: str,
    class_name: str,
) -> type[Any]:
    module_path = lcm_types_dir / f"{module_name}.py"
    if module_path.exists():
        resolved = module_path.resolve()
        module_id = f"_capx_g1_gateway_{module_name}_{abs(hash(str(resolved)))}"
        spec = importlib.util.spec_from_file_location(module_id, resolved)
        if spec is None or spec.loader is None:
            raise G1GatewayUnavailableError(f"Cannot load LCM type from {resolved}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_id] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise G1GatewayUnavailableError(
                f"Cannot import {module_name}.py from {lcm_types_dir}. "
                "Set CAPX_G1_GATEWAY_LCM_TYPES_DIR or OPENHOMIE_LCM_TYPES_DIR "
                "to the OpenHomie generated lcm_types directory."
            ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise G1GatewayUnavailableError(
            f"{module_name}.py does not define generated LCM class {class_name}."
        ) from exc


class G1GatewayArmActionBridge:
    """Bridge CaP-X single-arm joint targets onto Gateway ``arm_action`` LCM messages.

    The Gateway owns one 14-DoF upper-body command vector where indices 0..6 are
    the left arm and 7..13 are the right arm. This bridge preserves the latest
    observed command state and replaces only the configured arm slice.
    """

    def __init__(
        self,
        *,
        network_interface: str | None = None,
        domain_id: int = 0,
        dry_run: bool = False,
        arm_side: str = "right",
        lcm_url: str = DEFAULT_G1_GATEWAY_LCM_URL,
        lcm_types_dir: str | os.PathLike[str] | None = None,
        arm_channel: str = DEFAULT_G1_GATEWAY_ARM_CHANNEL,
        body_state_channel: str = DEFAULT_G1_GATEWAY_BODY_STATE_CHANNEL,
        imu_channel: str = DEFAULT_G1_GATEWAY_IMU_CHANNEL,
        require_state_before_publish: bool = True,
        initial_arm_action: Any | None = None,
        lcm_client: Any | None = None,
        arm_message_type: type[Any] | None = None,
        body_state_message_type: type[Any] | None = None,
        imu_message_type: type[Any] | None = None,
    ) -> None:
        self.network_interface = (
            network_interface
            or os.environ.get("UNITREE_NETWORK_INTERFACE")
            or os.environ.get("G1_NETWORK_INTERFACE")
            or DEFAULT_G1_NETWORK_INTERFACE
        )
        self.domain_id = int(domain_id)
        self.arm_side = normalize_g1_arm_side(arm_side)
        self.arm_motor_indices = G1_ARM_MOTOR_INDICES_BY_SIDE[self.arm_side]
        self.arm_action_slice = G1_ARM_DUAL_CFG_SLICE_BY_SIDE[self.arm_side]
        self.dry_run = bool(dry_run)
        self.lcm_url = str(lcm_url)
        self.lcm_types_dir = _resolve_lcm_types_dir(lcm_types_dir)
        self.arm_channel = str(arm_channel)
        self.publisher_topic = self.arm_channel
        self.body_state_channel = str(body_state_channel)
        self.state_topic = self.body_state_channel
        self.imu_channel = str(imu_channel)
        self.require_state_before_publish = bool(require_state_before_publish)

        if initial_arm_action is None:
            initial = np.zeros(G1_DUAL_ARM_NUM_JOINTS, dtype=np.float64)
        else:
            initial = np.asarray(initial_arm_action, dtype=np.float64)
            if initial.size != G1_DUAL_ARM_NUM_JOINTS:
                raise ValueError(
                    f"Expected {G1_DUAL_ARM_NUM_JOINTS} initial Gateway arm_action values, "
                    f"got shape {initial.shape}."
                )
            initial = initial.reshape(G1_DUAL_ARM_NUM_JOINTS)

        self._lc = lcm_client
        self._arm_message_type = arm_message_type
        self._body_state_message_type = body_state_message_type
        self._imu_message_type = imu_message_type
        self._body_subscription: Any | None = None
        self._imu_subscription: Any | None = None
        self._connected = False
        self._latest_body_state: Any | None = None
        self._latest_imu_state: Any | None = None
        self._last_command: Any | None = None
        self._last_commanded_joints = np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)
        self._last_observed_joints = np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)
        self._last_observed_all_joints = np.zeros(
            len(G1_LOWCMD_CONTROLLED_MOTOR_INDICES), dtype=np.float64
        )
        self._last_arm_action = initial.copy()
        self._last_body_state_at = 0.0
        self._last_imu_state_at = 0.0
        self._last_body_rpy = np.zeros(3, dtype=np.float64)
        self._last_body_quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._lock = threading.Lock()
        # Python LCM permits only one active handle/handle_timeout call per instance.
        # The arm-hold worker and motion thread share this bridge, so serialize all LCM I/O.
        self._lcm_io_lock = threading.Lock()

    @property
    def last_command(self) -> Any | None:
        return self._last_command

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def has_low_state(self) -> bool:
        with self._lock:
            return self._latest_body_state is not None

    @property
    def has_imu_state(self) -> bool:
        with self._lock:
            return self._latest_imu_state is not None

    def _load_lcm_types(self) -> None:
        if self._arm_message_type is None:
            self._arm_message_type = _load_generated_lcm_type(
                self.lcm_types_dir,
                "arm_action_lcmt",
                "arm_action_lcmt",
            )
        if self._body_state_message_type is None:
            self._body_state_message_type = _load_generated_lcm_type(
                self.lcm_types_dir,
                "body_control_data_lcmt",
                "body_control_data_lcmt",
            )
        if self._imu_message_type is None:
            self._imu_message_type = _load_generated_lcm_type(
                self.lcm_types_dir,
                "state_estimator_lcmt",
                "state_estimator_lcmt",
            )

    def _make_lcm_client(self) -> Any:
        if self._lc is not None:
            return self._lc
        try:
            import lcm  # type: ignore
        except ImportError as exc:
            raise G1GatewayUnavailableError(
                "The Python LCM bindings are not importable. Install the same LCM runtime "
                "used by OpenHomie/WholeBodyGateway before enabling Gateway arm_action mode."
            ) from exc

        self._lc = lcm.LCM(self.lcm_url)
        return self._lc

    def connect(self) -> None:
        """Initialize LCM publisher/subscribers for Gateway arm_action control."""

        if self.dry_run:
            return

        with self._lcm_io_lock:
            if self._connected:
                return
            self._load_lcm_types()
            lc = self._make_lcm_client()
            self._body_subscription = lc.subscribe(self.body_state_channel, self._handle_body_state)
            self._imu_subscription = lc.subscribe(self.imu_channel, self._handle_imu_state)
            self._connected = True

    @staticmethod
    def _decode_message(message_type: type[Any] | None, data: Any) -> Any:
        if message_type is None:
            return data
        if isinstance(data, (bytes, bytearray, memoryview)):
            return message_type.decode(bytes(data))
        if hasattr(data, "read"):
            return message_type.decode(data)
        return data

    def _handle_body_state(self, channel: str, data: Any) -> None:
        del channel
        msg = self._decode_message(self._body_state_message_type, data)
        try:
            q = np.asarray(getattr(msg, "q"), dtype=np.float64).reshape(-1)
        except AttributeError as exc:
            raise RuntimeError("Gateway body_control_data message is missing q[29].") from exc

        if q.size <= max(G1_LOWCMD_CONTROLLED_MOTOR_INDICES):
            raise RuntimeError(
                f"Gateway body_control_data q has {q.size} values; expected at least 29."
            )

        with self._lock:
            self._latest_body_state = msg
            self._last_body_state_at = time.monotonic()
            self._last_observed_all_joints = q[: len(G1_LOWCMD_CONTROLLED_MOTOR_INDICES)].copy()
            self._last_observed_joints = np.asarray(
                [q[i] for i in self.arm_motor_indices],
                dtype=np.float64,
            )
            self._last_arm_action = q[G1_GATEWAY_ARM_MOTOR_SLICE].copy()

    def _handle_imu_state(self, channel: str, data: Any) -> None:
        del channel
        msg = self._decode_message(self._imu_message_type, data)

        with self._lock:
            self._latest_imu_state = msg
            self._last_imu_state_at = time.monotonic()
            rpy = getattr(msg, "rpy", None)
            if rpy is not None:
                rpy_arr = np.asarray(rpy, dtype=np.float64).reshape(-1)
                if rpy_arr.size >= 3:
                    self._last_body_rpy = rpy_arr[:3].copy()

            quat = getattr(msg, "quat", None)
            if quat is not None:
                quat_arr = np.asarray(quat, dtype=np.float64).reshape(-1)
                if quat_arr.size >= 4:
                    quat_arr = quat_arr[:4].copy()
                    norm = float(np.linalg.norm(quat_arr))
                    if norm > 0.0:
                        quat_arr /= norm
                    self._last_body_quaternion_wxyz = quat_arr

    def poll_once(self, timeout_s: float = 0.0) -> bool:
        """Handle at most one pending LCM message."""

        if self.dry_run or self._lc is None:
            return False

        with self._lcm_io_lock:
            try:
                fileno = self._lc.fileno()
                if fileno < 0:
                    return False
                readable, _, _ = select.select([fileno], [], [], max(float(timeout_s), 0.0))
                if not readable:
                    return False
                self._lc.handle()
                return True
            except (OSError, ValueError):
                return False

    def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
        """Wait until a Gateway ``body_control_data`` message has been received."""

        if self.dry_run:
            return True
        if not self._connected:
            self.connect()

        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.has_low_state:
                return True
            self.poll_once(min(0.05, max(0.0, deadline - time.monotonic())))
        return self.has_low_state

    def _make_arm_message(self) -> Any:
        if self._arm_message_type is None:
            self._load_lcm_types()
        if self._arm_message_type is None:
            raise G1GatewayUnavailableError("Gateway arm_action LCM type is not loaded.")
        return self._arm_message_type()

    def build_arm_message(self, joints: Any) -> Any:
        """Build a Gateway ``arm_action_lcmt`` for a 7-DoF configured-arm target."""

        target = as_g1_arm_joints(joints, arm_side=self.arm_side)
        with self._lock:
            has_state = self._latest_body_state is not None
            arm_action = self._last_arm_action.copy()

        if not self.dry_run and self.require_state_before_publish and not has_state:
            raise RuntimeError(
                "Cannot publish Gateway arm_action before receiving body_control_data; "
                "the opposite-arm slice must be preserved from Gateway state."
            )

        arm_action[self.arm_action_slice] = target
        msg = self._make_arm_message()
        msg.act = arm_action.astype(np.float64).tolist()
        return msg

    def publish_joints(self, joints: Any) -> bool:
        """Publish a Gateway arm_action target, or only store it in dry-run mode."""

        target = as_g1_arm_joints(joints, arm_side=self.arm_side)
        if not self.dry_run and not self._connected:
            self.connect()

        if not self.dry_run:
            self.poll_once(0.0)

        msg = self.build_arm_message(target)
        with self._lock:
            self._last_command = msg
            self._last_commanded_joints = target.copy()
            self._last_arm_action = np.asarray(msg.act, dtype=np.float64).copy()

        if self.dry_run:
            return True
        if self._lc is None:
            raise RuntimeError("Gateway LCM client is not initialized.")
        with self._lcm_io_lock:
            self._lc.publish(self.arm_channel, msg.encode())
        return True

    def get_arm_joint_positions(self) -> np.ndarray:
        """Return latest G1 configured-arm joint positions in CaP-X 7-DoF order."""

        if not self.dry_run:
            if not self._connected:
                self.connect()
            self.poll_once(0.0)

        with self._lock:
            if self._latest_body_state is not None:
                return self._last_observed_joints.copy()
            return self._last_commanded_joints.copy()

    def get_body_rpy(self) -> np.ndarray:
        if not self.dry_run:
            if not self._connected:
                self.connect()
            self.poll_once(0.0)
        with self._lock:
            return self._last_body_rpy.copy()

    def get_body_quaternion_wxyz(self) -> np.ndarray:
        if not self.dry_run:
            if not self._connected:
                self.connect()
            self.poll_once(0.0)
        with self._lock:
            return self._last_body_quaternion_wxyz.copy()

    def close(self) -> None:
        with self._lcm_io_lock:
            if self._lc is not None:
                with contextlib.suppress(Exception):
                    if self._body_subscription is not None:
                        self._lc.unsubscribe(self._body_subscription)
                with contextlib.suppress(Exception):
                    if self._imu_subscription is not None:
                        self._lc.unsubscribe(self._imu_subscription)
            self._connected = False
