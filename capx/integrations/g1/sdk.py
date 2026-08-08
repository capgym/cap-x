from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

G1_LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

G1_RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_ARM_JOINT_NAMES: tuple[str, ...] = G1_RIGHT_ARM_JOINT_NAMES
G1_LOWCMD_CONTROLLED_MOTOR_INDICES: tuple[int, ...] = tuple(range(29))
G1_NUM_ARM_JOINTS = 7
G1_DUAL_ARM_NUM_JOINTS = 14
G1_LEFT_ARM_MOTOR_INDICES: tuple[int, ...] = tuple(range(15, 22))
G1_RIGHT_ARM_MOTOR_INDICES: tuple[int, ...] = tuple(range(22, 29))
# Backward-compatible aliases for the original right-arm-only interface.
G1_ARM_MOTOR_INDICES: tuple[int, ...] = G1_RIGHT_ARM_MOTOR_INDICES
G1_ARM_JOINT_NAMES_BY_SIDE: dict[str, tuple[str, ...]] = {
    "left": G1_LEFT_ARM_JOINT_NAMES,
    "right": G1_RIGHT_ARM_JOINT_NAMES,
}
G1_ARM_MOTOR_INDICES_BY_SIDE: dict[str, tuple[int, ...]] = {
    "left": G1_LEFT_ARM_MOTOR_INDICES,
    "right": G1_RIGHT_ARM_MOTOR_INDICES,
}
G1_LEFT_ARM_DUAL_CFG_SLICE = slice(0, 7)
G1_RIGHT_ARM_DUAL_CFG_SLICE = slice(7, 14)
G1_ARM_DUAL_CFG_SLICE_BY_SIDE: dict[str, slice] = {
    "left": G1_LEFT_ARM_DUAL_CFG_SLICE,
    "right": G1_RIGHT_ARM_DUAL_CFG_SLICE,
}
G1_WITH_HAND_NUM_JOINTS = 43
G1_LEFT_ARM_WITH_HAND_CFG_SLICE = slice(36, 43)
G1_RIGHT_ARM_WITH_HAND_CFG_SLICE = slice(29, 36)
G1_ARM_WITH_HAND_CFG_SLICE_BY_SIDE: dict[str, slice] = {
    "left": G1_LEFT_ARM_WITH_HAND_CFG_SLICE,
    "right": G1_RIGHT_ARM_WITH_HAND_CFG_SLICE,
}
DEFAULT_G1_NETWORK_INTERFACE = "enx6c1ff7c1192d"

G1_LOWCMD_KP: tuple[float, ...] = (
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,
    60.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
)
G1_LOWCMD_KD: tuple[float, ...] = (
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)
DEFAULT_G1_ARM_KP = 40.0
DEFAULT_G1_ARM_KD = 1.0

G1_RIGHT_DEX3_HAND_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
G1_RIGHT_DEX3_DDS_HAND_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)
G1_LEFT_DEX3_HAND_JOINT_NAMES: tuple[str, ...] = tuple(
    name.replace("right_", "left_", 1) for name in G1_RIGHT_DEX3_HAND_JOINT_NAMES
)
G1_LEFT_DEX3_DDS_HAND_JOINT_NAMES: tuple[str, ...] = tuple(
    name.replace("right_", "left_", 1) for name in G1_RIGHT_DEX3_DDS_HAND_JOINT_NAMES
)
# Backward-compatible aliases for the original right-hand-only interface.
G1_DEX3_HAND_JOINT_NAMES: tuple[str, ...] = G1_RIGHT_DEX3_HAND_JOINT_NAMES
G1_DEX3_DDS_HAND_JOINT_NAMES: tuple[str, ...] = G1_RIGHT_DEX3_DDS_HAND_JOINT_NAMES
G1_DEX3_HAND_JOINT_NAMES_BY_SIDE: dict[str, tuple[str, ...]] = {
    "left": G1_LEFT_DEX3_HAND_JOINT_NAMES,
    "right": G1_RIGHT_DEX3_HAND_JOINT_NAMES,
}
G1_DEX3_DDS_HAND_JOINT_NAMES_BY_SIDE: dict[str, tuple[str, ...]] = {
    "left": G1_LEFT_DEX3_DDS_HAND_JOINT_NAMES,
    "right": G1_RIGHT_DEX3_DDS_HAND_JOINT_NAMES,
}
G1_NUM_DEX3_HAND_JOINTS = 7
G1_DEX3_SEMANTIC_TO_DDS_INDICES: tuple[int, ...] = tuple(
    G1_RIGHT_DEX3_HAND_JOINT_NAMES.index(name) for name in G1_RIGHT_DEX3_DDS_HAND_JOINT_NAMES
)
G1_DEX3_DDS_TO_SEMANTIC_INDICES: tuple[int, ...] = tuple(
    G1_RIGHT_DEX3_DDS_HAND_JOINT_NAMES.index(name) for name in G1_RIGHT_DEX3_HAND_JOINT_NAMES
)
DEFAULT_G1_DEX3_HAND_KP = 1.5
DEFAULT_G1_DEX3_HAND_KD = 0.1
DEFAULT_G1_DEX3_RIGHT_CMD_TOPIC = "rt/dex3/right/cmd"
DEFAULT_G1_DEX3_RIGHT_STATE_TOPIC = "rt/lf/dex3/right/state"
DEFAULT_G1_DEX3_LEFT_CMD_TOPIC = "rt/dex3/left/cmd"
DEFAULT_G1_DEX3_LEFT_STATE_TOPIC = "rt/dex3/left/state"


class UnitreeSdkUnavailableError(RuntimeError):
    """Raised when the Unitree SDK2 Python package is not importable."""


def normalize_g1_arm_side(side: str) -> str:
    """Validate and normalize a single-arm G1 side selector."""

    normalized = str(side).lower()
    if normalized not in {"left", "right"}:
        raise ValueError("G1 arm side must be left or right.")
    return normalized


def as_g1_arm_joints(joints: Any, *, arm_side: str = "right") -> np.ndarray:
    """Normalize a joint vector to the selected CaP-X G1 arm order."""

    side = normalize_g1_arm_side(arm_side)
    arr = np.asarray(joints, dtype=np.float64)
    if arr.size != G1_NUM_ARM_JOINTS:
        raise ValueError(
            f"Expected {G1_NUM_ARM_JOINTS} G1 {side} arm joints in order "
            f"{G1_ARM_JOINT_NAMES_BY_SIDE[side]}, got shape {arr.shape}."
        )
    return arr.reshape(G1_NUM_ARM_JOINTS)


def as_g1_dex3_hand_joints(joints: Any, *, hand_side: str = "right") -> np.ndarray:
    """Normalize Dex3 joints in the selected hand semantic order."""

    side = normalize_g1_arm_side(hand_side)
    arr = np.asarray(joints, dtype=np.float64)
    if arr.size != G1_NUM_DEX3_HAND_JOINTS:
        raise ValueError(
            f"Expected {G1_NUM_DEX3_HAND_JOINTS} Dex3 {side} hand joints in order "
            f"{G1_DEX3_HAND_JOINT_NAMES_BY_SIDE[side]}, got shape {arr.shape}."
        )
    return arr.reshape(G1_NUM_DEX3_HAND_JOINTS)


def dex3_semantic_to_dds_joints(joints: Any, *, hand_side: str = "right") -> np.ndarray:
    target = as_g1_dex3_hand_joints(joints, hand_side=hand_side)
    return target[np.asarray(G1_DEX3_SEMANTIC_TO_DDS_INDICES, dtype=np.int64)]


def dex3_dds_to_semantic_joints(joints: Any, *, hand_side: str = "right") -> np.ndarray:
    side = normalize_g1_arm_side(hand_side)
    arr = np.asarray(joints, dtype=np.float64)
    if arr.size != G1_NUM_DEX3_HAND_JOINTS:
        raise ValueError(
            f"Expected {G1_NUM_DEX3_HAND_JOINTS} Dex3 {side} DDS joints in order "
            f"{G1_DEX3_DDS_HAND_JOINT_NAMES_BY_SIDE[side]}, got shape {arr.shape}."
        )
    return arr.reshape(G1_NUM_DEX3_HAND_JOINTS)[np.asarray(G1_DEX3_DDS_TO_SEMANTIC_INDICES, dtype=np.int64)]


def dex3_grasp_joints(trigger: float, squeeze: float, *, hand_side: str = "right") -> np.ndarray:
    """Map IsaacLab-style trigger/squeeze inputs to a selected Dex3 hand.

    The left hand mirrors the right hand flexion directions according to the G1
    URDF joint limits while retaining the same trigger/squeeze semantics.
    """

    trigger_f = float(np.clip(trigger, 0.0, 1.0))
    squeeze_f = float(np.clip(squeeze, 0.0, 1.0))
    thumb_button = max(trigger_f, squeeze_f)
    thumb_angle = -thumb_button
    thumb_rotation = -(0.5 * trigger_f - 0.5 * squeeze_f)
    right_target = np.asarray(
        [
            thumb_rotation,
            thumb_angle * 0.4,
            thumb_angle * 0.7,
            trigger_f,
            trigger_f,
            squeeze_f,
            squeeze_f,
        ],
        dtype=np.float64,
    )
    if normalize_g1_arm_side(hand_side) == "right":
        return right_target
    return right_target * np.asarray([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])


class G1ArmSdkBridge:
    """Bridge from CaP-X 7-DoF single-arm commands to Unitree G1 lowcmd DDS messages."""

    def __init__(
        self,
        *,
        network_interface: str | None = None,
        domain_id: int = 0,
        dry_run: bool = False,
        arm_side: str = "right",
        publisher_topic: str = "rt/lowcmd",
        state_topic: str = "rt/lowstate",
        release_motion_mode: bool = True,
        motion_switch_timeout_s: float = 5.0,
        kp: float = DEFAULT_G1_ARM_KP,
        kd: float = DEFAULT_G1_ARM_KD,
        dq: float = 0.0,
        tau: float = 0.0,
        write_timeout: float | None = None,
        low_cmd_factory: Callable[[], Any] | None = None,
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
        self.dry_run = bool(dry_run)
        self.publisher_topic = publisher_topic
        self.state_topic = state_topic
        self.release_motion_mode = bool(release_motion_mode)
        self.motion_switch_timeout_s = float(motion_switch_timeout_s)
        self.kp = float(kp)
        self.kd = float(kd)
        self.dq = float(dq)
        self.tau = float(tau)
        self.write_timeout = write_timeout
        self._low_cmd_factory = low_cmd_factory

        self._publisher: Any | None = None
        self._subscriber: Any | None = None
        self._crc: Any | None = None
        self._connected = False
        self._latest_state: Any | None = None
        self._last_command: Any | None = None
        self._last_commanded_joints = np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)
        self._last_observed_joints = np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)
        self._last_observed_all_joints = np.zeros(len(G1_LOWCMD_CONTROLLED_MOTOR_INDICES), dtype=np.float64)
        self._mode_machine = 0
        self._lock = threading.Lock()

    @property
    def last_command(self) -> Any | None:
        return self._last_command

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def has_low_state(self) -> bool:
        with self._lock:
            return self._latest_state is not None

    def _load_sdk(self) -> dict[str, Any]:
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # type: ignore
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # type: ignore
            from unitree_sdk2py.utils.crc import CRC  # type: ignore
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (  # type: ignore
                MotionSwitcherClient,
            )
        except ImportError as exc:
            raise UnitreeSdkUnavailableError(
                "unitree_sdk2py is not installed. Install it with "
                "`.venv/bin/python -m pip install -e /home/peilab/unitree_sdk2_python`."
            ) from exc

        return {
            "ChannelFactoryInitialize": ChannelFactoryInitialize,
            "ChannelPublisher": ChannelPublisher,
            "ChannelSubscriber": ChannelSubscriber,
            "LowCmd_": LowCmd_,
            "LowState_": LowState_,
            "low_cmd_factory": unitree_hg_msg_dds__LowCmd_,
            "CRC": CRC,
            "MotionSwitcherClient": MotionSwitcherClient,
        }

    def _make_low_cmd(self) -> Any:
        if self._low_cmd_factory is not None:
            return self._low_cmd_factory()

        sdk = self._load_sdk()
        self._low_cmd_factory = sdk["low_cmd_factory"]
        return self._low_cmd_factory()

    def connect(self) -> None:
        """Initialize DDS publisher/subscriber for G1 lowcmd control."""

        if self.dry_run or self._connected:
            return

        sdk = self._load_sdk()
        sdk["ChannelFactoryInitialize"](self.domain_id, self.network_interface)
        if self.release_motion_mode:
            self._release_motion_mode(sdk)

        self._publisher = sdk["ChannelPublisher"](self.publisher_topic, sdk["LowCmd_"])
        self._publisher.Init()

        self._subscriber = sdk["ChannelSubscriber"](self.state_topic, sdk["LowState_"])
        self._subscriber.Init(self._handle_low_state, 10)

        self._crc = sdk["CRC"]()
        self._connected = True

    def _release_motion_mode(self, sdk: dict[str, Any]) -> None:
        motion_switcher_cls = sdk.get("MotionSwitcherClient")
        if motion_switcher_cls is None:
            return

        client = motion_switcher_cls()
        client.SetTimeout(self.motion_switch_timeout_s)
        client.Init()

        deadline = time.monotonic() + self.motion_switch_timeout_s
        while True:
            status, result = client.CheckMode()
            if status != 0:
                raise RuntimeError(f"Unitree MotionSwitcher CheckMode failed with status {status}.")

            mode_name = result.get("name", "") if isinstance(result, dict) else ""
            if not mode_name:
                return

            print(f"[g1-lowcmd] releasing motion mode {mode_name!r} before lowcmd control")
            release_status, _ = client.ReleaseMode()
            if release_status != 0:
                raise RuntimeError(
                    f"Unitree MotionSwitcher ReleaseMode failed with status {release_status}."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out releasing Unitree motion mode {mode_name!r} before lowcmd control."
                )
            time.sleep(0.2)

    def _handle_low_state(self, msg: Any) -> None:
        with self._lock:
            self._latest_state = msg
            self._mode_machine = int(getattr(msg, "mode_machine", 0))
            try:
                self._last_observed_joints = np.asarray(
                    [msg.motor_state[i].q for i in self.arm_motor_indices],
                    dtype=np.float64,
                )
                self._last_observed_all_joints = np.asarray(
                    [msg.motor_state[i].q for i in G1_LOWCMD_CONTROLLED_MOTOR_INDICES],
                    dtype=np.float64,
                )
            except (AttributeError, IndexError):
                pass

    def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
        """Wait until at least one G1 lowstate message has been received."""

        if self.dry_run:
            return True
        if not self._connected:
            self.connect()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.has_low_state:
                return True
            time.sleep(0.01)
        return self.has_low_state

    def build_low_cmd(self, joints: Any) -> Any:
        """Build a Unitree ``LowCmd_`` for the selected 7-DoF G1 arm target."""

        target = as_g1_arm_joints(joints, arm_side=self.arm_side)
        cmd = self._make_low_cmd()
        if len(cmd.motor_cmd) <= max(G1_LOWCMD_CONTROLLED_MOTOR_INDICES):
            raise ValueError(
                f"LowCmd_ has {len(cmd.motor_cmd)} motor commands; "
                f"G1 lowcmd requires index {max(G1_LOWCMD_CONTROLLED_MOTOR_INDICES)}."
            )

        if hasattr(cmd, "mode_pr"):
            cmd.mode_pr = 0
        if hasattr(cmd, "mode_machine"):
            cmd.mode_machine = self._mode_machine

        with self._lock:
            state = self._latest_state
            all_joints = self._last_observed_all_joints.copy()
        if not self.dry_run and state is None:
            raise RuntimeError(
                "Cannot publish G1 lowcmd before receiving rt/lowstate; "
                "non-selected-arm joints must be preserved from the robot state."
            )

        for motor_idx in G1_LOWCMD_CONTROLLED_MOTOR_INDICES:
            motor = cmd.motor_cmd[motor_idx]
            if hasattr(motor, "mode"):
                motor.mode = 1
            motor.q = float(all_joints[motor_idx])
            motor.dq = self.dq
            motor.kp = float(G1_LOWCMD_KP[motor_idx])
            motor.kd = float(G1_LOWCMD_KD[motor_idx])
            motor.tau = self.tau

        for value, motor_idx in zip(target, self.arm_motor_indices, strict=True):
            motor = cmd.motor_cmd[motor_idx]
            motor.q = float(value)
            motor.dq = self.dq
            motor.kp = self.kp
            motor.kd = self.kd
            motor.tau = self.tau

        if self._crc is not None:
            cmd.crc = self._crc.Crc(cmd)

        return cmd

    def publish_joints(self, joints: Any) -> bool:
        """Publish a G1 arm joint target, or only store it when ``dry_run`` is true."""

        target = as_g1_arm_joints(joints, arm_side=self.arm_side)
        if not self.dry_run and not self._connected:
            self.connect()

        cmd = self.build_low_cmd(target)
        with self._lock:
            self._last_command = cmd
            self._last_commanded_joints = target.copy()

        if self.dry_run:
            return True
        if self._publisher is None:
            raise RuntimeError("G1 lowcmd publisher is not initialized.")
        if self.write_timeout is None:
            return bool(self._publisher.Write(cmd))
        return bool(self._publisher.Write(cmd, self.write_timeout))

    def get_arm_joint_positions(self) -> np.ndarray:
        """Return latest selected G1 arm joint positions in CaP-X 7-DoF order."""

        with self._lock:
            state = self._latest_state
            fallback = (
                self._last_commanded_joints.copy()
                if self.dry_run
                else self._last_observed_joints.copy()
            )

        if state is None:
            return fallback

        try:
            return np.asarray(
                [state.motor_state[i].q for i in self.arm_motor_indices],
                dtype=np.float64,
            )
        except (AttributeError, IndexError) as exc:
            raise RuntimeError("Received Unitree LowState_ without expected G1 motor_state data.") from exc

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._subscriber is not None:
                self._subscriber.Close()
        with contextlib.suppress(Exception):
            if self._publisher is not None:
                self._publisher.Close()
        self._connected = False


class G1Dex3HandBridge:
    """Bridge from CaP-X semantic hand commands to either Unitree G1 Dex3 hand."""

    def __init__(
        self,
        *,
        network_interface: str | None = None,
        domain_id: int = 0,
        dry_run: bool = False,
        hand_side: str = "right",
        publisher_topic: str | None = None,
        state_topic: str | None = None,
        kp: float = DEFAULT_G1_DEX3_HAND_KP,
        kd: float = DEFAULT_G1_DEX3_HAND_KD,
        dq: float = 0.0,
        tau: float = 0.0,
        write_timeout: float | None = None,
        hand_cmd_factory: Callable[[], Any] | None = None,
    ) -> None:
        side = normalize_g1_arm_side(hand_side)

        self.network_interface = (
            network_interface
            or os.environ.get("UNITREE_NETWORK_INTERFACE")
            or os.environ.get("G1_NETWORK_INTERFACE")
            or DEFAULT_G1_NETWORK_INTERFACE
        )
        self.domain_id = int(domain_id)
        self.dry_run = bool(dry_run)
        self.hand_side = side
        self.publisher_topic = publisher_topic or (
            DEFAULT_G1_DEX3_LEFT_CMD_TOPIC if side == "left" else DEFAULT_G1_DEX3_RIGHT_CMD_TOPIC
        )
        self.state_topic = state_topic or (
            DEFAULT_G1_DEX3_LEFT_STATE_TOPIC if side == "left" else DEFAULT_G1_DEX3_RIGHT_STATE_TOPIC
        )
        self.kp = float(kp)
        self.kd = float(kd)
        self.dq = float(dq)
        self.tau = float(tau)
        self.write_timeout = write_timeout
        self._hand_cmd_factory = hand_cmd_factory

        self._publisher: Any | None = None
        self._subscriber: Any | None = None
        self._connected = False
        self._latest_state: Any | None = None
        self._last_command: Any | None = None
        self._last_commanded_joints = np.zeros(G1_NUM_DEX3_HAND_JOINTS, dtype=np.float64)
        self._last_observed_joints = np.zeros(G1_NUM_DEX3_HAND_JOINTS, dtype=np.float64)
        self._lock = threading.Lock()

    @property
    def last_command(self) -> Any | None:
        return self._last_command

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def has_hand_state(self) -> bool:
        with self._lock:
            return self._latest_state is not None

    def _load_sdk(self) -> dict[str, Any]:
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_  # type: ignore
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_  # type: ignore
        except ImportError as exc:
            raise UnitreeSdkUnavailableError(
                "unitree_sdk2py is not installed. Install it with "
                ".venv/bin/python -m pip install -e /home/peilab/unitree_sdk2_python."
            ) from exc

        return {
            "ChannelFactoryInitialize": ChannelFactoryInitialize,
            "ChannelPublisher": ChannelPublisher,
            "ChannelSubscriber": ChannelSubscriber,
            "HandCmd_": HandCmd_,
            "HandState_": HandState_,
            "hand_cmd_factory": unitree_hg_msg_dds__HandCmd_,
        }

    def _make_hand_cmd(self) -> Any:
        if self._hand_cmd_factory is not None:
            return self._hand_cmd_factory()

        sdk = self._load_sdk()
        self._hand_cmd_factory = sdk["hand_cmd_factory"]
        return self._hand_cmd_factory()

    @staticmethod
    def _init_channel(channel: Any, *args: Any) -> None:
        if hasattr(channel, "Init"):
            channel.Init(*args)
        elif hasattr(channel, "InitChannel"):
            channel.InitChannel(*args)
        else:
            raise RuntimeError("Unitree DDS channel object has neither Init nor InitChannel.")

    def connect(self) -> None:
        """Initialize DDS publisher/subscriber for the selected Dex3 hand."""

        if self.dry_run or self._connected:
            return

        sdk = self._load_sdk()
        sdk["ChannelFactoryInitialize"](self.domain_id, self.network_interface)

        self._publisher = sdk["ChannelPublisher"](self.publisher_topic, sdk["HandCmd_"])
        self._init_channel(self._publisher)

        self._subscriber = sdk["ChannelSubscriber"](self.state_topic, sdk["HandState_"])
        self._init_channel(self._subscriber, self._handle_hand_state, 10)
        self._connected = True

    def _handle_hand_state(self, msg: Any) -> None:
        with self._lock:
            self._latest_state = msg
            try:
                dds_joints = np.asarray(
                    [msg.motor_state[i].q for i in range(G1_NUM_DEX3_HAND_JOINTS)],
                    dtype=np.float64,
                )
                self._last_observed_joints = dex3_dds_to_semantic_joints(dds_joints, hand_side=self.hand_side)
            except (AttributeError, IndexError):
                pass

    def wait_for_hand_state(self, timeout_s: float = 3.0) -> bool:
        """Wait until at least one Dex3 HandState_ message has been received."""

        if self.dry_run:
            return True
        if not self._connected:
            self.connect()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.has_hand_state:
                return True
            time.sleep(0.01)
        return self.has_hand_state

    @staticmethod
    def _motor_mode(motor_id: int, *, timeout: bool = False) -> int:
        status = 0x01
        return (motor_id & 0x0F) | ((status & 0x07) << 4) | ((0x01 if timeout else 0x00) << 7)

    def build_hand_cmd(self, joints: Any) -> Any:
        """Build a Unitree HandCmd_ for the selected Dex3 hand target."""

        target = as_g1_dex3_hand_joints(joints, hand_side=self.hand_side)
        dds_target = dex3_semantic_to_dds_joints(target, hand_side=self.hand_side)
        cmd = self._make_hand_cmd()
        if len(cmd.motor_cmd) < G1_NUM_DEX3_HAND_JOINTS:
            raise ValueError(
                f"HandCmd_ has {len(cmd.motor_cmd)} motor commands; "
                f"Dex3 requires {G1_NUM_DEX3_HAND_JOINTS}."
            )

        for motor_idx, value in enumerate(dds_target):
            motor = cmd.motor_cmd[motor_idx]
            motor.mode = self._motor_mode(motor_idx)
            motor.q = float(value)
            motor.dq = self.dq
            motor.kp = self.kp
            motor.kd = self.kd
            motor.tau = self.tau

        return cmd

    def publish_hand_joints(self, joints: Any) -> bool:
        """Publish a Dex3 hand target in CaP-X semantic order."""

        target = as_g1_dex3_hand_joints(joints, hand_side=self.hand_side)
        if not self.dry_run and not self._connected:
            self.connect()

        cmd = self.build_hand_cmd(target)
        with self._lock:
            self._last_command = cmd
            self._last_commanded_joints = target.copy()

        if self.dry_run:
            return True
        if self._publisher is None:
            raise RuntimeError("G1 Dex3 hand publisher is not initialized.")
        if self.write_timeout is None:
            return bool(self._publisher.Write(cmd))
        return bool(self._publisher.Write(cmd, self.write_timeout))

    def get_hand_joint_positions(self) -> np.ndarray:
        """Return latest selected Dex3 hand joints in CaP-X semantic order."""

        with self._lock:
            if self.dry_run:
                return self._last_commanded_joints.copy()
            return self._last_observed_joints.copy()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._subscriber is not None:
                self._subscriber.Close()
        with contextlib.suppress(Exception):
            if self._publisher is not None:
                self._publisher.Close()
        self._connected = False
