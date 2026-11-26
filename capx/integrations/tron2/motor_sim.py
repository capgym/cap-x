"""Legacy TRON2 motor and MROS behavior for a MuJoCo physics owner.

Call :meth:`Tron2MotorSim.apply_control` before stepping the model and
:meth:`Tron2MotorSim.publish_state_if_due` afterward, from the physics owner.
The SDK callback only replaces the buffered command. It never touches MuJoCo.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import mujoco
import numpy as np

from .official_client import PROTOCOL_JOINT_NAMES

_LOG = logging.getLogger(__name__)
_AXES = len(PROTOCOL_JOINT_NAMES)
_CLOCK_EPSILON = 1e-9


class MotorTransport(Protocol):
    """Exchange motor commands and feedback without touching the physics state."""

    def start(self, on_command: Callable[[Any], None]) -> None: ...

    def publish(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, sequence: int
    ) -> None: ...

    def close(self) -> None: ...


class LimxMotorTransport:
    """Protocol-matched generic limxsdk channels on the local simulation bus."""

    def __init__(self) -> None:
        self._robot = None
        self._publisher = None
        self._subscription = None
        self._state = None

    def start(self, on_command: Callable[[Any], None]) -> None:
        # Apply the same simulation environment as the legacy launchers before
        # the SDK loads its native libraries. The SDK endpoint is loopback only.
        environment = {"MROS_LOCALHOST_ONLY": "1", "IS_SIM": "1", "MROS_SIM_TIME": "0"}
        for name, value in environment.items():
            if os.environ.get(name, value) != value:
                raise RuntimeError(f"The local TRON2 motor transport requires {name}={value}")
        for name, value in environment.items():
            os.environ.setdefault(name, value)

        try:
            from limxsdk.msg.controller_msgs import JointCmd, JointState
            from limxsdk.robot.Robot import Robot
            from limxsdk.robot.RobotType import RobotType
        except ImportError as exc:
            raise RuntimeError("TRON2 motor mode requires limxsdk 4.1.2 with generic topic support") from exc

        self._robot = Robot(RobotType.Tron2)
        if not self._robot.init("127.0.0.1"):
            raise RuntimeError("limxsdk could not initialize the local TRON2 simulator bus")

        self._subscription = self._robot.subscribe(JointCmd, "/motor/cmd", on_command)
        if not self._subscription.valid:
            raise RuntimeError("limxsdk could not subscribe to /motor/cmd")
        self._publisher = self._robot.publish(JointState, "/motor/state")
        if not self._publisher.valid:
            raise RuntimeError("limxsdk could not advertise /motor/state")
        self._state = JointState()
        self._state.names = list(PROTOCOL_JOINT_NAMES)
        self._state.vd = [0.0] * _AXES
        self._state.na = _AXES
        self._state.header.frame_id = "mujoco"

    def publish(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, sequence: int) -> None:
        if self._publisher is None or self._state is None:
            raise RuntimeError("The local TRON2 motor transport is not started")
        state = self._state
        state.header.seq = sequence & 0xFFFFFFFF
        state.header.stamp.sec, state.header.stamp.nsec = divmod(time.time_ns(), 1_000_000_000)
        state.q = q.tolist()
        state.v = v.tolist()
        state.tau = tau.tolist()
        result = self._publisher.publish(state)
        if result is False or (isinstance(result, int) and result < 0):
            raise RuntimeError("limxsdk failed to publish /motor/state")

    def close(self) -> None:
        handles = (self._subscription, self._publisher)
        self._subscription = self._publisher = self._state = None
        try:
            for handle in handles:
                close = getattr(handle, "close", None)
                if close is not None:
                    close()
        finally:
            self._robot = None


@dataclass(frozen=True)
class _MotorCommand:
    q_des: np.ndarray
    v_des: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    received_at: float


def _read_field(values: Any, default: float | None = None) -> np.ndarray:
    supplied = np.asarray(values[:_AXES], dtype=np.float64)
    if supplied.ndim != 1 or not np.isfinite(supplied).all():
        raise ValueError("motor command fields must contain finite scalars")
    if default is None:
        return supplied.copy()
    result = np.full(_AXES, default)
    result[: supplied.size] = supplied
    return result


class Tron2MotorSim:
    """Drive the official 16 axes and report state from the same MuJoCo world.

    The caller owns the model, data, and physics stepping. A nonpositive
    timeout holds the last command, matching the legacy simulator.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        state_rate_hz: float = 300.0,
        command_timeout_sec: float = 0.0,
        transport: MotorTransport | None = None,
    ) -> None:
        if not math.isfinite(state_rate_hz) or state_rate_hz <= 0:
            raise ValueError("state_rate_hz must be finite and positive")
        if not math.isfinite(command_timeout_sec):
            raise ValueError("command_timeout_sec must be finite")
        if state_rate_hz * model.opt.timestep > 1.0 + _CLOCK_EPSILON:
            raise ValueError("State publish rate exceeds MuJoCo physics rate")

        joint_ids = np.array([model.joint(name).id for name in PROTOCOL_JOINT_NAMES])
        motor_ids = []
        for name, joint_id in zip(PROTOCOL_JOINT_NAMES, joint_ids, strict=True):
            motors = np.flatnonzero(
                (model.actuator_trnid[:, 0] == joint_id)
                & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            )
            if len(motors) != 1:
                raise ValueError(f"Expected one direct joint motor for {name}, found {len(motors)}")
            motor_ids.append(int(motors[0]))

        self._data = data
        self._qpos_ids = model.jnt_qposadr[joint_ids].copy()
        self._dof_ids = model.jnt_dofadr[joint_ids].copy()
        self._motor_ids = np.asarray(motor_ids)
        self._lower, self._upper = model.actuator_ctrlrange[self._motor_ids].T.copy()
        if not np.isfinite([self._lower, self._upper]).all() or np.any(self._lower >= self._upper):
            raise ValueError("Each TRON2 motor requires a finite, nonempty ctrlrange")

        self._transport = transport if transport is not None else LimxMotorTransport()
        self._state_period = 1.0 / state_rate_hz
        self._command_timeout_sec = command_timeout_sec
        self._command_lock = threading.Lock()
        self._command: _MotorCommand | None = None
        self._command_count = 0
        self._invalid_command_count = 0
        self._state_count = 0
        self._started = False
        self._last_state_time = -math.inf
        self._next_state_time = 0.0

    @property
    def command_count(self) -> int:
        with self._command_lock:
            return self._command_count

    @property
    def invalid_command_count(self) -> int:
        with self._command_lock:
            return self._invalid_command_count

    @property
    def last_command_age_sec(self) -> float | None:
        with self._command_lock:
            received_at = self._command.received_at if self._command is not None else None
        return time.monotonic() - received_at if received_at is not None else None

    @property
    def state_count(self) -> int:
        return self._state_count

    def has_arm_position_control(self) -> bool:
        """Whether the controller has enabled position feedback for both arms."""
        with self._command_lock:
            command = self._command
        arm_axes = 14
        return bool(command is not None and command.q_des.size >= arm_axes
                    and np.all(command.kp[:arm_axes] > 0.0))

    def start(self) -> None:
        if self._started:
            raise RuntimeError("TRON2 motor simulation is already started")
        try:
            self._transport.start(self._receive_command)
            self._started = True
            self._next_state_time = float(self._data.time)
            self.publish_state_if_due()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            self._transport.close()
        finally:
            self._started = False
            with self._command_lock:
                self._command = None

    def _receive_command(self, message: Any) -> None:
        received_at = time.monotonic()
        # The legacy simulator consumes command arrays in fixed wire order. It
        # overlays any supplied prefix on defaults for the remaining axes.
        try:
            command = _MotorCommand(
                q_des=_read_field(message.q),
                v_des=_read_field(message.v, 0.0),
                tau_ff=_read_field(message.tau, 0.0),
                kp=_read_field(message.kp, 0.0),
                kd=_read_field(message.kd, 1.0),
                received_at=received_at,
            )
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            with self._command_lock:
                self._invalid_command_count += 1
                first_invalid = self._invalid_command_count == 1
            if first_invalid:
                _LOG.warning("Ignoring malformed TRON2 motor command: %s", exc)
            return

        with self._command_lock:
            self._command = command
            self._command_count += 1

    def apply_control(self) -> None:
        """Write all arm/head motor torques for the next MuJoCo physics step."""
        with self._command_lock:
            command = self._command
        velocity = self._data.qvel[self._dof_ids]
        timeout = self._command_timeout_sec
        if command is None or (timeout > 0 and time.monotonic() - command.received_at > timeout):
            torque = -velocity
        else:
            torque = command.kd * (command.v_des - velocity) + command.tau_ff
            size = command.q_des.size
            positions = self._data.qpos[self._qpos_ids[:size]]
            torque[:size] += command.kp[:size] * (command.q_des - positions)
        self._data.ctrl[self._motor_ids] = np.clip(torque, self._lower, self._upper)

    def publish_state_if_due(self, *, clock_time_sec: float | None = None) -> bool:
        """Publish current feedback at physics or startup clock deadlines."""
        if not self._started:
            raise RuntimeError("TRON2 motor simulation is not started")
        now = float(self._data.time) if clock_time_sec is None else clock_time_sec
        if now < self._last_state_time - _CLOCK_EPSILON:
            self._next_state_time = now
        elapsed = now + _CLOCK_EPSILON - self._next_state_time
        if elapsed < 0:
            return False

        self._transport.publish(
            self._data.qpos[self._qpos_ids],
            self._data.qvel[self._dof_ids],
            self._data.qfrc_actuator[self._dof_ids],
            self._state_count,
        )
        self._last_state_time = now
        self._next_state_time += (math.floor(elapsed / self._state_period) + 1) * self._state_period
        self._state_count += 1
        return True
