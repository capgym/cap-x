from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def _load_g1_sdk_module():
    module_path = Path(__file__).resolve().parents[1] / "capx" / "integrations" / "g1" / "sdk.py"
    spec = importlib.util.spec_from_file_location("_capx_g1_sdk_diagnostic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load G1 SDK module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_G1_SDK = _load_g1_sdk_module()
DEFAULT_G1_NETWORK_INTERFACE = _G1_SDK.DEFAULT_G1_NETWORK_INTERFACE
DEFAULT_STATE_TOPICS = {
    "left": _G1_SDK.DEFAULT_G1_DEX3_LEFT_STATE_TOPIC,
    "right": _G1_SDK.DEFAULT_G1_DEX3_RIGHT_STATE_TOPIC,
}
DEFAULT_CMD_TOPICS = {
    "left": _G1_SDK.DEFAULT_G1_DEX3_LEFT_CMD_TOPIC,
    "right": _G1_SDK.DEFAULT_G1_DEX3_RIGHT_CMD_TOPIC,
}

DDS_JOINT_NAMES: tuple[str, ...] = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)
FINGER_TO_DDS_INDICES: dict[str, tuple[int, ...]] = {
    "thumb": (0, 1, 2),
    "middle": (3, 4),
    "index": (5, 6),
}

# Limits copied from Unitree's g1_move_hands_example.py, in HandState_/HandCmd_ DDS order.
RIGHT_DDS_LIMITS: tuple[tuple[float, float], ...] = (
    (-1.05, 1.05),
    (-1.05, 0.74),
    (-1.75, 0.0),
    (0.0, 1.57),
    (0.0, 1.75),
    (0.0, 1.57),
    (0.0, 1.75),
)
LEFT_DDS_LIMITS: tuple[tuple[float, float], ...] = (
    (-1.05, 1.05),
    (-0.72, 1.05),
    (0.0, 1.75),
    (-1.57, 0.0),
    (-1.75, 0.0),
    (-1.57, 0.0),
    (-1.75, 0.0),
)
DDS_LIMITS_BY_SIDE = {"left": LEFT_DDS_LIMITS, "right": RIGHT_DDS_LIMITS}


@dataclass(frozen=True)
class DiagnosticLimits:
    min_samples: int = 2
    min_state_rate_hz: float = 10.0
    max_motor_temperature_c: float = 70.0
    max_abs_tau: float = 0.5
    min_pressure_delta: float = 0.05
    max_following_error_rad: float = 0.15
    min_motion_fraction: float = 0.5


@dataclass(frozen=True)
class HandSample:
    received_at_s: float
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    motor_temperature_c: np.ndarray
    motor_voltage_v: np.ndarray
    motor_errors: np.ndarray
    pressure: np.ndarray
    pressure_temperature_c: np.ndarray
    pressure_lost: np.ndarray
    power_v: float
    power_a: float
    system_v: float
    device_v: float
    hand_errors: np.ndarray


@dataclass
class DiagnosticReport:
    passed: bool
    sample_count: int
    state_rate_hz: float | None
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pressure_delta_by_finger: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionResult:
    joint: str
    commanded_delta_rad: float
    observed_delta_rad: float
    following_error_rad: float
    return_error_rad: float
    passed: bool
    failures: tuple[str, ...]


def _numeric_vector(value: Any, size: int, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.full(size, np.nan, dtype=np.float64)
    try:
        incoming = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return result.astype(dtype, copy=False)
    copy_size = min(size, incoming.size)
    result[:copy_size] = incoming[:copy_size]
    return result.astype(dtype, copy=False)


def extract_hand_sample(msg: Any, *, received_at_s: float | None = None) -> HandSample:
    """Convert a Unitree HandState_ into fixed-shape arrays for deterministic checks."""

    motors = list(getattr(msg, "motor_state", []))
    q = np.full(7, np.nan, dtype=np.float64)
    dq = np.full(7, np.nan, dtype=np.float64)
    tau_est = np.full(7, np.nan, dtype=np.float64)
    motor_temperature = np.full((7, 2), np.nan, dtype=np.float64)
    motor_voltage = np.full(7, np.nan, dtype=np.float64)
    motor_errors = np.full(7, -1, dtype=np.int64)
    for idx, motor in enumerate(motors[:7]):
        q[idx] = float(getattr(motor, "q", np.nan))
        dq[idx] = float(getattr(motor, "dq", np.nan))
        tau_est[idx] = float(getattr(motor, "tau_est", np.nan))
        motor_temperature[idx] = _numeric_vector(getattr(motor, "temperature", []), 2)
        motor_voltage[idx] = float(getattr(motor, "vol", np.nan))
        motor_errors[idx] = int(getattr(motor, "motorstate", -1))

    pressure = np.full((7, 12), np.nan, dtype=np.float64)
    pressure_temperature = np.full((7, 12), np.nan, dtype=np.float64)
    pressure_lost = np.full(7, -1, dtype=np.int64)
    pressure_states = list(getattr(msg, "press_sensor_state", []))
    for idx, sensor in enumerate(pressure_states[:7]):
        pressure[idx] = _numeric_vector(getattr(sensor, "pressure", []), 12)
        pressure_temperature[idx] = _numeric_vector(getattr(sensor, "temperature", []), 12)
        pressure_lost[idx] = int(getattr(sensor, "lost", -1))

    return HandSample(
        received_at_s=time.monotonic() if received_at_s is None else float(received_at_s),
        q=q,
        dq=dq,
        tau_est=tau_est,
        motor_temperature_c=motor_temperature,
        motor_voltage_v=motor_voltage,
        motor_errors=motor_errors,
        pressure=pressure,
        pressure_temperature_c=pressure_temperature,
        pressure_lost=pressure_lost,
        power_v=float(getattr(msg, "power_v", np.nan)),
        power_a=float(getattr(msg, "power_a", np.nan)),
        system_v=float(getattr(msg, "system_v", np.nan)),
        device_v=float(getattr(msg, "device_v", np.nan)),
        hand_errors=_numeric_vector(getattr(msg, "error", []), 2, dtype=np.int64),
    )


def _finger_indices(excluded_fingers: set[str]) -> dict[str, tuple[int, ...]]:
    unknown = excluded_fingers.difference(FINGER_TO_DDS_INDICES)
    if unknown:
        raise ValueError(f"Unknown fingers: {sorted(unknown)}")
    return {
        finger: indices
        for finger, indices in FINGER_TO_DDS_INDICES.items()
        if finger not in excluded_fingers
    }


def evaluate_state_samples(
    samples: Sequence[HandSample],
    *,
    limits: DiagnosticLimits,
    require_pressure_response: bool = False,
    excluded_fingers: set[str] | None = None,
) -> DiagnosticReport:
    """Evaluate communication, motor feedback, and pressure arrays without sending commands."""

    excluded = set() if excluded_fingers is None else set(excluded_fingers)
    intact_fingers = _finger_indices(excluded)
    intact_motor_indices = {index for indices in intact_fingers.values() for index in indices}
    failures: list[str] = []
    warnings: list[str] = []
    rate_hz: float | None = None

    if len(samples) < limits.min_samples:
        failures.append(f"received {len(samples)} state samples; need at least {limits.min_samples}")
        return DiagnosticReport(False, len(samples), None, failures=failures)

    elapsed = samples[-1].received_at_s - samples[0].received_at_s
    if elapsed <= 0.0:
        failures.append("state timestamps did not advance")
    else:
        rate_hz = (len(samples) - 1) / elapsed
        if rate_hz < limits.min_state_rate_hz:
            failures.append(
                f"state rate {rate_hz:.1f} Hz is below {limits.min_state_rate_hz:.1f} Hz"
            )

    q = np.stack([sample.q for sample in samples])
    dq = np.stack([sample.dq for sample in samples])
    tau = np.stack([sample.tau_est for sample in samples])
    motor_temperature = np.stack([sample.motor_temperature_c for sample in samples])
    motor_voltage = np.stack([sample.motor_voltage_v for sample in samples])
    pressure = np.stack([sample.pressure for sample in samples])
    pressure_temperature = np.stack([sample.pressure_temperature_c for sample in samples])

    for label, values in (
        ("motor position", q),
        ("motor velocity", dq),
        ("estimated torque", tau),
        ("motor temperature", motor_temperature),
        ("motor voltage", motor_voltage),
    ):
        if not np.all(np.isfinite(values)):
            failures.append(f"{label} contains missing or non-finite values")

    max_temperature = float(np.nanmax(motor_temperature))
    if max_temperature > limits.max_motor_temperature_c:
        failures.append(
            f"motor temperature {max_temperature:.1f} C exceeds {limits.max_motor_temperature_c:.1f} C"
        )

    motor_errors = np.stack([sample.motor_errors for sample in samples])
    for motor_idx, joint_name in enumerate(DDS_JOINT_NAMES):
        error_values = sorted(set(int(value) for value in motor_errors[:, motor_idx] if value != 0))
        for error in error_values:
            finding = f"{joint_name} motorstate={error}"
            if motor_idx in intact_motor_indices:
                failures.append(finding)
            else:
                warnings.append(f"excluded broken finger: {finding}")

    mean_motor_voltage = np.nanmean(motor_voltage, axis=0)
    max_motor_temperature = np.nanmax(motor_temperature, axis=(0, 2))
    for motor_idx, joint_name in enumerate(DDS_JOINT_NAMES):
        if mean_motor_voltage[motor_idx] <= 0.0 and max_motor_temperature[motor_idx] <= 0.0:
            finding = f"{joint_name} has no powered telemetry (voltage=0, temperature=0)"
            if motor_idx in intact_motor_indices:
                failures.append(finding)
            else:
                warnings.append(f"excluded broken finger: {finding}")

    hand_errors = np.stack([sample.hand_errors for sample in samples])
    for error_idx in range(hand_errors.shape[1]):
        error_values = sorted(set(int(value) for value in hand_errors[:, error_idx] if value != 0))
        for error in error_values:
            failures.append(f"hand error[{error_idx}]={error}")

    pressure_delta_by_finger: dict[str, float] = {}
    pressure_lost = np.stack([sample.pressure_lost for sample in samples])
    for finger, indices in intact_fingers.items():
        selected_pressure = pressure[:, indices, :]
        selected_temperature = pressure_temperature[:, indices, :]
        if not np.all(np.isfinite(selected_pressure)):
            failures.append(f"{finger} pressure contains missing or non-finite values")
        if not np.all(np.isfinite(selected_temperature)):
            failures.append(f"{finger} pressure temperature contains missing or non-finite values")
        for sensor_idx in indices:
            lost_values = pressure_lost[:, sensor_idx]
            if np.any(lost_values != 0):
                failures.append(
                    f"{finger} pressure channel={sensor_idx} lost counter/status is nonzero "
                    f"(first={int(lost_values[0])}, last={int(lost_values[-1])}, "
                    f"delta={int(lost_values[-1] - lost_values[0])})"
                )

        if np.all(np.isfinite(selected_pressure)):
            delta = float(np.max(np.ptp(selected_pressure, axis=0)))
        else:
            delta = float("nan")
        pressure_delta_by_finger[finger] = delta
        if require_pressure_response and (
            not np.isfinite(delta) or delta < limits.min_pressure_delta
        ):
            failures.append(
                f"{finger} pressure change {delta:.4g} is below {limits.min_pressure_delta:.4g}"
            )

    if not require_pressure_response:
        warnings.append(
            "pressure arrays were checked for transport/lost/finite values only; "
            "use --pressure-test while gently pressing each intact fingertip to prove response"
        )

    rail_names = ("power_v", "system_v", "device_v")
    rail_values = np.asarray(
        [[getattr(sample, name) for name in rail_names] for sample in samples], dtype=np.float64
    )
    if not np.all(np.isfinite(rail_values)):
        failures.append("hand voltage rails contain missing or non-finite values")
    else:
        for rail_idx, rail_name in enumerate(rail_names):
            if np.all(rail_values[:, rail_idx] == 0.0):
                warnings.append(f"{rail_name} stayed at 0; firmware may not populate this field")

    finite_tau = np.abs(tau[np.isfinite(tau)])
    max_abs_tau = float(np.max(finite_tau)) if finite_tau.size else float("nan")
    if np.isfinite(max_abs_tau) and max_abs_tau > limits.max_abs_tau:
        failures.append(
            f"estimated torque {max_abs_tau:.3f} exceeds {limits.max_abs_tau:.3f}"
        )

    metrics = {
        "max_motor_temperature_c": max_temperature,
        "max_abs_tau": max_abs_tau,
        "q_min": np.nanmin(q, axis=0).tolist(),
        "q_max": np.nanmax(q, axis=0).tolist(),
        "mean_motor_voltage_v": np.nanmean(motor_voltage, axis=0).tolist(),
        "mean_power_v": float(np.nanmean([sample.power_v for sample in samples])),
        "mean_power_a": float(np.nanmean([sample.power_a for sample in samples])),
        "mean_system_v": float(np.nanmean([sample.system_v for sample in samples])),
        "mean_device_v": float(np.nanmean([sample.device_v for sample in samples])),
    }
    return DiagnosticReport(
        passed=not failures,
        sample_count=len(samples),
        state_rate_hz=rate_hz,
        failures=failures,
        warnings=warnings,
        pressure_delta_by_finger=pressure_delta_by_finger,
        metrics=metrics,
    )


def _motor_mode(motor_idx: int, *, enabled: bool, timeout: bool) -> int:
    status = 0x01 if enabled else 0x00
    return (motor_idx & 0x0F) | (status << 4) | ((0x01 if timeout else 0x00) << 7)


def build_isolated_hand_command(
    baseline_dds: Sequence[float],
    *,
    active_dds_joint: int | None,
    active_target: float | None,
    excluded_dds_joints: set[int],
    kp: float,
    kd: float,
    hand_cmd_factory: Callable[[], Any],
) -> Any:
    """Build a command that applies gains to one joint and zero gains everywhere else."""

    baseline = np.asarray(baseline_dds, dtype=np.float64).reshape(-1)
    if baseline.size != 7 or not np.all(np.isfinite(baseline)):
        raise ValueError("baseline_dds must contain seven finite joint positions")
    if active_dds_joint is not None:
        if active_dds_joint not in range(7):
            raise ValueError("active_dds_joint must be in [0, 6]")
        if active_dds_joint in excluded_dds_joints:
            raise ValueError("the active joint belongs to an excluded broken finger")
        if active_target is None or not np.isfinite(active_target):
            raise ValueError("active_target must be finite when a joint is active")

    cmd = hand_cmd_factory()
    if len(cmd.motor_cmd) < 7:
        raise ValueError(f"HandCmd_ has {len(cmd.motor_cmd)} motor commands; Dex3 requires 7")
    for motor_idx in range(7):
        active = motor_idx == active_dds_joint
        motor = cmd.motor_cmd[motor_idx]
        motor.mode = _motor_mode(motor_idx, enabled=active, timeout=active)
        motor.q = float(active_target if active else baseline[motor_idx])
        motor.dq = 0.0
        motor.kp = float(kp if active else 0.0)
        motor.kd = float(kd if active else 0.0)
        motor.tau = 0.0
    return cmd


def choose_small_motion_target(current: float, *, lower: float, upper: float, amplitude: float) -> float:
    """Choose the direction with more joint-limit clearance for a small diagnostic move."""

    if not all(np.isfinite(value) for value in (current, lower, upper, amplitude)):
        raise ValueError("current, limits, and amplitude must be finite")
    if lower >= upper or amplitude <= 0.0:
        raise ValueError("joint limits and amplitude are invalid")
    positive_room = upper - current
    negative_room = current - lower
    if positive_room >= amplitude and positive_room >= negative_room:
        return current + amplitude
    if negative_room >= amplitude:
        return current - amplitude
    if positive_room >= amplitude:
        return current + amplitude
    raise ValueError(
        f"joint at {current:.4f} has less than {amplitude:.4f} rad clearance to both limits"
    )


class Dex3DdsConnection:
    def __init__(
        self,
        *,
        network_interface: str,
        domain_id: int,
        state_topic: str,
        cmd_topic: str,
        write_timeout: float | None,
    ) -> None:
        self.network_interface = network_interface
        self.domain_id = int(domain_id)
        self.state_topic = state_topic
        self.cmd_topic = cmd_topic
        self.write_timeout = write_timeout
        self._samples: list[HandSample] = []
        self._lock = threading.Lock()
        self._subscriber: Any | None = None
        self._publisher: Any | None = None
        self._hand_cmd_factory: Callable[[], Any] | None = None
        self._sdk: dict[str, Any] | None = None

    @staticmethod
    def _load_sdk() -> dict[str, Any]:
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import (  # type: ignore
                unitree_hg_msg_dds__HandCmd_,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is unavailable in this Python environment; install the Unitree SDK2 package first"
            ) from exc
        return {
            "ChannelFactoryInitialize": ChannelFactoryInitialize,
            "ChannelPublisher": ChannelPublisher,
            "ChannelSubscriber": ChannelSubscriber,
            "HandCmd_": HandCmd_,
            "HandState_": HandState_,
            "hand_cmd_factory": unitree_hg_msg_dds__HandCmd_,
        }

    @staticmethod
    def _init_channel(channel: Any, *args: Any) -> None:
        if hasattr(channel, "Init"):
            channel.Init(*args)
        elif hasattr(channel, "InitChannel"):
            channel.InitChannel(*args)
        else:
            raise RuntimeError("Unitree DDS channel has neither Init nor InitChannel")

    def connect_read_only(self) -> None:
        self._sdk = self._load_sdk()
        self._sdk["ChannelFactoryInitialize"](self.domain_id, self.network_interface)
        self._subscriber = self._sdk["ChannelSubscriber"](self.state_topic, self._sdk["HandState_"])
        self._init_channel(self._subscriber, self._handle_state, 10)
        self._hand_cmd_factory = self._sdk["hand_cmd_factory"]

    def _handle_state(self, msg: Any) -> None:
        sample = extract_hand_sample(msg)
        with self._lock:
            self._samples.append(sample)
            if len(self._samples) > 20_000:
                del self._samples[:10_000]

    def samples_since(self, started_at_s: float) -> list[HandSample]:
        with self._lock:
            return [sample for sample in self._samples if sample.received_at_s >= started_at_s]

    def latest_samples(self, count: int) -> list[HandSample]:
        with self._lock:
            return list(self._samples[-count:])

    def wait_for_samples(self, count: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._samples) >= count:
                    return True
            time.sleep(0.01)
        with self._lock:
            return len(self._samples) >= count

    @property
    def hand_cmd_factory(self) -> Callable[[], Any]:
        if self._hand_cmd_factory is None:
            raise RuntimeError("DDS connection is not initialized")
        return self._hand_cmd_factory

    def enable_publisher(self) -> None:
        if self._sdk is None:
            raise RuntimeError("connect_read_only must be called before enabling commands")
        if self._publisher is None:
            self._publisher = self._sdk["ChannelPublisher"](self.cmd_topic, self._sdk["HandCmd_"])
            self._init_channel(self._publisher)

    def write(self, cmd: Any) -> None:
        if self._publisher is None:
            raise RuntimeError("command publisher is not enabled")
        if self.write_timeout is None:
            ok = self._publisher.Write(cmd)
        else:
            ok = self._publisher.Write(cmd, self.write_timeout)
        if ok is False:
            raise RuntimeError("Dex3 command publish returned false")

    def close(self) -> None:
        for channel in (self._publisher, self._subscriber):
            if channel is not None and hasattr(channel, "Close"):
                try:
                    channel.Close()
                except Exception:
                    pass


def _print_report(report: DiagnosticReport) -> None:
    verdict = "PASS" if report.passed else "FAIL"
    rate = "n/a" if report.state_rate_hz is None else f"{report.state_rate_hz:.1f} Hz"
    print(f"[g1-dex3-diagnostic] state/sensor verdict={verdict} samples={report.sample_count} rate={rate}")
    for finger, delta in report.pressure_delta_by_finger.items():
        print(f"[g1-dex3-diagnostic] pressure finger={finger} max_delta={delta:.6g}")
    for failure in report.failures:
        print(f"[g1-dex3-diagnostic] FAIL: {failure}")
    for warning in report.warnings:
        print(f"[g1-dex3-diagnostic] WARN: {warning}")


def _check_motion_safety(samples: Sequence[HandSample], limits: DiagnosticLimits) -> None:
    if not samples:
        raise RuntimeError("state feedback stopped during motion")
    latest = samples[-1]
    errors = [(DDS_JOINT_NAMES[i], int(value)) for i, value in enumerate(latest.motor_errors) if value != 0]
    if errors:
        raise RuntimeError(f"motor error during motion: {errors}")
    if np.any(latest.hand_errors != 0):
        raise RuntimeError(f"hand error during motion: {latest.hand_errors.tolist()}")
    max_temperature = float(np.nanmax(latest.motor_temperature_c))
    if not np.isfinite(max_temperature) or max_temperature > limits.max_motor_temperature_c:
        raise RuntimeError(f"unsafe motor temperature during motion: {max_temperature:.1f} C")
    max_tau = float(np.nanmax(np.abs(latest.tau_est)))
    if not np.isfinite(max_tau) or max_tau > limits.max_abs_tau:
        raise RuntimeError(
            f"estimated torque {max_tau:.3f} exceeds limit {limits.max_abs_tau:.3f}"
        )


def _publish_ramp(
    connection: Dex3DdsConnection,
    *,
    baseline: np.ndarray,
    joint_idx: int,
    start: float,
    target: float,
    excluded_joints: set[int],
    kp: float,
    kd: float,
    duration_s: float,
    command_rate_hz: float,
    limits: DiagnosticLimits,
) -> None:
    steps = max(2, int(np.ceil(duration_s * command_rate_hz)))
    period_s = 1.0 / command_rate_hz
    for value in np.linspace(start, target, steps):
        command = build_isolated_hand_command(
            baseline,
            active_dds_joint=joint_idx,
            active_target=float(value),
            excluded_dds_joints=excluded_joints,
            kp=kp,
            kd=kd,
            hand_cmd_factory=connection.hand_cmd_factory,
        )
        connection.write(command)
        time.sleep(period_s)
        _check_motion_safety(connection.latest_samples(2), limits)


def _median_q(connection: Dex3DdsConnection, count: int = 5) -> np.ndarray:
    samples = connection.latest_samples(count)
    if not samples:
        raise RuntimeError("no hand state available")
    q = np.stack([sample.q for sample in samples])
    if not np.all(np.isfinite(q)):
        raise RuntimeError("non-finite hand position feedback")
    return np.median(q, axis=0)


def run_isolated_motion_tests(
    connection: Dex3DdsConnection,
    *,
    side: str,
    excluded_fingers: set[str],
    test_fingers: set[str] | None,
    amplitude: float,
    kp: float,
    kd: float,
    duration_s: float,
    settle_s: float,
    command_rate_hz: float,
    limits: DiagnosticLimits,
) -> list[MotionResult]:
    if not excluded_fingers:
        raise ValueError("real motion testing requires at least one --exclude-finger for the broken finger")
    intact = _finger_indices(excluded_fingers)
    selected_fingers = set(intact) if test_fingers is None else set(test_fingers)
    if not selected_fingers or not selected_fingers.issubset(intact):
        raise ValueError("--test-finger must select only intact, non-excluded fingers")
    excluded_joints = {
        idx for finger in excluded_fingers for idx in FINGER_TO_DDS_INDICES[finger]
    }
    selected_joints = [
        idx for finger in FINGER_TO_DDS_INDICES if finger in selected_fingers for idx in intact[finger]
    ]

    baseline = _median_q(connection)
    limits_for_side = DDS_LIMITS_BY_SIDE[side]
    connection.enable_publisher()
    results: list[MotionResult] = []
    try:
        for joint_idx in selected_joints:
            joint_name = DDS_JOINT_NAMES[joint_idx]
            lower, upper = limits_for_side[joint_idx]
            start = float(_median_q(connection)[joint_idx])
            target = choose_small_motion_target(start, lower=lower, upper=upper, amplitude=amplitude)
            baseline[joint_idx] = start
            print(
                f"[g1-dex3-diagnostic] testing joint={joint_name} start={start:.4f} target={target:.4f}"
            )
            _publish_ramp(
                connection,
                baseline=baseline,
                joint_idx=joint_idx,
                start=start,
                target=target,
                excluded_joints=excluded_joints,
                kp=kp,
                kd=kd,
                duration_s=duration_s,
                command_rate_hz=command_rate_hz,
                limits=limits,
            )
            time.sleep(settle_s)
            _check_motion_safety(connection.latest_samples(2), limits)
            observed_target = float(_median_q(connection)[joint_idx])
            _publish_ramp(
                connection,
                baseline=baseline,
                joint_idx=joint_idx,
                start=target,
                target=start,
                excluded_joints=excluded_joints,
                kp=kp,
                kd=kd,
                duration_s=duration_s,
                command_rate_hz=command_rate_hz,
                limits=limits,
            )
            time.sleep(settle_s)
            observed_return = float(_median_q(connection)[joint_idx])

            commanded_delta = target - start
            observed_delta = observed_target - start
            following_error = abs(observed_target - target)
            return_error = abs(observed_return - start)
            failures: list[str] = []
            if np.sign(observed_delta) != np.sign(commanded_delta) or abs(observed_delta) < (
                limits.min_motion_fraction * abs(commanded_delta)
            ):
                failures.append("insufficient or wrong-direction joint response")
            if following_error > limits.max_following_error_rad:
                failures.append("following error exceeds limit")
            if return_error > limits.max_following_error_rad:
                failures.append("return-to-baseline error exceeds limit")
            result = MotionResult(
                joint=joint_name,
                commanded_delta_rad=commanded_delta,
                observed_delta_rad=observed_delta,
                following_error_rad=following_error,
                return_error_rad=return_error,
                passed=not failures,
                failures=tuple(failures),
            )
            results.append(result)
            print(
                f"[g1-dex3-diagnostic] joint={joint_name} verdict={'PASS' if result.passed else 'FAIL'} "
                f"commanded_delta={commanded_delta:.4f} observed_delta={observed_delta:.4f} "
                f"follow_error={following_error:.4f} return_error={return_error:.4f}"
            )
            if failures:
                break
    finally:
        disabled = build_isolated_hand_command(
            _median_q(connection),
            active_dds_joint=None,
            active_target=None,
            excluded_dds_joints=set(range(7)),
            kp=0.0,
            kd=0.0,
            hand_cmd_factory=connection.hand_cmd_factory,
        )
        for _ in range(3):
            connection.write(disabled)
            time.sleep(0.02)
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose one Unitree G1 Dex3 hand. Default mode is read-only; isolated motion "
            "requires explicit exclusion of the mechanically broken finger."
        )
    )
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--interface", default=DEFAULT_G1_NETWORK_INTERFACE)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-topic")
    parser.add_argument("--cmd-topic")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--write-timeout", type=float, default=2.0)
    parser.add_argument(
        "--exclude-finger",
        action="append",
        choices=tuple(FINGER_TO_DDS_INDICES),
        default=[],
        help="Broken finger to ignore and electrically de-energize; may be repeated.",
    )
    parser.add_argument(
        "--pressure-test",
        action="store_true",
        help="Require a pressure response; gently press every intact fingertip during the sampling window.",
    )
    parser.add_argument("--min-pressure-delta", type=float, default=0.05)
    parser.add_argument("--min-state-rate-hz", type=float, default=10.0)
    parser.add_argument("--max-motor-temp-c", type=float, default=70.0)
    parser.add_argument("--max-abs-tau", type=float, default=0.5)
    parser.add_argument(
        "--execute-motion",
        action="store_true",
        help="Publish small one-joint-at-a-time commands after the read-only checks pass.",
    )
    parser.add_argument(
        "--test-finger",
        action="append",
        choices=tuple(FINGER_TO_DDS_INDICES),
        help="Limit motion to selected intact fingers; default tests every non-excluded finger.",
    )
    parser.add_argument("--motion-amplitude-rad", type=float, default=0.08)
    parser.add_argument("--motion-duration-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument("--command-rate-hz", type=float, default=50.0)
    parser.add_argument("--kp", type=float, default=0.35)
    parser.add_argument("--kd", type=float, default=0.05)
    parser.add_argument("--max-following-error-rad", type=float, default=0.15)
    parser.add_argument("--min-motion-fraction", type=float, default=0.5)
    parser.add_argument(
        "--confirm",
        help="For real motion, must equal BROKEN_FINGER_SECURED_AND_AREA_CLEAR.",
    )
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--duration-s": args.duration_s,
        "--min-state-rate-hz": args.min_state_rate_hz,
        "--max-motor-temp-c": args.max_motor_temp_c,
        "--max-abs-tau": args.max_abs_tau,
        "--motion-amplitude-rad": args.motion_amplitude_rad,
        "--motion-duration-s": args.motion_duration_s,
        "--command-rate-hz": args.command_rate_hz,
        "--kp": args.kp,
        "--kd": args.kd,
        "--max-following-error-rad": args.max_following_error_rad,
        "--min-motion-fraction": args.min_motion_fraction,
    }
    invalid = [name for name, value in positive.items() if value <= 0.0]
    if invalid:
        raise ValueError(f"these options must be positive: {', '.join(invalid)}")
    if args.settle_s < 0.0 or args.min_pressure_delta < 0.0:
        raise ValueError("--settle-s and --min-pressure-delta must be non-negative")
    if args.motion_amplitude_rad > 0.15:
        raise ValueError("--motion-amplitude-rad is capped at 0.15 rad for this damage diagnostic")
    if args.execute_motion:
        if not args.exclude_finger:
            raise ValueError("--execute-motion requires --exclude-finger for the broken finger")
        if args.confirm != "BROKEN_FINGER_SECURED_AND_AREA_CLEAR":
            raise ValueError(
                "--execute-motion requires --confirm BROKEN_FINGER_SECURED_AND_AREA_CLEAR"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"[g1-dex3-diagnostic] configuration error: {exc}", file=sys.stderr)
        return 2

    limits = DiagnosticLimits(
        min_state_rate_hz=args.min_state_rate_hz,
        max_motor_temperature_c=args.max_motor_temp_c,
        max_abs_tau=args.max_abs_tau,
        min_pressure_delta=args.min_pressure_delta,
        max_following_error_rad=args.max_following_error_rad,
        min_motion_fraction=args.min_motion_fraction,
    )
    excluded_fingers = set(args.exclude_finger)
    state_topic = args.state_topic or DEFAULT_STATE_TOPICS[args.side]
    cmd_topic = args.cmd_topic or DEFAULT_CMD_TOPICS[args.side]
    connection = Dex3DdsConnection(
        network_interface=args.interface,
        domain_id=args.domain_id,
        state_topic=state_topic,
        cmd_topic=cmd_topic,
        write_timeout=args.write_timeout,
    )
    report: DiagnosticReport | None = None
    motion_results: list[MotionResult] = []
    try:
        print(
            f"[g1-dex3-diagnostic] READ-ONLY sampling side={args.side} topic={state_topic} "
            f"interface={args.interface} duration={args.duration_s:.1f}s"
        )
        if args.pressure_test:
            intact = ", ".join(_finger_indices(excluded_fingers))
            print(
                "[g1-dex3-diagnostic] During this window, gently press and release each intact "
                f"fingertip once: {intact}"
            )
        connection.connect_read_only()
        if not connection.wait_for_samples(2, timeout_s=3.0):
            raise RuntimeError(
                f"no Dex3 state received on {state_topic}; verify side/interface/topic "
                "(some firmware uses rt/dex3/<side>/state, while this project may use an rt/lf/... topic)"
            )
        started_at = time.monotonic()
        time.sleep(args.duration_s)
        samples = connection.samples_since(started_at)
        report = evaluate_state_samples(
            samples,
            limits=limits,
            require_pressure_response=args.pressure_test,
            excluded_fingers=excluded_fingers,
        )
        _print_report(report)
        if args.execute_motion:
            if not report.passed:
                raise RuntimeError("read-only checks failed; refusing to publish motion commands")
            print(
                "[g1-dex3-diagnostic] MOTION enabled: one joint at a time; "
                f"excluded fingers={sorted(excluded_fingers)}"
            )
            motion_results = run_isolated_motion_tests(
                connection,
                side=args.side,
                excluded_fingers=excluded_fingers,
                test_fingers=None if args.test_finger is None else set(args.test_finger),
                amplitude=args.motion_amplitude_rad,
                kp=args.kp,
                kd=args.kd,
                duration_s=args.motion_duration_s,
                settle_s=args.settle_s,
                command_rate_hz=args.command_rate_hz,
                limits=limits,
            )
    except (RuntimeError, ValueError) as exc:
        print(f"[g1-dex3-diagnostic] ABORT: {exc}", file=sys.stderr)
        return_code = 2
    else:
        return_code = 0 if report is not None and report.passed and all(r.passed for r in motion_results) else 1
    finally:
        connection.close()

    if args.report_json is not None:
        payload = {
            "state_sensor_report": None if report is None else report.to_dict(),
            "motion_results": [asdict(result) for result in motion_results],
            "return_code": return_code,
        }
        args.report_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"[g1-dex3-diagnostic] report={args.report_json}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
