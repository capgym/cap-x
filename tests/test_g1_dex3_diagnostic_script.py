from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _hand_state(
    *,
    q: np.ndarray | None = None,
    motor_errors: list[int] | None = None,
    pressure: np.ndarray | None = None,
    pressure_lost: list[int] | None = None,
):
    q_values = np.zeros(7) if q is None else np.asarray(q, dtype=np.float64)
    error_values = [0] * 7 if motor_errors is None else motor_errors
    pressure_values = np.zeros((7, 12)) if pressure is None else np.asarray(pressure, dtype=np.float64)
    lost_values = [0] * 7 if pressure_lost is None else pressure_lost
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(
                q=float(q_values[i]),
                dq=0.0,
                tau_est=0.0,
                temperature=[30, 31],
                vol=24.0,
                motorstate=error_values[i],
            )
            for i in range(7)
        ],
        press_sensor_state=[
            SimpleNamespace(
                pressure=pressure_values[i].tolist(),
                temperature=[25.0] * 12,
                lost=lost_values[i],
            )
            for i in range(7)
        ],
        power_v=24.0,
        power_a=0.2,
        system_v=12.0,
        device_v=5.0,
        error=[0, 0],
    )


def _fake_hand_cmd():
    return SimpleNamespace(
        motor_cmd=[
            SimpleNamespace(mode=0, q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0)
            for _ in range(7)
        ]
    )


def test_state_evaluation_fails_on_motor_error_and_lost_pressure_sensor() -> None:
    from tools.g1_dex3_diagnostic import DiagnosticLimits, evaluate_state_samples, extract_hand_sample

    samples = [
        extract_hand_sample(
            _hand_state(motor_errors=[0, 0, 0, 0, 23, 0, 0], pressure_lost=[0, 0, 1, 0, 0, 0, 0]),
            received_at_s=float(i) * 0.01,
        )
        for i in range(5)
    ]

    report = evaluate_state_samples(samples, limits=DiagnosticLimits(min_state_rate_hz=50.0))

    assert report.passed is False
    assert any("motorstate=23" in finding for finding in report.failures)
    assert any("lost counter/status is nonzero" in finding for finding in report.failures)


def test_diagnostic_defaults_use_project_state_topics() -> None:
    from tools.g1_dex3_diagnostic import DEFAULT_STATE_TOPICS

    assert DEFAULT_STATE_TOPICS == {
        "left": "rt/dex3/left/state",
        "right": "rt/lf/dex3/right/state",
    }


def test_state_evaluation_fails_on_excessive_estimated_torque() -> None:
    from tools.g1_dex3_diagnostic import DiagnosticLimits, evaluate_state_samples, extract_hand_sample

    samples = [
        extract_hand_sample(_hand_state(), received_at_s=float(i) * 0.01)
        for i in range(5)
    ]
    for sample in samples:
        sample.tau_est[0] = 0.8

    report = evaluate_state_samples(
        samples,
        limits=DiagnosticLimits(min_state_rate_hz=50.0, max_abs_tau=0.5),
    )

    assert report.passed is False
    assert any("estimated torque 0.800 exceeds 0.500" in finding for finding in report.failures)


def test_pressure_response_requires_each_intact_finger_but_skips_broken_finger() -> None:
    from tools.g1_dex3_diagnostic import DiagnosticLimits, evaluate_state_samples, extract_hand_sample

    pressure_start = np.zeros((7, 12), dtype=np.float64)
    pressure_end = pressure_start.copy()
    pressure_end[0, 0] = 2.0  # thumb
    pressure_end[3, 0] = 3.0  # middle, in DDS order
    # Index channels 5 and 6 intentionally do not respond because it is excluded.
    samples = [
        extract_hand_sample(_hand_state(pressure=pressure), received_at_s=float(i) * 0.01)
        for i, pressure in enumerate([pressure_start, pressure_end])
    ]

    report = evaluate_state_samples(
        samples,
        limits=DiagnosticLimits(min_state_rate_hz=50.0, min_pressure_delta=1.0),
        require_pressure_response=True,
        excluded_fingers={"index"},
    )

    assert report.passed is True
    assert report.pressure_delta_by_finger["thumb"] == 2.0
    assert report.pressure_delta_by_finger["middle"] == 3.0
    assert "index" not in report.pressure_delta_by_finger


def test_isolated_command_only_energizes_active_joint_and_disables_broken_finger() -> None:
    from tools.g1_dex3_diagnostic import build_isolated_hand_command

    baseline = np.linspace(0.0, 0.6, 7)
    cmd = build_isolated_hand_command(
        baseline,
        active_dds_joint=3,
        active_target=0.55,
        excluded_dds_joints={5, 6},
        kp=0.35,
        kd=0.05,
        hand_cmd_factory=_fake_hand_cmd,
    )

    assert np.allclose([motor.q for motor in cmd.motor_cmd], [0.0, 0.1, 0.2, 0.55, 0.4, 0.5, 0.6])
    assert [motor.kp for motor in cmd.motor_cmd] == [0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
    assert [motor.kd for motor in cmd.motor_cmd] == [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
    assert [motor.tau for motor in cmd.motor_cmd] == [0.0] * 7
    assert cmd.motor_cmd[5].mode == 5
    assert cmd.motor_cmd[6].mode == 6
    assert cmd.motor_cmd[3].mode == 0x93


def test_small_motion_target_stays_inside_joint_limits() -> None:
    from tools.g1_dex3_diagnostic import choose_small_motion_target

    assert choose_small_motion_target(1.54, lower=0.0, upper=1.57, amplitude=0.08) == 1.46
    assert choose_small_motion_target(0.03, lower=0.0, upper=1.57, amplitude=0.08) == 0.11
