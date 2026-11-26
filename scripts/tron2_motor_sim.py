"""Run a standalone TRON2 motor simulator in an isolated simulation network."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "capx/assets/tron2/DACH_TRON2A/xml/robot_grasper.xml"


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rate", type=float, default=300.0, help="Target motor state rate in Hz.")
    parser.add_argument("--cmd-timeout", type=float, default=0.0, help="Seconds. Zero holds the last command.")
    parser.add_argument("--duration", type=float, help="Seconds to run after transport starts.")
    parser.add_argument("--log-interval", type=float, default=2.0, help="Seconds between timing reports.")
    parser.add_argument("--check-model", action="store_true", help="Check the model without opening the SDK.")
    args = parser.parse_args()
    for option, value in (("--rate", args.rate), ("--duration", args.duration), ("--log-interval", args.log_interval)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{option} must be finite and positive")
    if not math.isfinite(args.cmd_timeout) or args.cmd_timeout < 0:
        parser.error("--cmd-timeout must be finite and nonnegative")

    import mujoco

    # A robot-only trial uses the small probe venv without the task's perception stack.
    sys.path.insert(0, str(ROOT / "capx/integrations"))
    from tron2.motor_sim import Tron2MotorSim
    from tron2.official_client import PROTOCOL_JOINT_NAMES

    model_path = args.model.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    motor = Tron2MotorSim(model, data, state_rate_hz=args.rate, command_timeout_sec=args.cmd_timeout)
    joint_qpos = [int(model.joint(name).qposadr[0]) for name in PROTOCOL_JOINT_NAMES]
    dt = model.opt.timestep
    if args.check_model:
        emit("model_ok", model=str(model_path), mujoco_version=mujoco.__version__,
             timestep_sec=dt, joint_names=PROTOCOL_JOINT_NAMES)
        return

    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        motor.start()
        emit(
            "ready",
            model=str(model_path),
            mujoco_version=mujoco.__version__,
            timestep_sec=dt,
            state_rate_hz=args.rate,
            command_timeout_sec=args.cmd_timeout,
            joint_names=PROTOCOL_JOINT_NAMES,
        )
        start = next_step = last_report = time.monotonic()
        steps = report_steps = schedule_resets = 0
        report_commands = motor.command_count
        report_states = motor.state_count
        max_lateness = 0.0
        while not stopped:
            now = time.monotonic()
            if args.duration is not None and now - start >= args.duration:
                break
            if now < next_step:
                time.sleep(next_step - now)
                now = time.monotonic()
            lateness = now - next_step
            max_lateness = max(max_lateness, lateness)
            if lateness > 0.05:
                # Bound the catch-up burst after a long scheduling pause.
                next_step = now
                schedule_resets += 1

            motor.apply_control()
            mujoco.mj_step(model, data)
            motor.publish_state_if_due()
            steps += 1
            next_step += dt
            elapsed = now - last_report
            if elapsed >= args.log_interval:
                commands, states = motor.command_count, motor.state_count
                age = motor.last_command_age_sec
                emit(
                    "timing",
                    wall_sec=round(now - start, 3),
                    sim_sec=round(data.time, 3),
                    physics_hz=round((steps - report_steps) / elapsed, 1),
                    command_hz=round((commands - report_commands) / elapsed, 1),
                    state_hz=round((states - report_states) / elapsed, 1),
                    invalid_commands=motor.invalid_command_count,
                    command_age_ms=round(age * 1000, 3) if age is not None and math.isfinite(age) else None,
                    max_lateness_ms=round(max_lateness * 1000, 3),
                    schedule_resets=schedule_resets,
                    q_rad=data.qpos[joint_qpos].tolist(),
                )
                last_report, report_steps = now, steps
                report_commands, report_states = commands, states
                max_lateness = 0.0

        emit(
            "stopped",
            steps=steps,
            elapsed_sec=round(time.monotonic() - start, 3),
            commands=motor.command_count,
            invalid_commands=motor.invalid_command_count,
            states=motor.state_count,
            q_rad=data.qpos[joint_qpos].tolist(),
        )
    finally:
        try:
            motor.close()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
