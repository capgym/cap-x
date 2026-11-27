"""Stream a planned TRON2 joint path through the official ServoJ interface."""

from __future__ import annotations

import threading
import time

import numpy as np

from .servoj_trajectory import resample_servoj


_RATE_HZ = 300
_GOAL_TOLERANCE_RAD = 0.015
_HOLD_VERIFY_TIMEOUT_SEC = 0.5


def move_left_servoj_path(env, path: np.ndarray, duration_sec: float) -> dict:
    """Stream the time-scaled left-arm path while the caller records the scene."""
    left = resample_servoj(path, duration_sec, rate_hz=_RATE_HZ)
    env._ensure_connected()
    client = env.client
    other = env._read_arm_joints()[7:].tolist()
    commands = [point.tolist() + other for point in left]
    stop = threading.Event()
    sent_at = []
    sent_final_target = False
    settle_wall_s = 0.0
    result = {
        "rate_hz": _RATE_HZ,
        "planned_samples": len(commands),
        "ik_waypoints": len(path),
        "scheduled_duration_s": (len(commands) - 1) / _RATE_HZ,
        "late_samples": 0,
        "dropped_samples": 0,
        "notifications": client.drain_servoj_notifications(),
        "error": None,
        "contact_stop": None,
    }

    def stream(guard) -> None:
        nonlocal sent_final_target
        start = time.monotonic()
        i = 0
        while i < len(commands) and not stop.is_set():
            due = start + i / _RATE_HZ
            if guard.detected.wait(max(0.0, due - time.monotonic())) or stop.is_set():
                return
            result["notifications"].extend(client.drain_servoj_notifications())
            if result["notifications"]:
                return
            if not client.servoj_receive_alive():
                raise ConnectionError("ServoJ feedback receiver stopped")
            now = time.monotonic()
            # Skip outdated targets when the sender briefly falls behind.
            latest = min(len(commands) - 1, max(i, int((now - start) * _RATE_HZ)))
            result["dropped_samples"] += latest - i
            result["late_samples"] += now - start - latest / _RATE_HZ > 0.001
            if guard.detected.is_set():
                return
            client.send_servoj(commands[latest], filter_ratio=1.0)
            sent_at.append(now)
            sent_final_target = latest == len(commands) - 1
            i = latest + 1

    def settle(guard) -> None:
        nonlocal settle_wall_s
        start = time.monotonic()
        minimum = start + 0.35
        deadline = minimum + 3.0
        # Keep watching for contact even after the last trajectory frame.
        while not (stop.is_set() or guard.detected.is_set()):
            result["notifications"].extend(client.drain_servoj_notifications())
            now = time.monotonic()
            next_deadline = minimum
            if now >= minimum:
                next_deadline = deadline
                error_rad = float(np.max(np.abs(env._read_arm_joints()[:7] - left[-1])))
                if (error_rad <= _GOAL_TOLERANCE_RAD or now >= deadline
                        or result["error"] is not None or result["notifications"]):
                    break
                if not client.servoj_receive_alive():
                    result["error"] = "ServoJ feedback receiver stopped"
                    break
            guard.detected.wait(min(0.025, max(0.0, next_deadline - now)))
        settle_wall_s = time.monotonic() - start

    def hold_after_contact(guard, detected_during: str) -> None:
        # This worker is still the sole writer for the action. An original
        # trajectory frame already in flight can finish before the hold.
        target = np.asarray(guard.trigger["q16_rad"])
        contact = {
            "detected_during": detected_during,
            "trigger": guard.trigger,
            "hold_sent": False,
            "hold_send_latency_ms": None,
            "motor_target_matched": False,
            "hold_notifications": [],
            "hold_transport_error": None,
        }
        result["contact_stop"] = contact
        result["notifications"].extend(client.drain_servoj_notifications())
        try:
            if not client.servoj_receive_alive():
                raise ConnectionError("ServoJ feedback receiver stopped before the contact hold")
            sent_after_s = time.monotonic()
            client.send_servoj(target, filter_ratio=1.0)
            contact["hold_sent"] = True
            contact["hold_send_latency_ms"] = 1e3 * (sent_after_s - guard.trigger["wall_monotonic_s"])
            deadline = sent_after_s + _HOLD_VERIFY_TIMEOUT_SEC
            while True:
                contact["hold_notifications"].extend(client.drain_servoj_notifications())
                state = env.servoj_contact_state(guard)
                contact["motor_target_matched"] = env.servoj_hold_target_received(target, sent_after_s)
                if not client.servoj_receive_alive():
                    raise ConnectionError("ServoJ feedback receiver stopped during the contact hold")
                if contact["motor_target_matched"] and state.get("unloaded") and state.get("stationary"):
                    break
                if time.monotonic() >= deadline or stop.wait(1 / _RATE_HZ):
                    break
        except Exception as exc:
            contact["hold_transport_error"] = f"{type(exc).__name__}: {exc}"
        contact["after"] = env.servoj_contact_state(guard)

    def run_action(guard) -> None:
        try:
            stream(guard)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        detected_during = "settle" if sent_final_target else "stream"
        try:
            if not guard.detected.is_set():
                detected_during = "settle"
                settle(guard)
            if not env.finish_servoj_without_contact(guard):
                hold_after_contact(guard, detected_during)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

    with env.servoj_table_contact_guard() as guard:
        worker = threading.Thread(target=run_action, args=(guard,), name="tron2-servoj", daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                env._wait_and_record(0.025)
        finally:
            stop.set()
            worker.join()

    if not client.servoj_receive_alive() and result["error"] is None:
        result["error"] = "ServoJ feedback receiver stopped"
    error_rad = float(np.max(np.abs(env._read_arm_joints()[:7] - left[-1])))
    gaps = np.diff(sent_at)
    elapsed = float(sum(gaps))
    dispatch_completed = sent_final_target and not result["notifications"] and result["error"] is None
    joint_goal_reached = error_rad <= _GOAL_TOLERANCE_RAD
    result.update(
        sent_samples=len(sent_at),
        actual_hz=float(len(gaps) / elapsed) if elapsed > 0 else None,
        stream_wall_s=elapsed,
        settle_wall_s=settle_wall_s,
        final_joint_error_rad=error_rad,
        max_send_gap_ms=1000 * float(np.max(gaps)) if len(gaps) else None,
        dispatch_completed=dispatch_completed,
        joint_goal_reached=joint_goal_reached,
        completed=dispatch_completed and joint_goal_reached and result["contact_stop"] is None,
    )
    return result
