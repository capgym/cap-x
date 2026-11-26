"""Stream a planned TRON2 joint path through the official ServoJ interface."""

from __future__ import annotations

import threading
import time

import numpy as np

from .servoj_trajectory import resample_servoj


_RATE_HZ = 300


def move_left_servoj_path(env, path: np.ndarray, duration_sec: float) -> dict:
    """Stream the time-scaled left-arm path while the caller records the scene."""
    left = resample_servoj(path, duration_sec, rate_hz=_RATE_HZ)
    env._ensure_connected()
    client = env.client
    other = env._read_arm_joints()[7:].tolist()
    commands = [point.tolist() + other for point in left]
    stop = threading.Event()
    sent_at = []
    result = {
        "rate_hz": _RATE_HZ,
        "planned_samples": len(commands),
        "ik_waypoints": len(path),
        "scheduled_duration_s": (len(commands) - 1) / _RATE_HZ,
        "late_samples": 0,
        "dropped_samples": 0,
        "notifications": client.drain_servoj_notifications(),
        "error": None,
    }

    def stream() -> None:
        try:
            start = time.monotonic()
            i = 0
            while i < len(commands):
                due = start + i / _RATE_HZ
                if stop.wait(max(0.0, due - time.monotonic())):
                    return
                result["notifications"].extend(client.drain_servoj_notifications())
                if result["notifications"]:
                    return
                if not client.servoj_receive_alive():
                    raise ConnectionError("ServoJ feedback receiver stopped")
                now = time.monotonic()
                # Preserve the time line if the sender is briefly descheduled.
                # A burst of outdated targets would not reproduce the planned motion.
                latest = min(len(commands) - 1, max(i, int((now - start) * _RATE_HZ)))
                result["dropped_samples"] += latest - i
                result["late_samples"] += now - start - latest / _RATE_HZ > 0.001
                client.send_servoj(commands[latest], filter_ratio=1.0)
                sent_at.append(now)
                i = latest + 1
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=stream, name="tron2-servoj", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            env._wait_and_record(0.025)
    finally:
        stop.set()
        worker.join()

    # Keep the same minimum settle interval and joint tolerance as MoveJ.
    settle_start = time.monotonic()
    env._wait_and_record(0.35)
    result["notifications"].extend(client.drain_servoj_notifications())
    deadline = time.monotonic() + 3.0
    error_rad = float(np.max(np.abs(env._read_arm_joints()[:7] - left[-1])))
    while (error_rad > 0.015 and time.monotonic() < deadline
           and result["error"] is None and not result["notifications"]):
        env._wait_and_record(0.025)
        result["notifications"].extend(client.drain_servoj_notifications())
        error_rad = float(np.max(np.abs(env._read_arm_joints()[:7] - left[-1])))
    if not client.servoj_receive_alive() and result["error"] is None:
        result["error"] = "ServoJ feedback receiver stopped"
    gaps = np.diff(sent_at)
    elapsed = float(sum(gaps))
    result.update(
        sent_samples=len(sent_at),
        actual_hz=float(len(gaps) / elapsed) if elapsed > 0 else None,
        stream_wall_s=elapsed,
        settle_wall_s=time.monotonic() - settle_start,
        final_joint_error_rad=error_rad,
        max_send_gap_ms=1000 * float(np.max(gaps)) if len(gaps) else None,
        completed=bool(sent_at and not result["notifications"] and result["error"] is None),
    )
    return result
