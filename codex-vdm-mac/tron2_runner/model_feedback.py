"""Summarize motion feedback for the model without simulator contact details."""

import json
from math import dist


_STREAM_FIELDS = (
    "dispatch_completed", "joint_goal_reached", "completed", "final_joint_error_rad", "error",
    "notification_count",
)
_ABORT_FIELDS = ("stage", "current_tcp_base_m", "target_tcp_base_m")
_STOP_EXCEPTION = "Tron2ServoJContactStop"
_STOP_MESSAGE = "left-arm motion interrupted by table contact"


def compact_servoj_feedback(log: str) -> str:
    """Build the model's copy of stdout or stderr from the raw log."""
    lines = []
    for line in log.splitlines(keepends=True):
        prefix, _, message = line.partition("] ")
        if prefix not in ("[TRON2", "[TRON2 RGB-D"):
            # Keep unrelated tracebacks and neutralize this specific exception.
            if line.lstrip().startswith(f"raise {_STOP_EXCEPTION}("):
                continue
            exception = line.replace(f"capx.integrations.tron2.left_motion.{_STOP_EXCEPTION}:", "RuntimeError:")
            lines.append(exception.replace(_STOP_MESSAGE, "left-arm motion aborted"))
            continue
        record = message.rstrip("\r\n")
        newline = message[len(record):]
        label = record.partition(" ")[0]
        if label == "TABLE_CONTACT":
            continue
        header, separator, payload = record.partition(" {")
        is_stop = label == "STOPPED_ON_CONTACT"
        is_stream = label == "SERVOJ_STREAM"
        if not is_stream and not is_stop:
            lines.append(line)
            continue
        try:
            raw = json.loads("{" + payload) if separator else {}
        except json.JSONDecodeError:
            raw = {}
        if is_stop:
            summary = {key: raw[key] for key in _ABORT_FIELDS if key in raw}
            actual = summary.get("current_tcp_base_m")
            target = summary.get("target_tcp_base_m")
            if actual is not None and target is not None:
                summary["position_error_mm"] = round(1000 * dist(actual, target), 2)
            header = "MOTION_ABORTED"
        elif raw.get("contact_stop"):
            # Stopped streams have a separate record with the resulting TCP.
            continue
        else:
            summary = {key: raw[key] for key in _STREAM_FIELDS if key in raw and raw[key] is not None}
            notifications = raw.get("notifications")
            if isinstance(notifications, list) and notifications:
                summary["notification_count"] = len(notifications)
            if not summary:
                continue
        payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        lines.append(f"{prefix}] {header} {payload}{newline}")
    return "".join(lines)
