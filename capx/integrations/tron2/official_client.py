"""Official LimX TRON2 WebSocket protocol client for CaP-X.

Talks to Tron2_controller's signaling_node over ws://<host>:5000 (same protocol
on local 4090w simulation, SSH-tunneled Mac dev, and real TRON2 hardware).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import websocket

# Official 16-axis wire order (NOTE: head_pitch is 14, head_yaw is 15)
PROTOCOL_JOINT_NAMES: tuple[str, ...] = (
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "elbow_L_Joint",
    "wrist_yaw_L_Joint",
    "wrist_pitch_L_Joint",
    "wrist_roll_L_Joint",
    "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
    "elbow_R_Joint",
    "wrist_yaw_R_Joint",
    "wrist_pitch_R_Joint",
    "wrist_roll_R_Joint",
    "head_pitch_Joint",
    "head_yaw_Joint",
)

# Safe raised ready posture (L7 + R7 + head2) from official SDK
DEFAULT_HOME_Q16: list[float] = [
    0.0, 0.25, 0.0, -1.57, 0.236, 0.0, 0.0,
    0.0, -0.25, 0.0, -1.57, -0.236, 0.0, 0.0,
    0.0, 0.0,
]

# Horizontal forward-pointing gripper orientation (w, x, y, z) in base_Link frame:
# fingers point along +X_base, jaws open/close horizontally along +/-Y_base.
HORIZONTAL_GRASP_QUAT_WXYZ: list[float] = [0.70710678, 0.0, -0.70710678, 0.0]


@dataclass
class Tron2MovePose:
    left_pos: np.ndarray   # (3,) in base_Link frame (meters)
    left_quat: np.ndarray  # (4,) wxyz
    right_pos: np.ndarray  # (3,) in base_Link frame (meters)
    right_quat: np.ndarray # (4,) wxyz


class Tron2OfficialClient:
    """Thread-safe WebSocket client for LimX Tron2_controller (ws://host:5000)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        accid: str = "DACH_TRON2A_001",
        timeout_sec: float = 5.0,
    ) -> None:
        self.url = f"ws://{host}:{port}"
        self.accid = accid
        self.timeout_sec = timeout_sec

        self._ws: websocket.WebSocket | None = None
        self._recv_thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._responses: dict[str, dict[str, Any]] = {}

    def connect(self) -> None:
        self._ws = websocket.create_connection(self.url, timeout=self.timeout_sec)
        self._closed.clear()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._recv_thread is not None and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)

    def _recv_loop(self) -> None:
        while not self._closed.is_set() and self._ws is not None:
            try:
                raw = self._ws.recv()
            except Exception:
                break
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            with self._cond:
                title = str(msg.get("title", ""))
                guid = msg.get("guid")
                if title.startswith("response_") and guid:
                    self._responses[guid] = msg
                    self._cond.notify_all()

    def request(self, title: str, data: dict[str, Any] | None = None, timeout_sec: float | None = None) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Tron2OfficialClient is not connected.")
        if not title.startswith("request_"):
            raise ValueError(f"Expected title starting with 'request_', got: {title}")

        guid = str(uuid.uuid4())
        expected_title = "response_" + title[len("request_"):]
        payload = {
            "accid": self.accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": guid,
            "data": data or {},
        }
        with self._send_lock:
            self._ws.send(json.dumps(payload, ensure_ascii=False))

        deadline = time.perf_counter() + (timeout_sec or self.timeout_sec)
        with self._cond:
            try:
                while time.perf_counter() < deadline:
                    resp = self._responses.get(guid)
                    if resp is not None and resp.get("title") == expected_title:
                        return self._responses.pop(guid)
                    rem = max(0.0, deadline - time.perf_counter())
                    self._cond.wait(timeout=min(0.05, rem))
            finally:
                self._responses.pop(guid, None)
        raise TimeoutError(f"Timed out waiting for {expected_title} (guid={guid})")

    def get_joint_state(self) -> np.ndarray:
        """Return current 16-axis joint angles (rad) in PROTOCOL_JOINT_NAMES order."""
        resp = self.request("request_get_joint_state", {})
        data = resp.get("data", {})
        q = data.get("q", [])
        if len(q) != 16:
            raise RuntimeError(f"Expected 16 joint values, got {len(q)}: {resp}")
        return np.asarray(q, dtype=np.float64)

    def get_move_pose(self) -> Tron2MovePose:
        """Return current dual-arm TCP poses in base_Link frame."""
        resp = self.request("request_get_move_pose", {})
        data = resp.get("data", {})
        return Tron2MovePose(
            left_pos=np.asarray(data["left_position"], dtype=np.float64),
            left_quat=np.asarray(data["left_quat"], dtype=np.float64),
            right_pos=np.asarray(data["right_position"], dtype=np.float64),
            right_quat=np.asarray(data["right_quat"], dtype=np.float64),
        )

    def move_j(self, arm_q14: list[float] | np.ndarray, duration_sec: float = 2.5) -> dict[str, Any]:
        """Send official MoveJ (14 arm joints in rad) and return server response data."""
        joint = [float(x) for x in arm_q14]
        if len(joint) != 14:
            raise ValueError(f"MoveJ expects 14 arm joints, got {len(joint)}")
        resp = self.request("request_movej", {"time": float(duration_sec), "joint": joint})
        data = resp.get("data", {})
        if data.get("result") != "success":
            raise RuntimeError(f"MoveJ rejected by controller: {data}")
        return data

    def move_p(
        self,
        left_pos: list[float] | np.ndarray,
        left_quat_wxyz: list[float] | np.ndarray,
        right_pos: list[float] | np.ndarray,
        right_quat_wxyz: list[float] | np.ndarray,
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        """Send official MoveP (dual-arm TCP poses in base_Link frame)."""
        pos14 = (
            [float(x) for x in left_pos]
            + [float(x) for x in left_quat_wxyz]
            + [float(x) for x in right_pos]
            + [float(x) for x in right_quat_wxyz]
        )
        if len(pos14) != 14:
            raise ValueError(f"MoveP expects 14 pose values, got {len(pos14)}")
        resp = self.request("request_movep", {"time": float(duration_sec), "pos": pos14})
        data = resp.get("data", {})
        if data.get("result") != "success":
            raise RuntimeError(f"MoveP rejected by controller: {data}")
        return data
