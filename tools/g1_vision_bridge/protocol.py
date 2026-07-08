from __future__ import annotations

import socket
import struct
import time
from typing import Any

import msgpack
import msgpack_numpy as msgpack_numpy
import numpy as np

msgpack_numpy.patch()

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _as_rgb(rgb: Any) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"rgb must have shape HxWx3, got {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_depth_m(depth_m: Any) -> np.ndarray:
    arr = np.asarray(depth_m, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"depth_m must have shape HxW or HxWx1, got {arr.shape}.")
    return np.ascontiguousarray(arr)


def _as_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return np.ascontiguousarray(arr)


def quaternion_wxyz_from_matrix(rotation: Any) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized WXYZ quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diag = np.diag(matrix)
        axis = int(np.argmax(diag))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Rotation matrix produced a zero-norm quaternion.")
    return (quat / norm).astype(np.float32)


def pose_xyz_wxyz_from_pose_mat(pose_mat: Any) -> np.ndarray:
    pose = _as_matrix(pose_mat, (4, 4), "pose_mat")
    quat_wxyz = quaternion_wxyz_from_matrix(pose[:3, :3])
    return np.concatenate([pose[:3, 3].astype(np.float32), quat_wxyz]).astype(np.float32)


def build_frame(
    *,
    camera_name: str,
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    pose_mat: Any,
    timestamp: float | None = None,
    frame_id: int | None = None,
) -> dict[str, Any]:
    """Build a validated RGB-D frame packet for transport."""

    rgb_arr = _as_rgb(rgb)
    depth_arr = _as_depth_m(depth_m)
    if depth_arr.shape != rgb_arr.shape[:2]:
        raise ValueError(
            f"depth_m shape {depth_arr.shape} must match rgb image height/width {rgb_arr.shape[:2]}."
        )

    pose_mat_arr = _as_matrix(pose_mat, (4, 4), "pose_mat")
    frame: dict[str, Any] = {
        "type": "g1_vision_frame",
        "version": PROTOCOL_VERSION,
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "camera_name": str(camera_name),
        "rgb": rgb_arr,
        "depth_m": depth_arr,
        "intrinsics": _as_matrix(intrinsics, (3, 3), "intrinsics"),
        "pose_mat": pose_mat_arr,
        "pose": pose_xyz_wxyz_from_pose_mat(pose_mat_arr),
    }
    if frame_id is not None:
        frame["frame_id"] = int(frame_id)
    return frame


def frame_to_capx_observation(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert a bridge frame to the observation shape consumed by CaP-X visual APIs."""

    if frame.get("type") != "g1_vision_frame":
        raise ValueError(f"Unsupported frame type: {frame.get('type')!r}.")
    if int(frame.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported frame version: {frame.get('version')!r}.")

    camera_name = str(frame["camera_name"])
    depth = _as_depth_m(frame["depth_m"])[:, :, None]
    pose_mat = _as_matrix(frame["pose_mat"], (4, 4), "pose_mat")
    pose = np.asarray(frame.get("pose", pose_xyz_wxyz_from_pose_mat(pose_mat)), dtype=np.float32)
    if pose.shape != (7,):
        raise ValueError(f"pose must have shape (7,), got {pose.shape}.")

    return {
        camera_name: {
            "images": {
                "rgb": _as_rgb(frame["rgb"]),
                "depth": np.ascontiguousarray(depth.astype(np.float32)),
            },
            "intrinsics": _as_matrix(frame["intrinsics"], (3, 3), "intrinsics"),
            "pose": pose,
            "pose_mat": pose_mat,
        },
        "timestamp": float(frame["timestamp"]),
    }


def frame_to_g1_lowlevel_message(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert a frame to the normalized input schema expected by ``G1RealLowLevel``.

    ``G1RealLowLevel`` already normalizes the ``camera_top`` message shape after
    msgpack decoding, including byte keys produced by its legacy decoder.
    """

    obs = frame_to_capx_observation(frame)
    camera_obs = obs[str(frame["camera_name"])]
    return {
        "camera_top": {
            "images": {
                "rgb": camera_obs["images"]["rgb"],
                "depth": camera_obs["images"]["depth"],
            },
            "intrinsics_matrix": camera_obs["intrinsics"],
            "pose": camera_obs["pose"],
            "pose_mat": camera_obs["pose_mat"],
        },
        "timestamp": obs["timestamp"],
    }


def encode_message(message: dict[str, Any]) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


def decode_message(payload: bytes) -> dict[str, Any]:
    decoded = msgpack.unpackb(payload, raw=False)
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected decoded message to be a dict, got {type(decoded)!r}.")
    return decoded


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Socket closed while reading framed message.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_framed(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = encode_message(message)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Message is too large: {len(payload)} bytes.")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_framed(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, 4)
    (payload_size,) = struct.unpack("!I", header)
    if payload_size <= 0 or payload_size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid framed payload size: {payload_size}.")
    return decode_message(recv_exact(sock, payload_size))


def connect_with_retry(host: str, port: int, *, retry_s: float = 1.0) -> socket.socket:
    while True:
        try:
            return socket.create_connection((host, port), timeout=5.0)
        except OSError as exc:
            print(f"Waiting for {host}:{port}: {exc}")
            time.sleep(retry_s)
