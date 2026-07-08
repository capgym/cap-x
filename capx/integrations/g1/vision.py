from __future__ import annotations

import base64
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import msgpack
import msgpack_numpy as msgpack_numpy
import numpy as np

msgpack_numpy.patch()

DEFAULT_G1_CAMERA_HOST = "192.168.123.164"
DEFAULT_G1_CAMERA_PORT = 5555
DEFAULT_G1_CAMERA_INTERFACE = "enx6c1ff7c1192d"
DEFAULT_G1_RGB_KEY = "ego_view"
DEFAULT_G1_DEPTH_KEY = "ego_view_depth_m"
DEFAULT_CAPX_CAMERA_NAME = "robot0_robotview"


@dataclass
class G1CameraConfig:
    """Configuration for the G1 protocol-A camera server."""

    host: str = DEFAULT_G1_CAMERA_HOST
    port: int = DEFAULT_G1_CAMERA_PORT
    interface: str | None = DEFAULT_G1_CAMERA_INTERFACE
    topic: str | bytes = b""
    rgb_key: str | None = DEFAULT_G1_RGB_KEY
    depth_key: str | None = DEFAULT_G1_DEPTH_KEY
    capx_camera_name: str = DEFAULT_CAPX_CAMERA_NAME
    intrinsics: np.ndarray | None = None
    pose_mat: np.ndarray | None = None
    depth_intrinsics: np.ndarray | None = None
    rgb_intrinsics: np.ndarray | None = None
    depth_to_rgb_mat: np.ndarray | None = None
    rgb_distortion_coeffs: np.ndarray | None = None
    align_rgb_to_depth: bool = False
    allow_default_intrinsics: bool = False
    allow_identity_extrinsics: bool = False
    constant_depth_m: float | None = None
    depth_scale: float = 1.0
    receive_timeout_ms: int | None = 5000
    jpeg_color_order: str = "rgb"


@dataclass(frozen=True)
class G1CameraSample:
    """One decoded G1 camera sample."""

    rgb_key: str
    rgb: np.ndarray
    timestamp: float
    raw_message: dict[str, Any]
    depth_key: str | None = None
    depth_image: np.ndarray | None = None
    depth_m: np.ndarray | None = None


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python or opencv-python-headless is required for the G1 camera API."
        ) from exc
    return cv2


def default_intrinsics_for_image(height: int, width: int) -> np.ndarray:
    focal = float(max(height, width))
    return np.asarray(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _quaternion_wxyz_from_matrix(rotation: Any) -> np.ndarray:
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

    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Rotation matrix produced a zero-norm quaternion.")
    return (quat / norm).astype(np.float32)


def _pose_xyz_wxyz_from_pose_mat(pose_mat: Any) -> np.ndarray:
    pose = np.asarray(pose_mat, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"pose_mat must have shape 4x4, got {pose.shape}.")
    return np.concatenate(
        [pose[:3, 3].astype(np.float32), _quaternion_wxyz_from_matrix(pose[:3, :3])]
    ).astype(np.float32)


def _g1_lowlevel_observation(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    timestamp: float,
) -> dict[str, Any]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[:, :, None]
    pose = _pose_xyz_wxyz_from_pose_mat(pose_mat)
    return {
        "camera_top": {
            "images": {
                "rgb": np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)),
                "depth": np.ascontiguousarray(depth.astype(np.float32)),
            },
            "intrinsics_matrix": np.ascontiguousarray(np.asarray(intrinsics, dtype=np.float32)),
            "pose": pose,
            "pose_mat": np.ascontiguousarray(np.asarray(pose_mat, dtype=np.float32)),
        },
        "timestamp": float(timestamp),
    }


def _normalize_message(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, bytes):
                try:
                    key = key.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            normalized[key] = _normalize_message(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_message(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_message(item) for item in value)
    return value


def unpack_camera_payload(payload: bytes) -> dict[str, Any]:
    decoded = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    decoded = _normalize_message(decoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected G1 camera payload to decode to dict, got {type(decoded)!r}.")
    return decoded


def _extract_image_payload(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("data", "image", "jpeg", "jpg", "bytes", "buffer", "rgb", "color", "value"):
            if key in value:
                return _extract_image_payload(value[key])
        if len(value) == 1:
            return _extract_image_payload(next(iter(value.values())))
        raise ValueError(f"Could not find image data in keys: {sorted(map(str, value.keys()))}.")
    return value


def _image_bytes(value: Any) -> bytes:
    value = _extract_image_payload(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("data:image") and "," in text:
            text = text.split(",", 1)[1]
        return base64.b64decode("".join(text.split()).encode("ascii"), validate=False)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    arr = np.asarray(value)
    if arr.ndim == 1 and arr.dtype == np.uint8:
        return arr.tobytes()
    raise ValueError(f"Unsupported encoded image payload type={type(value)!r}, shape={arr.shape}.")


def decode_image(value: Any, *, rgb: bool = True, jpeg_color_order: str = "bgr") -> np.ndarray:
    """Decode raw JPEG bytes or base64 JPEG into a numpy image."""

    cv2 = _load_cv2()
    if not isinstance(value, (str, bytes, bytearray, memoryview, dict)):
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            image = arr[:, :, :3]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(image)
        if arr.ndim == 2:
            return np.ascontiguousarray(arr)

    encoded = np.frombuffer(_image_bytes(value), dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if rgb else cv2.IMREAD_UNCHANGED
    decoded = cv2.imdecode(encoded, flag)
    if decoded is None:
        raise ValueError("OpenCV could not decode G1 camera image payload.")
    if rgb and decoded.ndim == 3:
        order = str(jpeg_color_order).lower()
        if order == "bgr":
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        elif order != "rgb":
            raise ValueError(f"jpeg_color_order must be 'rgb' or 'bgr', got {jpeg_color_order!r}.")
    return np.ascontiguousarray(decoded)


def _select_image(message: dict[str, Any], key: str | None) -> tuple[str, Any]:
    images = message.get("images")
    if images is None:
        images = message.get("image", message.get("rgb", message.get("color")))
    if images is None:
        raise ValueError("G1 camera payload must contain images/image/rgb/color.")
    if isinstance(images, dict):
        if key is not None:
            if key not in images:
                raise ValueError(f"Image key {key!r} not found in {sorted(map(str, images.keys()))}.")
            return str(key), images[key]
        if not images:
            raise ValueError("G1 camera payload images dict is empty.")
        selected_key = next(iter(images.keys()))
        return str(selected_key), images[selected_key]
    return str(key or DEFAULT_G1_RGB_KEY), images


def _timestamp(message: dict[str, Any], key: str) -> float:
    for top_key in ("timestamp", "ts", "time", "t"):
        if top_key in message:
            try:
                return float(message[top_key])
            except (TypeError, ValueError):
                pass
    timestamps = message.get("timestamps")
    if isinstance(timestamps, dict):
        for timestamp_key in (key, "timestamp", "ts", "time", "t"):
            if timestamp_key in timestamps:
                try:
                    return float(timestamps[timestamp_key])
                except (TypeError, ValueError):
                    pass
        for value in timestamps.values():
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return time.time()


def _as_depth_m(value: Any, *, depth_scale: float) -> np.ndarray | None:
    if value is None or isinstance(value, (str, bytes, bytearray, memoryview)):
        return None
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return None
    return np.ascontiguousarray(arr.astype(np.float32) * float(depth_scale))


def _candidate_depth_keys(rgb_key: str, configured_depth_key: str | None) -> list[str]:
    candidates = [
        configured_depth_key,
        f"{rgb_key}_depth_m",
        f"{rgb_key}_depth",
        "depth_m",
        "depth",
    ]
    out: list[str] = []
    for key in candidates:
        if key and key not in out:
            out.append(key)
    return out


def _find_depth_m(
    message: dict[str, Any],
    rgb_key: str,
    *,
    depth_key: str | None,
    depth_scale: float,
) -> tuple[str | None, np.ndarray | None]:
    for key in ("depth_m", "depth", "depth_image"):
        depth = _as_depth_m(message.get(key), depth_scale=depth_scale)
        if depth is not None:
            return key, depth

    depths = message.get("depths")
    if isinstance(depths, dict):
        for key in (rgb_key, "depth_m", "depth"):
            if key in depths:
                depth = _as_depth_m(depths[key], depth_scale=depth_scale)
                if depth is not None:
                    return key, depth

    images = message.get("images")
    if isinstance(images, dict):
        for key in _candidate_depth_keys(rgb_key, depth_key):
            if key in images:
                depth = _as_depth_m(images[key], depth_scale=depth_scale)
                if depth is not None:
                    return key, depth
        image_entry = images.get(rgb_key)
        if isinstance(image_entry, dict):
            for key in ("depth_m", "depth", "depth_image"):
                depth = _as_depth_m(image_entry.get(key), depth_scale=depth_scale)
                if depth is not None:
                    return key, depth
    return None, None


def _find_depth_image(
    message: dict[str, Any],
    rgb_key: str,
    depth_key: str | None,
) -> tuple[str | None, np.ndarray | None]:
    images = message.get("images")
    if not isinstance(images, dict):
        return None, None
    for key in _candidate_depth_keys(rgb_key, depth_key):
        if key not in images:
            continue
        value = images[key]
        if isinstance(value, (str, bytes, bytearray, memoryview, dict)):
            return key, decode_image(value, rgb=False)
    return None, None



def _validate_intrinsics_matrix(intrinsics: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {matrix.shape}.")
    return np.ascontiguousarray(matrix)


def _validate_pose_matrix(matrix: Any, *, name: str) -> np.ndarray:
    pose = np.asarray(matrix, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {pose.shape}.")
    return np.ascontiguousarray(pose)


def align_rgb_to_depth_frame(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    depth_intrinsics: np.ndarray,
    rgb_intrinsics: np.ndarray,
    depth_to_rgb_mat: np.ndarray,
    rgb_distortion_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Sample RGB pixels into the depth image frame using D435 calibration.

    Args:
        rgb: Color image in RGB camera pixel coordinates, shape (Hc, Wc, 3).
        depth_m: Metric depth image in depth camera coordinates, shape (Hd, Wd).
        depth_intrinsics: Depth camera K matrix.
        rgb_intrinsics: RGB/color camera K matrix.
        depth_to_rgb_mat: T_rgb_depth, mapping depth optical-frame points into
            RGB/color optical-frame points.
        rgb_distortion_coeffs: Optional RGB camera Brown-Conrady coefficients.

    Returns:
        RGB image resampled into depth pixel coordinates, shape (Hd, Wd, 3).
    """

    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    depth_arr = np.asarray(depth_m, dtype=np.float32)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb_arr.shape}.")
    if depth_arr.ndim != 2:
        raise ValueError(f"depth_m must have shape (H, W), got {depth_arr.shape}.")

    k_depth = _validate_intrinsics_matrix(depth_intrinsics, name="depth_intrinsics").astype(np.float64)
    k_rgb = _validate_intrinsics_matrix(rgb_intrinsics, name="rgb_intrinsics").astype(np.float64)
    t_rgb_depth = _validate_pose_matrix(depth_to_rgb_mat, name="depth_to_rgb_mat").astype(np.float64)

    height, width = depth_arr.shape
    u_depth, v_depth = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    z_depth = depth_arr.reshape(-1).astype(np.float64)
    u_flat = u_depth.reshape(-1).astype(np.float64)
    v_flat = v_depth.reshape(-1).astype(np.float64)

    valid_depth = np.isfinite(z_depth) & (z_depth > 0.0)
    x_depth = (u_flat - k_depth[0, 2]) * z_depth / k_depth[0, 0]
    y_depth = (v_flat - k_depth[1, 2]) * z_depth / k_depth[1, 1]
    points_depth = np.stack([x_depth, y_depth, z_depth], axis=1)

    rot = t_rgb_depth[:3, :3]
    trans = t_rgb_depth[:3, 3]
    points_rgb = points_depth @ rot.T + trans
    valid = valid_depth & np.all(np.isfinite(points_rgb), axis=1) & (points_rgb[:, 2] > 0.0)

    rgb_h, rgb_w = rgb_arr.shape[:2]
    if rgb_distortion_coeffs is not None and np.any(np.asarray(rgb_distortion_coeffs, dtype=np.float64) != 0.0):
        cv2 = _load_cv2()
        projected = np.full((points_rgb.shape[0], 2), np.nan, dtype=np.float64)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size:
            image_points, _ = cv2.projectPoints(
                points_depth[valid_idx].reshape(-1, 1, 3).astype(np.float64),
                cv2.Rodrigues(rot)[0],
                trans.astype(np.float64),
                k_rgb.astype(np.float64),
                np.asarray(rgb_distortion_coeffs, dtype=np.float64).reshape(-1),
            )
            projected[valid_idx] = image_points.reshape(-1, 2)
        u_rgb = np.rint(projected[:, 0]).astype(np.int64)
        v_rgb = np.rint(projected[:, 1]).astype(np.int64)
    else:
        u_rgb = np.rint(points_rgb[:, 0] * k_rgb[0, 0] / points_rgb[:, 2] + k_rgb[0, 2]).astype(np.int64)
        v_rgb = np.rint(points_rgb[:, 1] * k_rgb[1, 1] / points_rgb[:, 2] + k_rgb[1, 2]).astype(np.int64)

    valid &= (u_rgb >= 0) & (u_rgb < rgb_w) & (v_rgb >= 0) & (v_rgb < rgb_h)

    aligned = np.zeros((height * width, 3), dtype=np.uint8)
    valid_indices = np.flatnonzero(valid)
    aligned[valid_indices] = rgb_arr[v_rgb[valid_indices], u_rgb[valid_indices], :3]
    return np.ascontiguousarray(aligned.reshape(height, width, 3))

def _topic_bytes(topic: str | bytes) -> bytes:
    return topic if isinstance(topic, bytes) else topic.encode("utf-8")


class G1CameraApi:
    """Camera helpers for Unitree G1 protocol-A image servers.

    Functions:
      - capture_camera_sample() -> G1CameraSample
      - capture_camera_observation() -> dict
      - save_camera_sample(output_dir: str = "./output") -> dict
    """

    def __init__(self, env: Any | None = None, config: G1CameraConfig | None = None) -> None:
        self._env = env
        self._webui_enabled = False
        self.config = config or G1CameraConfig()

    def enable_webui(self, enabled: bool = True) -> None:
        self._webui_enabled = enabled

    def functions(self) -> dict[str, Any]:
        return {
            "capture_camera_sample": self.capture_camera_sample,
            "capture_camera_observation": self.capture_camera_observation,
            "save_camera_sample": self.save_camera_sample,
        }

    def combined_doc(self) -> str:
        lines: list[str] = []
        for name, fn in self.functions().items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(...)"
            doc = inspect.getdoc(fn) or ""
            lines.append(f"{name}{sig}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {line}" for line in doc.splitlines())
            lines.append("")
        return "\n".join(lines).strip()

    def decode_payload(self, payload: bytes) -> G1CameraSample:
        message = unpack_camera_payload(payload)
        rgb_key, rgb_value = _select_image(message, self.config.rgb_key)
        rgb = decode_image(rgb_value, rgb=True, jpeg_color_order=self.config.jpeg_color_order)
        depth_m_key, depth_m = _find_depth_m(
            message,
            rgb_key,
            depth_key=self.config.depth_key,
            depth_scale=self.config.depth_scale,
        )
        depth_image_key, depth_image = _find_depth_image(message, rgb_key, self.config.depth_key)
        return G1CameraSample(
            rgb_key=rgb_key,
            rgb=rgb,
            timestamp=_timestamp(message, rgb_key),
            raw_message=message,
            depth_key=depth_m_key or depth_image_key,
            depth_image=depth_image,
            depth_m=depth_m,
        )

    def make_subscriber(self) -> Any:
        try:
            import zmq  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for the G1 camera API.") from exc

        context = zmq.Context.instance()
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        if self.config.receive_timeout_ms is not None:
            sock.setsockopt(zmq.RCVTIMEO, int(self.config.receive_timeout_ms))
        sock.setsockopt(zmq.SUBSCRIBE, _topic_bytes(self.config.topic))

        bind_to_device = getattr(zmq, "BINDTODEVICE", None)
        if self.config.interface and bind_to_device is not None:
            try:
                sock.setsockopt_string(bind_to_device, self.config.interface)
            except Exception as exc:  # pragma: no cover - OS/libzmq specific.
                print(
                    f"[g1-camera-api] could not bind to {self.config.interface!r}: {exc}. "
                    "Continuing with OS routing."
                )
        sock.connect(f"tcp://{self.config.host}:{int(self.config.port)}")
        return sock

    def recv_payload(self, sock: Any) -> bytes:
        try:
            parts = sock.recv_multipart()
        except Exception as exc:
            try:
                import zmq  # type: ignore
            except ImportError:  # pragma: no cover - pyzmq already needed by make_subscriber.
                zmq = None  # type: ignore[assignment]

            if zmq is not None and isinstance(exc, zmq.Again):
                timeout = self.config.receive_timeout_ms
                timeout_text = (
                    f"within {timeout} ms"
                    if timeout is not None
                    else "before the ZMQ receive call returned"
                )
                raise TimeoutError(
                    f"No G1 camera message received {timeout_text} from "
                    f"tcp://{self.config.host}:{int(self.config.port)} "
                    f"topic={self.config.topic!r} interface={self.config.interface or '<route>'}. "
                    "Check that the robot camera publisher is running, the IP/port are correct, "
                    "the selected network interface can reach the robot, and the SUB topic matches."
                ) from exc
            raise
        if not parts:
            raise EOFError("Received empty ZMQ camera message.")
        return parts[-1]

    def samples(self) -> Iterator[G1CameraSample]:
        sock = self.make_subscriber()
        try:
            while True:
                yield self.decode_payload(self.recv_payload(sock))
        finally:
            sock.close(linger=0)

    def capture_camera_sample(self) -> G1CameraSample:
        """Capture one RGB/depth sample from the G1 camera server.

        Args:
            None.

        Returns:
            G1CameraSample containing RGB, optional metric depth, optional raw depth image, and timestamps.
        """

        sock = self.make_subscriber()
        try:
            return self.decode_payload(self.recv_payload(sock))
        finally:
            sock.close(linger=0)

    def _depth_for_observation(self, sample: G1CameraSample) -> np.ndarray:
        if sample.depth_m is not None:
            depth = np.asarray(sample.depth_m, dtype=np.float32)
        elif self.config.constant_depth_m is not None:
            depth = np.full(sample.rgb.shape[:2], float(self.config.constant_depth_m), dtype=np.float32)
        else:
            raise ValueError(
                "G1 camera sample does not contain metric depth_m. Current depth JPEG is only "
                "a visualization image; provide metric depth or set constant_depth_m for connectivity tests."
            )
        if depth.ndim != 2:
            raise ValueError(f"Depth shape must be 2D, got {depth.shape}.")
        if not self.config.align_rgb_to_depth and depth.shape != sample.rgb.shape[:2]:
            raise ValueError(f"Depth shape {depth.shape} does not match RGB shape {sample.rgb.shape[:2]}.")
        return np.ascontiguousarray(depth)

    def _intrinsics_for_observation(self, sample: G1CameraSample) -> np.ndarray:
        if self.config.align_rgb_to_depth:
            if self.config.depth_intrinsics is not None:
                return _validate_intrinsics_matrix(self.config.depth_intrinsics, name="depth_intrinsics")
            if self.config.intrinsics is not None:
                return _validate_intrinsics_matrix(self.config.intrinsics, name="intrinsics")
            raise ValueError("Depth intrinsics are required when align_rgb_to_depth=True.")
        if self.config.intrinsics is not None:
            return _validate_intrinsics_matrix(self.config.intrinsics, name="intrinsics")
        if self.config.allow_default_intrinsics:
            return default_intrinsics_for_image(*sample.rgb.shape[:2])
        raise ValueError("G1 camera intrinsics are required for CaP-X visual/grasp flow.")

    def _rgb_for_observation(self, sample: G1CameraSample, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        if not self.config.align_rgb_to_depth:
            return np.ascontiguousarray(np.asarray(sample.rgb, dtype=np.uint8))
        if self.config.rgb_intrinsics is None:
            raise ValueError("RGB intrinsics are required when align_rgb_to_depth=True.")
        if self.config.depth_to_rgb_mat is None:
            raise ValueError("depth_to_rgb_mat / T_rgb_depth is required when align_rgb_to_depth=True.")
        return align_rgb_to_depth_frame(
            rgb=sample.rgb,
            depth_m=depth,
            depth_intrinsics=intrinsics,
            rgb_intrinsics=self.config.rgb_intrinsics,
            depth_to_rgb_mat=self.config.depth_to_rgb_mat,
            rgb_distortion_coeffs=self.config.rgb_distortion_coeffs,
        )

    def _pose_mat_for_observation(self) -> np.ndarray:
        if self.config.pose_mat is not None:
            pose_mat = np.asarray(self.config.pose_mat, dtype=np.float32)
        elif self.config.allow_identity_extrinsics:
            pose_mat = np.eye(4, dtype=np.float32)
        else:
            raise ValueError("G1 camera extrinsics pose_mat are required for CaP-X visual/grasp flow.")
        if pose_mat.shape != (4, 4):
            raise ValueError(f"G1 camera pose_mat must be 4x4, got {pose_mat.shape}.")
        return np.ascontiguousarray(pose_mat)

    def sample_to_capx_observation(self, sample: G1CameraSample) -> dict[str, Any]:
        depth = self._depth_for_observation(sample)
        intrinsics = self._intrinsics_for_observation(sample)
        rgb = self._rgb_for_observation(sample, depth, intrinsics)
        return _g1_lowlevel_observation(
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            pose_mat=self._pose_mat_for_observation(),
            timestamp=sample.timestamp,
        )

    def capture_camera_observation(self) -> dict[str, Any]:
        """Capture one camera sample and convert it to the G1RealLowLevel observation schema.

        Args:
            None.

        Returns:
            Dict with "camera_top" and "timestamp", ready to send to G1RealLowLevel's observation server.
        """

        return self.sample_to_capx_observation(self.capture_camera_sample())

    def save_sample(self, sample: G1CameraSample, output_dir: str | Path = "./output") -> dict[str, str]:
        cv2 = _load_cv2()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = out_dir / f"{stamp}_{sample.rgb_key}"

        rgb_path = prefix.with_name(prefix.name + "_rgb.png")
        cv2.imwrite(str(rgb_path), cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR))
        saved = {"rgb": str(rgb_path)}

        if sample.depth_image is not None:
            depth_image_path = prefix.with_name(prefix.name + "_depth_image.png")
            cv2.imwrite(str(depth_image_path), sample.depth_image)
            saved["depth_image"] = str(depth_image_path)
        if sample.depth_m is not None:
            depth_m_path = prefix.with_name(prefix.name + "_depth_m.npy")
            np.save(depth_m_path, sample.depth_m)
            saved["depth_m"] = str(depth_m_path)

        meta_path = prefix.with_name(prefix.name + "_metadata.json")
        metadata = {
            "host": self.config.host,
            "port": self.config.port,
            "interface": self.config.interface,
            "rgb_key": sample.rgb_key,
            "depth_key": sample.depth_key,
            "timestamp": sample.timestamp,
            "rgb_shape": list(sample.rgb.shape),
            "rgb_dtype": str(sample.rgb.dtype),
            "depth_image_shape": None if sample.depth_image is None else list(sample.depth_image.shape),
            "depth_image_dtype": None if sample.depth_image is None else str(sample.depth_image.dtype),
            "has_metric_depth_m": sample.depth_m is not None,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        saved["metadata"] = str(meta_path)
        return saved

    def save_camera_sample(self, output_dir: str = "./output") -> dict[str, str]:
        """Capture one camera sample and save RGB/depth artifacts.

        Args:
            output_dir: Output directory. Defaults to "./output".

        Returns:
            Mapping from artifact name to saved path.
        """

        return self.save_sample(self.capture_camera_sample(), output_dir)
