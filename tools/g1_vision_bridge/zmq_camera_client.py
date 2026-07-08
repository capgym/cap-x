from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Iterator

import msgpack
import msgpack_numpy as msgpack_numpy
import numpy as np

msgpack_numpy.patch()

DEFAULT_ROBOT_HOST = "192.168.123.164"
DEFAULT_ZMQ_PORT = 5555
DEFAULT_INTERFACE = "enx6c1ff7c1192d"
DEFAULT_CAMERA_NAME = "robot0_robotview"


@dataclass(frozen=True)
class ZmqCameraFrame:
    camera_name: str
    rgb: np.ndarray
    timestamp: float
    raw_message: dict[str, Any]


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python or opencv-python-headless is required to decode JPEG camera frames."
        ) from exc
    return cv2


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


def _unpack_msgpack(payload: bytes) -> dict[str, Any]:
    decoded = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    decoded = _normalize_message(decoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected msgpack camera payload to decode to a dict, got {type(decoded)!r}.")
    return decoded


def _is_image_container(value: dict[Any, Any]) -> bool:
    return any(
        key in value
        for key in (
            "data",
            "image",
            "jpeg",
            "jpg",
            "bytes",
            "buffer",
            "rgb",
            "color",
            "value",
        )
    )


def _extract_image_payload(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("data", "image", "jpeg", "jpg", "bytes", "buffer", "rgb", "color", "value"):
            if key in value:
                return _extract_image_payload(value[key])
        if len(value) == 1:
            return _extract_image_payload(next(iter(value.values())))
        raise ValueError(f"Could not find JPEG image data in keys: {sorted(map(str, value.keys()))}.")
    return value


def decode_image_to_rgb(value: Any) -> np.ndarray:
    """Decode one protocol-A image field into an RGB uint8 OpenCV/numpy image."""

    value = _extract_image_payload(value)
    cv2 = _load_cv2()

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("data:image") and "," in text:
            text = text.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode("".join(text.split()).encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("String image payload must be base64-encoded JPEG data.") from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        image_bytes = bytes(value)
    else:
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.dtype == np.uint8:
            image_bytes = arr.tobytes()
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            rgb = arr[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(rgb)
        else:
            raise ValueError(f"Unsupported image payload shape/type: {type(value)!r}, shape={arr.shape}.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError("OpenCV could not decode JPEG camera payload.")
    rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def _select_image(message: dict[str, Any], camera_key: str | None) -> tuple[str, Any]:
    images = message.get("images")
    if images is None:
        images = message.get("image", message.get("rgb", message.get("color")))
    if images is None:
        raise ValueError("Camera payload must contain an 'images', 'image', 'rgb', or 'color' field.")

    fallback_name = str(
        camera_key
        or message.get("camera_name")
        or message.get("camera")
        or message.get("name")
        or DEFAULT_CAMERA_NAME
    )

    if isinstance(images, dict):
        if _is_image_container(images):
            return fallback_name, images
        if camera_key is not None:
            if camera_key not in images:
                raise ValueError(
                    f"Requested camera_key={camera_key!r} is not present in images keys "
                    f"{sorted(map(str, images.keys()))}."
                )
            return str(camera_key), images[camera_key]
        if not images:
            raise ValueError("Camera payload 'images' dict is empty.")
        selected_key = next(iter(images.keys()))
        return str(selected_key), images[selected_key]

    return fallback_name, images


def _timestamp_from_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_timestamp(message: dict[str, Any], camera_name: str) -> float:
    for key in ("timestamp", "ts", "time", "t"):
        if key in message:
            timestamp = _timestamp_from_value(message[key])
            if timestamp is not None:
                return timestamp

    timestamps = message.get("timestamps")
    if isinstance(timestamps, dict):
        for key in (camera_name, "timestamp", "ts", "time", "t"):
            if key in timestamps:
                timestamp = _timestamp_from_value(timestamps[key])
                if timestamp is not None:
                    return timestamp
        for value in timestamps.values():
            timestamp = _timestamp_from_value(value)
            if timestamp is not None:
                return timestamp

    return time.time()


def decode_zmq_camera_payload(payload: bytes, *, camera_key: str | None = None) -> ZmqCameraFrame:
    """Decode a protocol-A ZMQ msgpack payload into one RGB frame."""

    message = _unpack_msgpack(payload)
    camera_name, image_value = _select_image(message, camera_key)
    rgb = decode_image_to_rgb(image_value)
    timestamp = _select_timestamp(message, camera_name)
    return ZmqCameraFrame(
        camera_name=camera_name,
        rgb=rgb,
        timestamp=timestamp,
        raw_message=message,
    )


def _topic_bytes(topic: str | bytes) -> bytes:
    if isinstance(topic, bytes):
        return topic
    return topic.encode("utf-8")


def make_zmq_subscriber(
    *,
    host: str = DEFAULT_ROBOT_HOST,
    port: int = DEFAULT_ZMQ_PORT,
    topic: str | bytes = b"",
    interface: str | None = DEFAULT_INTERFACE,
    receive_timeout_ms: int | None = None,
    conflate: bool = True,
) -> Any:
    """Create and connect a ZMQ SUB socket for the robot camera server."""

    try:
        import zmq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyzmq is required for the protocol-A camera client.") from exc

    context = zmq.Context.instance()
    sock = context.socket(zmq.SUB)
    if conflate:
        sock.setsockopt(zmq.CONFLATE, 1)
    if receive_timeout_ms is not None:
        sock.setsockopt(zmq.RCVTIMEO, int(receive_timeout_ms))
    sock.setsockopt(zmq.SUBSCRIBE, _topic_bytes(topic))

    bind_to_device = getattr(zmq, "BINDTODEVICE", None)
    if interface and bind_to_device is not None:
        try:
            sock.setsockopt_string(bind_to_device, interface)
        except Exception as exc:  # pragma: no cover - depends on OS/libzmq permissions.
            print(
                f"[zmq-camera-client] could not bind socket to interface {interface!r}: {exc}. "
                "Continuing with OS routing."
            )
    elif interface:
        print(
            f"[zmq-camera-client] pyzmq/libzmq does not expose BINDTODEVICE; "
            f"using OS routing for interface {interface!r}."
        )

    sock.connect(f"tcp://{host}:{int(port)}")
    return sock


def recv_zmq_payload(sock: Any) -> bytes:
    """Receive a ZMQ message and return the msgpack payload part."""

    parts = sock.recv_multipart()
    if not parts:
        raise EOFError("Received empty ZMQ multipart message.")
    return parts[-1]


def iter_zmq_camera_frames(
    *,
    host: str = DEFAULT_ROBOT_HOST,
    port: int = DEFAULT_ZMQ_PORT,
    camera_key: str | None = None,
    topic: str | bytes = b"",
    interface: str | None = DEFAULT_INTERFACE,
    receive_timeout_ms: int | None = None,
) -> Iterator[ZmqCameraFrame]:
    sock = make_zmq_subscriber(
        host=host,
        port=port,
        topic=topic,
        interface=interface,
        receive_timeout_ms=receive_timeout_ms,
    )
    try:
        while True:
            yield decode_zmq_camera_payload(recv_zmq_payload(sock), camera_key=camera_key)
    finally:
        sock.close(linger=0)
