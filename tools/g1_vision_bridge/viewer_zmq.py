from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_G1_CAMERA_HOST = "192.168.123.164"
DEFAULT_G1_CAMERA_PORT = 5555
DEFAULT_G1_CAMERA_INTERFACE = "enx6c1ff7c1192d"
DEFAULT_G1_RGB_KEY = "ego_view"
DEFAULT_G1_DEPTH_KEY = "ego_view_depth_m"

try:
    from .viewer import colorize_depth
except ImportError:  # Allow running as `python viewer_zmq.py`.
    from viewer import colorize_depth  # type: ignore


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python or opencv-python-headless is required for viewer_zmq.py.") from exc
    return cv2


def _load_g1_vision_module():
    module_path = Path(__file__).resolve().parents[2] / "capx" / "integrations" / "g1" / "vision.py"
    spec = importlib.util.spec_from_file_location("_capx_g1_vision_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load G1 vision API module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_rgb_canvas(frame: Any, *, received_fps: float) -> np.ndarray:
    cv2 = _load_cv2()
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if frame.depth_image is not None:
        depth = frame.depth_image
        if depth.ndim == 2:
            depth_bgr = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        else:
            depth_bgr = depth[:, :, :3]
        if depth_bgr.shape[:2] != bgr.shape[:2]:
            depth_bgr = cv2.resize(depth_bgr, (bgr.shape[1], bgr.shape[0]))
        out = np.hstack([bgr, depth_bgr])
    elif frame.depth_m is not None:
        depth_bgr = colorize_depth(frame.depth_m, max_depth_m=3.0)
        out = np.hstack([bgr, depth_bgr])
    else:
        out = bgr.copy()
    text_lines = [
        f"camera: {frame.rgb_key}",
        f"timestamp: {frame.timestamp:.6f}, fps: {received_fps:.1f}",
        "keys: s save, q quit",
    ]
    for idx, text in enumerate(text_lines):
        y = 24 + idx * 24
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return out


def save_frame(api: Any, frame: Any, output_dir: Path) -> None:
    saved = api.save_sample(frame, output_dir)
    print(f"[g1-zmq-viewer] saved {saved}")


def run(args: argparse.Namespace) -> None:
    cv2 = _load_cv2()
    frame_count = 0
    fps_window_start = time.time()
    received_fps = 0.0
    output_dir = Path(args.output_dir).expanduser()
    interface = None if args.no_interface_bind else args.interface
    vision = _load_g1_vision_module()
    camera_api = vision.G1CameraApi(
        config=vision.G1CameraConfig(
            host=args.robot_host,
            port=args.robot_port,
            interface=interface,
            topic=args.topic,
            rgb_key=args.camera_key,
            depth_key=args.depth_key,
        )
    )

    print(
        f"[g1-zmq-viewer] subscribing tcp://{args.robot_host}:{args.robot_port} "
        f"rgb_key={args.camera_key or '<first>'} depth_key={args.depth_key or '<none>'} "
        f"topic={args.topic!r} interface={interface or '<route>'}"
    )

    for frame in camera_api.samples():
        frame_count += 1
        now = time.time()
        elapsed = now - fps_window_start
        if elapsed >= 1.0:
            received_fps = frame_count / elapsed
            frame_count = 0
            fps_window_start = now

        if args.no_window:
            if frame_count == 1 or frame_count % args.log_every == 0:
                print(
                    f"[g1-zmq-viewer] camera={frame.rgb_key} rgb={frame.rgb.shape} "
                    f"timestamp={frame.timestamp:.6f} fps={received_fps:.1f}"
                )
            continue

        canvas = build_rgb_canvas(frame, received_fps=received_fps)
        cv2.imshow(args.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            save_frame(camera_api, frame, output_dir)

    if not args.no_window:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize protocol-A ZMQ msgpack JPEG camera frames from the G1."
    )
    parser.add_argument("--robot-host", default=DEFAULT_G1_CAMERA_HOST)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_G1_CAMERA_PORT)
    parser.add_argument("--interface", default=DEFAULT_G1_CAMERA_INTERFACE)
    parser.add_argument(
        "--no-interface-bind",
        action="store_true",
        help="Do not request ZMQ BINDTODEVICE; rely only on the OS route table.",
    )
    parser.add_argument("--camera-key", default=DEFAULT_G1_RGB_KEY, help="RGB key under msgpack['images'].")
    parser.add_argument("--depth-key", default=DEFAULT_G1_DEPTH_KEY, help="Depth image key under msgpack['images'].")
    parser.add_argument("--topic", default="", help="Optional ZMQ SUB topic prefix.")
    parser.add_argument("--window-name", default="G1 ZMQ RGB viewer")
    parser.add_argument("--output-dir", default="./outputs/g1_zmq_viewer")
    parser.add_argument("--no-window", action="store_true", help="Print frame stats without opening GUI.")
    parser.add_argument("--log-every", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
