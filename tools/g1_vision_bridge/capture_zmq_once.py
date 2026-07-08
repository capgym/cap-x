from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

DEFAULT_G1_CAMERA_HOST = "192.168.123.164"
DEFAULT_G1_CAMERA_PORT = 5555
DEFAULT_G1_CAMERA_INTERFACE = "enx6c1ff7c1192d"
DEFAULT_G1_RGB_KEY = "ego_view"
DEFAULT_G1_DEPTH_KEY = "ego_view_depth_m"


def _load_g1_vision_module():
    module_path = Path(__file__).resolve().parents[2] / "capx" / "integrations" / "g1" / "vision.py"
    spec = importlib.util.spec_from_file_location("_capx_g1_vision_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load G1 vision API module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> None:
    vision = _load_g1_vision_module()
    api = vision.G1CameraApi(
        config=vision.G1CameraConfig(
            host=args.robot_host,
            port=args.robot_port,
            interface=None if args.no_interface_bind else args.interface,
            topic=args.topic,
            rgb_key=args.camera_key,
            depth_key=args.depth_key,
        )
    )
    saved = api.save_camera_sample(args.output_dir)
    for name, path in saved.items():
        print(f"{name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one G1 protocol-A ZMQ camera sample.")
    parser.add_argument("--robot-host", default=DEFAULT_G1_CAMERA_HOST)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_G1_CAMERA_PORT)
    parser.add_argument("--interface", default=DEFAULT_G1_CAMERA_INTERFACE)
    parser.add_argument("--no-interface-bind", action="store_true")
    parser.add_argument("--camera-key", default=DEFAULT_G1_RGB_KEY)
    parser.add_argument("--depth-key", default=DEFAULT_G1_DEPTH_KEY)
    parser.add_argument("--topic", default="")
    parser.add_argument("--output-dir", default="./output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
