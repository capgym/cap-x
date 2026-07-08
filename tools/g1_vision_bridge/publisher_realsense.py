from __future__ import annotations

import argparse
import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .calibration import load_pose_mat
    from .protocol import build_frame, send_framed
except ImportError:  # Allow running as `python publisher_realsense.py`.
    from calibration import load_pose_mat
    from protocol import build_frame, send_framed


@dataclass
class CameraSample:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray


class MockRgbdCamera:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.period_s = 1.0 / float(fps)
        self.intrinsics = np.array(
            [[width, 0.0, width / 2.0], [0.0, width, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.frame_id = 0

    def frames(self) -> Iterator[CameraSample]:
        while True:
            start = time.time()
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            rgb[:, :, 1] = (self.frame_id * 5) % 255
            depth_m = np.full((self.height, self.width), 1.0, dtype=np.float32)
            self.frame_id += 1
            yield CameraSample(rgb=rgb, depth_m=depth_m, intrinsics=self.intrinsics)
            elapsed = time.time() - start
            if elapsed < self.period_s:
                time.sleep(self.period_s - elapsed)


class RealSenseRgbdCamera:
    def __init__(self, *, serial: str | None, width: int, height: int, fps: int) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required on the G1/onboard computer for --source realsense."
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.intrinsics = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def frames(self) -> Iterator[CameraSample]:
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue
            bgr = np.asanyarray(color.get_data())
            rgb = bgr[:, :, ::-1].copy()
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
            yield CameraSample(rgb=rgb, depth_m=depth_m, intrinsics=self.intrinsics)

    def close(self) -> None:
        self.pipeline.stop()


def _build_camera(args: argparse.Namespace) -> Any:
    if args.source == "mock":
        return MockRgbdCamera(width=args.width, height=args.height, fps=args.fps)
    return RealSenseRgbdCamera(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


def run(args: argparse.Namespace) -> None:
    pose_mat = load_pose_mat(
        args.extrinsics_yaml,
        allow_identity=args.allow_identity_extrinsics,
    )
    camera = _build_camera(args)
    frame_iter = camera.frames()
    frame_id = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind_host, args.port))
        server.listen(1)
        print(f"[g1-vision-publisher] listening on {args.bind_host}:{args.port}")
        while True:
            conn, addr = server.accept()
            print(f"[g1-vision-publisher] client connected from {addr}")
            with conn:
                while True:
                    sample = next(frame_iter)
                    frame = build_frame(
                        camera_name=args.camera_name,
                        rgb=sample.rgb,
                        depth_m=sample.depth_m,
                        intrinsics=sample.intrinsics,
                        pose_mat=pose_mat,
                        frame_id=frame_id,
                    )
                    frame_id += 1
                    try:
                        send_framed(conn, frame)
                    except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
                        print(f"[g1-vision-publisher] client disconnected: {exc}")
                        break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish G1 RGB-D frames for CaP-X.")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--source", choices=["realsense", "mock"], default="realsense")
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-name", default="robot0_robotview")
    parser.add_argument("--extrinsics-yaml", default=None)
    parser.add_argument(
        "--allow-identity-extrinsics",
        action="store_true",
        help="Use identity camera pose. Only use for bench tests, not robot grasping.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

