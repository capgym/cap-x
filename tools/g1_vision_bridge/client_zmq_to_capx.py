from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_G1_CAMERA_HOST = "192.168.123.164"
DEFAULT_G1_CAMERA_PORT = 5555
DEFAULT_G1_CAMERA_INTERFACE = "enx6c1ff7c1192d"
DEFAULT_G1_RGB_KEY = "ego_view"
DEFAULT_G1_DEPTH_KEY = "ego_view_depth_m"

try:
    from .calibration import intrinsics_from_values, load_depth_to_rgb_mat, load_distortion_coeffs, load_intrinsics, load_pose_mat
    from .protocol import connect_with_retry, recv_framed, send_framed
    from .web_viewer import DEFAULT_VIEWER_HOST, DEFAULT_VIEWER_PORT, G1VisionWebViewer
except ImportError:  # Allow running as `python client_zmq_to_capx.py`.
    from calibration import intrinsics_from_values, load_depth_to_rgb_mat, load_distortion_coeffs, load_intrinsics, load_pose_mat  # type: ignore
    from protocol import connect_with_retry, recv_framed, send_framed  # type: ignore
    from web_viewer import DEFAULT_VIEWER_HOST, DEFAULT_VIEWER_PORT, G1VisionWebViewer  # type: ignore


def _connect_capx(host: str, port: int) -> socket.socket:
    return connect_with_retry(host, port, retry_s=1.0)


def _load_g1_vision_module():
    module_path = Path(__file__).resolve().parents[2] / "capx" / "integrations" / "g1" / "vision.py"
    spec = importlib.util.spec_from_file_location("_capx_g1_vision_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load G1 vision API module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_sam3_module():
    module_path = Path(__file__).resolve().parents[2] / "capx" / "integrations" / "vision" / "sam3.py"
    spec = importlib.util.spec_from_file_location("_capx_viewer_sam3_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SAM3 API module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sam3_fn_from_args(args: argparse.Namespace) -> Any:
    sam3 = _load_sam3_module()
    sam3.SERVICE_URL = args.viewer_sam3_url.rstrip("/")
    return sam3.init_sam3(device=args.viewer_sam3_device)


def _load_intrinsics_from_args(args: argparse.Namespace) -> Any:
    intrinsics = load_intrinsics(args.intrinsics_yaml)
    if intrinsics is not None:
        return intrinsics
    if args.fx is not None or args.fy is not None:
        if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
            raise SystemExit("When using numeric intrinsics, pass all of --fx --fy --cx --cy.")
        fx = float(args.fx if args.fx is not None else args.fy)
        fy = float(args.fy if args.fy is not None else args.fx)
        return intrinsics_from_values(
            fx=fx,
            fy=fy,
            cx=float(args.cx),
            cy=float(args.cy),
        )
    return None


def _load_optional_intrinsics(path: str | None) -> Any:
    return None if path is None else load_intrinsics(path)


def _load_depth_intrinsics_from_args(args: argparse.Namespace) -> Any:
    if args.depth_intrinsics_yaml is not None:
        return load_intrinsics(args.depth_intrinsics_yaml)
    return _load_intrinsics_from_args(args)


def _camera_api_from_args(args: argparse.Namespace) -> Any:
    vision = _load_g1_vision_module()
    return vision.G1CameraApi(
        config=vision.G1CameraConfig(
            host=args.robot_host,
            port=args.robot_port,
            interface=None if args.no_interface_bind else args.interface,
            topic=args.topic,
            rgb_key=args.camera_key,
            depth_key=args.depth_key,
            capx_camera_name=args.capx_camera_name,
            intrinsics=_load_intrinsics_from_args(args),
            pose_mat=load_pose_mat(
                args.extrinsics_yaml,
                allow_identity=args.allow_identity_extrinsics,
            ),
            depth_intrinsics=_load_depth_intrinsics_from_args(args),
            rgb_intrinsics=_load_optional_intrinsics(args.rgb_intrinsics_yaml),
            depth_to_rgb_mat=load_depth_to_rgb_mat(args.depth_to_rgb_extrinsics_yaml),
            rgb_distortion_coeffs=load_distortion_coeffs(args.rgb_intrinsics_yaml),
            align_rgb_to_depth=args.align_rgb_to_depth,
            allow_default_intrinsics=args.allow_default_intrinsics,
            allow_identity_extrinsics=args.allow_identity_extrinsics,
            constant_depth_m=args.constant_depth_m,
            depth_scale=args.depth_scale,
        )
    )



def _viewer_sample_for_display(camera_api: Any, sample: Any) -> Any:
    """Return the exact RGB/depth frame that should be shown in the live viewer."""

    config = getattr(camera_api, "config", None)
    if not getattr(config, "align_rgb_to_depth", False):
        return sample

    observation = camera_api.sample_to_capx_observation(sample)
    camera = observation["camera_top"]
    depth = camera["images"]["depth"]
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    return SimpleNamespace(
        rgb_key=f"{getattr(sample, 'rgb_key', 'rgb')}:aligned_to_depth",
        depth_key=getattr(sample, "depth_key", None),
        timestamp=getattr(sample, "timestamp", 0.0),
        rgb=camera["images"]["rgb"],
        depth_m=depth,
        depth_image=None,
    )

def _run_viewer_sample_loop(
    camera_api: Any,
    viewer: Any,
    stop_event: threading.Event,
    *,
    reconnect_sleep_s: float = 1.0,
    log_errors: bool = True,
) -> None:
    while not stop_event.is_set():
        try:
            for sample in camera_api.samples():
                viewer.update_sample(_viewer_sample_for_display(camera_api, sample))
                if stop_event.is_set():
                    break
            if not stop_event.is_set():
                stop_event.wait(reconnect_sleep_s)
        except KeyboardInterrupt:
            stop_event.set()
            break
        except (EOFError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            if log_errors and not stop_event.is_set():
                print(f"[g1-zmq-capx] viewer camera error, reconnecting: {exc}")
            stop_event.wait(reconnect_sleep_s)


def _run_viewer_sam3_loop(
    viewer: Any,
    sam3_segment_fn: Any,
    prompt: str,
    stop_event: threading.Event,
    *,
    period_s: float = 2.0,
    log_errors: bool = True,
) -> None:
    last_frame_id: int | None = None
    while not stop_event.is_set():
        frame_snapshot = viewer.latest_snapshot()
        if frame_snapshot is None or frame_snapshot.frame_id == last_frame_id:
            stop_event.wait(period_s)
            continue
        last_frame_id = int(frame_snapshot.frame_id)
        try:
            results = sam3_segment_fn(frame_snapshot.rgb, text_prompt=prompt)
            viewer.update_sam3_results(frame_snapshot, prompt=prompt, results=results)
        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as exc:
            if hasattr(viewer, "update_sam3_error"):
                viewer.update_sam3_error(frame_snapshot, prompt=prompt, error=str(exc))
            if log_errors and not stop_event.is_set():
                print(f"[g1-zmq-capx] viewer SAM3 error: {exc}")
        stop_event.wait(period_s)


def _validate_args(args: argparse.Namespace) -> None:
    if args.intrinsics_yaml is None and args.fx is None and args.fy is None and not args.allow_default_intrinsics:
        raise SystemExit(
            "Missing camera intrinsics. Pass --intrinsics-yaml or --fx/--fy/--cx/--cy. "
            "--allow-default-intrinsics is only for connectivity tests."
        )
    if args.align_rgb_to_depth:
        if args.rgb_intrinsics_yaml is None:
            raise SystemExit("--align-rgb-to-depth requires --rgb-intrinsics-yaml.")
        if args.depth_to_rgb_extrinsics_yaml is None:
            raise SystemExit("--align-rgb-to-depth requires --depth-to-rgb-extrinsics-yaml containing T_rgb_depth.")
    if args.constant_depth_m is not None and args.constant_depth_m <= 0.0:
        raise SystemExit("--constant-depth-m must be positive when provided.")
    if args.viewer_sam3 and not args.viewer:
        raise SystemExit("--viewer-sam3 requires --viewer.")
    if args.viewer_sam3_period_s <= 0.0:
        raise SystemExit("--viewer-sam3-period-s must be positive.")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    camera_api = _camera_api_from_args(args)
    capx_sock: socket.socket | None = None
    forwarded = 0
    viewer = None
    viewer_stop_event: threading.Event | None = None
    viewer_thread: threading.Thread | None = None
    viewer_sam3_thread: threading.Thread | None = None
    if args.viewer:
        viewer = G1VisionWebViewer(
            host=args.viewer_host,
            port=args.viewer_port,
            max_depth_m=args.viewer_max_depth_m,
            poll_ms=args.viewer_poll_ms,
        )
        viewer.start()
        viewer_stop_event = threading.Event()
        viewer_camera_api = _camera_api_from_args(args)
        viewer_thread = threading.Thread(
            target=_run_viewer_sample_loop,
            args=(viewer_camera_api, viewer, viewer_stop_event),
            daemon=True,
            name="g1-zmq-web-viewer-camera",
        )
        viewer_thread.start()
        if args.viewer_sam3:
            sam3_segment_fn = _sam3_fn_from_args(args)
            viewer_sam3_thread = threading.Thread(
                target=_run_viewer_sam3_loop,
                args=(viewer, sam3_segment_fn, args.viewer_sam3_prompt, viewer_stop_event),
                kwargs={"period_s": args.viewer_sam3_period_s},
                daemon=True,
                name="g1-zmq-web-viewer-sam3",
            )
            viewer_sam3_thread.start()
            print(
                f"[g1-zmq-capx] web viewer SAM3 overlay: prompt={args.viewer_sam3_prompt!r} "
                f"url={args.viewer_sam3_url} period={args.viewer_sam3_period_s:.2f}s"
            )
        print(f"[g1-zmq-capx] web viewer: {viewer.url} (independent camera subscription)")

    if args.constant_depth_m is not None:
        print(
            f"[g1-zmq-capx] using constant depth {args.constant_depth_m:.3f} m. "
            "This is for connectivity tests, not real grasping."
        )

    while True:
        try:
            cfg = camera_api.config
            print(
                f"[g1-zmq-capx] subscribing tcp://{cfg.host}:{cfg.port} "
                f"rgb_key={cfg.rgb_key or '<first>'} depth_key={cfg.depth_key or '<none>'} "
                f"topic={cfg.topic!r} interface={cfg.interface or '<route>'}"
            )

            for sample in camera_api.samples():
                observation = camera_api.sample_to_capx_observation(sample)

                if args.dry_run:
                    forwarded += 1
                    if forwarded == 1 or forwarded % args.log_every == 0:
                        depth = observation["camera_top"]["images"]["depth"]
                        print(
                            f"[g1-zmq-capx] decoded {forwarded} frames "
                            f"rgb={sample.rgb.shape} depth={depth.shape} t={sample.timestamp:.6f}"
                        )
                    continue

                if capx_sock is None:
                    print(f"[g1-zmq-capx] connecting to CaP-X {args.capx_host}:{args.capx_port}")
                    capx_sock = _connect_capx(args.capx_host, args.capx_port)
                    print("[g1-zmq-capx] CaP-X connected")

                send_framed(capx_sock, observation)
                _latest_action = recv_framed(capx_sock)
                forwarded += 1
                if forwarded == 1 or forwarded % args.log_every == 0:
                    print(
                        f"[g1-zmq-capx] forwarded {forwarded} frames "
                        f"from {sample.rgb_key} at t={sample.timestamp:.6f}"
                    )
        except KeyboardInterrupt:
            break
        except (EOFError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            print(f"[g1-zmq-capx] bridge error, reconnecting: {exc}")
            if capx_sock is not None:
                capx_sock.close()
                capx_sock = None
            time.sleep(1.0)

    if capx_sock is not None:
        capx_sock.close()
    if viewer_stop_event is not None:
        viewer_stop_event.set()
    if viewer_sam3_thread is not None:
        viewer_sam3_thread.join(timeout=2.0)
    if viewer_thread is not None:
        viewer_thread.join(timeout=2.0)
    if viewer is not None:
        viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to G1 protocol-A ZMQ camera frames and forward observations to CaP-X."
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
    parser.add_argument("--capx-host", default="127.0.0.1")
    parser.add_argument("--capx-port", type=int, default=9000)
    parser.add_argument("--capx-camera-name", default="robot0_robotview")
    parser.add_argument("--extrinsics-yaml", default=None)
    parser.add_argument(
        "--allow-identity-extrinsics",
        action="store_true",
        help="Use identity camera pose for bench tests only.",
    )
    parser.add_argument("--intrinsics-yaml", default=None, help="Canonical intrinsics for CaP-X. For D435 raw streams, use depth intrinsics here.")
    parser.add_argument("--depth-intrinsics-yaml", default=None, help="Depth camera intrinsics for RGB-to-depth alignment. Defaults to --intrinsics-yaml.")
    parser.add_argument("--rgb-intrinsics-yaml", default=None, help="RGB/color camera intrinsics for RGB-to-depth alignment.")
    parser.add_argument("--depth-to-rgb-extrinsics-yaml", default=None, help="YAML containing T_rgb_depth, mapping depth optical frame to RGB optical frame.")
    parser.add_argument("--align-rgb-to-depth", action="store_true", help="Resample RGB into the depth camera frame before forwarding to CaP-X.")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument(
        "--allow-default-intrinsics",
        action="store_true",
        help="Use image-size-derived placeholder intrinsics for bench tests only.",
    )
    parser.add_argument(
        "--constant-depth-m",
        type=float,
        default=None,
        help="Fill every pixel with this depth for connectivity tests only.",
    )
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Scale numeric depth payloads into meters.")
    parser.add_argument("--dry-run", action="store_true", help="Decode and validate frames without connecting to CaP-X.")
    parser.add_argument("--viewer", action="store_true", help="Start a local web RGB/depth viewer without changing the CaP-X forwarding socket.")
    parser.add_argument("--viewer-host", default=DEFAULT_VIEWER_HOST)
    parser.add_argument("--viewer-port", type=int, default=DEFAULT_VIEWER_PORT)
    parser.add_argument("--viewer-max-depth-m", type=float, default=3.0)
    parser.add_argument("--viewer-poll-ms", type=int, default=200)
    parser.add_argument("--viewer-sam3", action="store_true", help="Show a live SAM3 text-prompt overlay in the web viewer.")
    parser.add_argument("--viewer-sam3-prompt", default="plastic water bottle")
    parser.add_argument("--viewer-sam3-url", default="http://127.0.0.1:8114")
    parser.add_argument("--viewer-sam3-device", default="cuda")
    parser.add_argument("--viewer-sam3-period-s", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
