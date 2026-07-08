from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .protocol import connect_with_retry, recv_framed
except ImportError:  # Allow running as `python viewer.py`.
    from protocol import connect_with_retry, recv_framed


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python or opencv-python-headless is required for viewer.py.") from exc
    return cv2


def colorize_depth(depth_m: Any, *, max_depth_m: float = 3.0) -> np.ndarray:
    """Convert a metric depth image to a uint8 BGR colormap."""

    cv2 = _load_cv2()
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have shape HxW or HxWx1, got {depth.shape}.")

    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0)
    clipped = np.zeros_like(depth, dtype=np.float32)
    clipped[valid] = np.clip(depth[valid], 0.0, max_depth_m)
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if max_depth_m > 0.0:
        normalized[valid] = ((clipped[valid] / max_depth_m) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _frame_stats(frame: dict[str, Any]) -> tuple[float, float, float]:
    depth = np.asarray(frame["depth_m"], dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    return (
        float(np.min(depth[valid])),
        float(np.max(depth[valid])),
        float(np.mean(depth[valid])),
    )


def build_display_canvas(frame: dict[str, Any], *, max_depth_m: float = 3.0) -> np.ndarray:
    """Build an RGB/depth side-by-side BGR canvas for OpenCV display."""

    cv2 = _load_cv2()
    rgb = np.asarray(frame["rgb"])
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape HxWx3, got {rgb.shape}.")
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    depth_color = colorize_depth(frame["depth_m"], max_depth_m=max_depth_m)
    if depth_color.shape[:2] != bgr.shape[:2]:
        depth_color = cv2.resize(depth_color, (bgr.shape[1], bgr.shape[0]))
    return np.ascontiguousarray(np.hstack([bgr, depth_color]))


def overlay_text(canvas: np.ndarray, frame: dict[str, Any], *, received_fps: float) -> np.ndarray:
    cv2 = _load_cv2()
    depth_min, depth_max, depth_mean = _frame_stats(frame)
    text_lines = [
        f"camera: {frame.get('camera_name', 'unknown')}",
        f"frame: {frame.get('frame_id', '-')}, fps: {received_fps:.1f}",
        f"depth m min/mean/max: {depth_min:.2f}/{depth_mean:.2f}/{depth_max:.2f}",
        "keys: s save, q quit",
    ]
    out = canvas.copy()
    for idx, text in enumerate(text_lines):
        y = 24 + idx * 24
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return out


def save_frame(frame: dict[str, Any], output_dir: Path) -> None:
    cv2 = _load_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    frame_id = frame.get("frame_id", "unknown")
    prefix = output_dir / f"{stamp}_frame_{frame_id}"
    rgb = np.asarray(frame["rgb"], dtype=np.uint8)
    depth = np.asarray(frame["depth_m"], dtype=np.float32)
    cv2.imwrite(str(prefix) + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(str(prefix) + "_depth_m.npy", depth)
    np.save(str(prefix) + "_intrinsics.npy", np.asarray(frame["intrinsics"], dtype=np.float32))
    np.save(str(prefix) + "_pose_mat.npy", np.asarray(frame["pose_mat"], dtype=np.float32))
    print(f"[g1-vision-viewer] saved {prefix}_rgb.png and numpy metadata")


def run(args: argparse.Namespace) -> None:
    cv2 = _load_cv2()
    sock = None
    frame_count = 0
    fps_window_start = time.time()
    received_fps = 0.0
    output_dir = Path(args.output_dir).expanduser()

    while True:
        try:
            if sock is None:
                print(f"[g1-vision-viewer] connecting to {args.robot_host}:{args.robot_port}")
                sock = connect_with_retry(args.robot_host, args.robot_port, retry_s=1.0)
                print("[g1-vision-viewer] connected")

            frame = recv_framed(sock)
            frame_count += 1
            now = time.time()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                received_fps = frame_count / elapsed
                frame_count = 0
                fps_window_start = now

            if args.no_window:
                if frame_count == 1:
                    depth_min, depth_max, depth_mean = _frame_stats(frame)
                    print(
                        f"[g1-vision-viewer] camera={frame.get('camera_name')} "
                        f"rgb={np.asarray(frame['rgb']).shape} depth={np.asarray(frame['depth_m']).shape} "
                        f"depth_min/mean/max={depth_min:.2f}/{depth_mean:.2f}/{depth_max:.2f}"
                    )
                continue

            canvas = build_display_canvas(frame, max_depth_m=args.max_depth_m)
            canvas = overlay_text(canvas, frame, received_fps=received_fps)
            cv2.imshow(args.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_frame(frame, output_dir)
        except (EOFError, OSError, ValueError) as exc:
            print(f"[g1-vision-viewer] connection error, reconnecting: {exc}")
            if sock is not None:
                sock.close()
                sock = None
            time.sleep(1.0)
    if sock is not None:
        sock.close()
    if not args.no_window:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize G1 RGB-D frames from publisher.")
    parser.add_argument("--robot-host", required=True, help="G1/onboard publisher IP address.")
    parser.add_argument("--robot-port", type=int, default=9100)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--window-name", default="G1 RGB-D viewer")
    parser.add_argument("--output-dir", default="./outputs/g1_vision_viewer")
    parser.add_argument("--no-window", action="store_true", help="Print frame stats without opening GUI.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

