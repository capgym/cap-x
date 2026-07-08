from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import numpy as np

try:
    from .viewer import colorize_depth
except ImportError:  # Allow direct script-style imports in tests/tools.
    from viewer import colorize_depth  # type: ignore


DEFAULT_VIEWER_HOST = "127.0.0.1"
DEFAULT_VIEWER_PORT = 9010


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python or opencv-python-headless is required for the web viewer.") from exc
    return cv2


@dataclass(frozen=True)
class FrameSnapshot:
    frame_id: int
    received_time: float
    rgb_key: str
    depth_key: str | None
    timestamp: float
    rgb: np.ndarray
    depth_m: np.ndarray | None
    depth_image: np.ndarray | None
    sam3_prompt: str | None = None
    sam3_received_time: float | None = None
    sam3_result_count: int = 0
    sam3_best_score: float | None = None
    sam3_overlay_rgb: np.ndarray | None = None
    sam3_error: str | None = None


@dataclass(frozen=True)
class Sam3Snapshot:
    frame_id: int
    received_time: float
    prompt: str
    result_count: int
    best_score: float | None
    overlay_rgb: np.ndarray | None
    error: str | None = None


def snapshot_from_sample(sample: Any, *, frame_id: int, received_time: float | None = None) -> FrameSnapshot:
    return FrameSnapshot(
        frame_id=int(frame_id),
        received_time=float(time.time() if received_time is None else received_time),
        rgb_key=str(getattr(sample, "rgb_key", "unknown")),
        depth_key=getattr(sample, "depth_key", None),
        timestamp=float(getattr(sample, "timestamp", 0.0)),
        rgb=np.asarray(getattr(sample, "rgb"), dtype=np.uint8).copy(),
        depth_m=None
        if getattr(sample, "depth_m", None) is None
        else np.asarray(getattr(sample, "depth_m"), dtype=np.float32).copy(),
        depth_image=None
        if getattr(sample, "depth_image", None) is None
        else np.asarray(getattr(sample, "depth_image")).copy(),
    )


def _encode_jpeg_data_url(image: np.ndarray, *, rgb_input: bool, quality: int = 85) -> str:
    cv2 = _load_cv2()
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image for JPEG encoding, got {arr.shape}.")
    u8 = np.clip(arr, 0, 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR) if rgb_input else u8
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode viewer image as JPEG.")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _sam3_result_summary(results: list[dict[str, Any]]) -> tuple[int, float | None]:
    if not results:
        return 0, None
    scores = [float(result.get("score", 0.0)) for result in results]
    return len(results), max(scores) if scores else None


def build_sam3_overlay(rgb: np.ndarray, results: list[dict[str, Any]], *, max_masks: int = 5) -> np.ndarray:
    cv2 = _load_cv2()
    base = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = base.copy()
    colors = [
        (254, 238, 178),
        (118, 185, 0),
        (249, 212, 255),
        (203, 245, 255),
        (255, 215, 215),
    ]
    borders = [
        (249, 197, 0),
        (38, 86, 0),
        (149, 47, 198),
        (0, 116, 223),
        (229, 32, 32),
    ]
    for idx, result in enumerate(results[:max_masks]):
        mask_obj = result.get("mask")
        if mask_obj is not None:
            mask = np.asarray(mask_obj, dtype=bool)
            if mask.shape[:2] == overlay.shape[:2]:
                color = np.asarray(colors[idx % len(colors)], dtype=np.float32)
                overlay[mask] = (0.45 * overlay[mask].astype(np.float32) + 0.55 * color).astype(np.uint8)
                contours, _ = cv2.findContours(
                    (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, borders[idx % len(borders)], thickness=2)
        box = result.get("box")
        if box is not None and len(box) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), borders[idx % len(borders)], 2)
            score = float(result.get("score", 0.0))
            cv2.putText(
                overlay,
                f"{score:.2f}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                borders[idx % len(borders)],
                1,
                cv2.LINE_AA,
            )
    return overlay


def update_snapshot_sam3(
    snapshot: FrameSnapshot,
    *,
    prompt: str,
    results: list[dict[str, Any]],
    overlay_rgb: np.ndarray | None = None,
    received_time: float | None = None,
    error: str | None = None,
) -> FrameSnapshot:
    result_count, best_score = _sam3_result_summary(results)
    if overlay_rgb is None and results:
        overlay_rgb = build_sam3_overlay(snapshot.rgb, results)
    return replace(
        snapshot,
        sam3_prompt=prompt,
        sam3_received_time=float(time.time() if received_time is None else received_time),
        sam3_result_count=result_count,
        sam3_best_score=best_score,
        sam3_overlay_rgb=None if overlay_rgb is None else np.asarray(overlay_rgb, dtype=np.uint8).copy(),
        sam3_error=error,
    )


def _sam3_payload(snapshot: FrameSnapshot | None, sam3_snapshot: Sam3Snapshot | None = None) -> dict[str, Any]:
    if sam3_snapshot is not None:
        payload: dict[str, Any] = {
            "ready": sam3_snapshot.overlay_rgb is not None and sam3_snapshot.error is None,
            "frame_id": sam3_snapshot.frame_id,
            "received_time": sam3_snapshot.received_time,
            "prompt": sam3_snapshot.prompt,
            "result_count": sam3_snapshot.result_count,
            "best_score": sam3_snapshot.best_score,
            "error": sam3_snapshot.error,
            "overlay_data_url": None,
        }
        if sam3_snapshot.overlay_rgb is not None:
            payload["overlay_data_url"] = _encode_jpeg_data_url(sam3_snapshot.overlay_rgb, rgb_input=True)
        return payload

    if snapshot is None or snapshot.sam3_prompt is None:
        return {"ready": False, "overlay_data_url": None, "error": None}
    payload = {
        "ready": snapshot.sam3_overlay_rgb is not None and snapshot.sam3_error is None,
        "frame_id": snapshot.frame_id,
        "received_time": snapshot.sam3_received_time,
        "prompt": snapshot.sam3_prompt,
        "result_count": snapshot.sam3_result_count,
        "best_score": snapshot.sam3_best_score,
        "error": snapshot.sam3_error,
        "overlay_data_url": None,
    }
    if snapshot.sam3_overlay_rgb is not None:
        payload["overlay_data_url"] = _encode_jpeg_data_url(snapshot.sam3_overlay_rgb, rgb_input=True)
    return payload


def _depth_stats(depth_m: np.ndarray | None) -> dict[str, float | int | None]:
    if depth_m is None:
        return {"valid_px": 0, "min_m": None, "mean_m": None, "max_m": None}
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    valid_px = int(np.count_nonzero(valid))
    if valid_px == 0:
        return {"valid_px": 0, "min_m": None, "mean_m": None, "max_m": None}
    values = depth[valid]
    return {
        "valid_px": valid_px,
        "min_m": float(np.min(values)),
        "mean_m": float(np.mean(values)),
        "max_m": float(np.max(values)),
    }


def _depth_image_for_view(snapshot: FrameSnapshot, *, max_depth_m: float) -> tuple[np.ndarray | None, list[int] | None]:
    if snapshot.depth_m is not None:
        depth = np.asarray(snapshot.depth_m, dtype=np.float32)
        return colorize_depth(depth, max_depth_m=max_depth_m), list(depth.shape)
    if snapshot.depth_image is None:
        return None, None
    cv2 = _load_cv2()
    depth_image = np.asarray(snapshot.depth_image)
    if depth_image.ndim == 2:
        return cv2.cvtColor(depth_image.astype(np.uint8), cv2.COLOR_GRAY2BGR), list(depth_image.shape)
    if depth_image.ndim == 3 and depth_image.shape[2] >= 3:
        return depth_image[:, :, :3].astype(np.uint8), list(depth_image.shape)
    return None, list(depth_image.shape)


def build_frame_payload(
    snapshot: FrameSnapshot | None,
    *,
    max_depth_m: float = 3.0,
    sam3_snapshot: Sam3Snapshot | None = None,
) -> dict[str, Any]:
    if snapshot is None:
        return {"ready": False, "sam3": _sam3_payload(None, sam3_snapshot)}

    depth_view, depth_shape = _depth_image_for_view(snapshot, max_depth_m=max_depth_m)
    payload: dict[str, Any] = {
        "ready": True,
        "frame_id": snapshot.frame_id,
        "received_time": snapshot.received_time,
        "timestamp": snapshot.timestamp,
        "rgb_key": snapshot.rgb_key,
        "depth_key": snapshot.depth_key,
        "rgb_shape": list(snapshot.rgb.shape),
        "depth_shape": depth_shape,
        "depth_stats": _depth_stats(snapshot.depth_m),
        "rgb_data_url": _encode_jpeg_data_url(snapshot.rgb, rgb_input=True),
        "depth_data_url": None,
        "sam3": _sam3_payload(snapshot, sam3_snapshot),
    }
    if depth_view is not None:
        payload["depth_data_url"] = _encode_jpeg_data_url(depth_view, rgb_input=False)
    return payload


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>G1 RGB-D Bridge Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background: #111; color: #eee; }
    body { margin: 0; padding: 16px; }
    header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    #status { font-size: 13px; color: #9ad; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .panel { background: #1b1b1b; border: 1px solid #333; border-radius: 6px; overflow: hidden; min-width: 0; }
    .panel h2 { font-size: 14px; margin: 0; padding: 8px 10px; background: #242424; border-bottom: 1px solid #333; }
    img { width: 100%; display: block; image-rendering: auto; background: #050505; aspect-ratio: 4 / 3; object-fit: contain; }
    #stats { margin-top: 12px; padding: 10px; background: #181818; border: 1px solid #333; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: pre-wrap; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>G1 RGB-D Bridge Viewer</h1>
    <div id="status">waiting for frames</div>
  </header>
  <main class="grid">
    <section class="panel"><h2>RGB</h2><img id="rgb" alt="RGB camera frame"></section>
    <section class="panel"><h2>Depth</h2><img id="depth" alt="Depth colormap"></section>
    <section class="panel"><h2>SAM3</h2><img id="sam3" alt="SAM3 segmentation overlay"></section>
  </main>
  <pre id="stats"></pre>
<script>
const pollMs = Number(new URLSearchParams(location.search).get('poll_ms') || '%POLL_MS%');
let lastFrame = null;
async function refresh() {
  try {
    const res = await fetch('/frame.json', {cache: 'no-store'});
    const data = await res.json();
    if (!data.ready) {
      document.getElementById('status').textContent = 'waiting for frames';
      return;
    }
    if (data.frame_id !== lastFrame) {
      lastFrame = data.frame_id;
      document.getElementById('rgb').src = data.rgb_data_url;
      if (data.depth_data_url) document.getElementById('depth').src = data.depth_data_url;
      if (data.sam3 && data.sam3.overlay_data_url) document.getElementById('sam3').src = data.sam3.overlay_data_url;
    }
    const age = Math.max(0, Date.now() / 1000 - data.received_time);
    document.getElementById('status').textContent = `frame ${data.frame_id}, age ${age.toFixed(2)}s`;
    document.getElementById('stats').textContent = JSON.stringify({
      rgb_key: data.rgb_key,
      depth_key: data.depth_key,
      rgb_shape: data.rgb_shape,
      depth_shape: data.depth_shape,
      timestamp: data.timestamp,
      depth_stats: data.depth_stats,
      sam3: data.sam3,
    }, null, 2);
  } catch (err) {
    document.getElementById('status').textContent = `viewer error: ${err}`;
  }
}
refresh();
setInterval(refresh, pollMs);
</script>
</body>
</html>
"""


class _ViewerHandler(BaseHTTPRequestHandler):
    server: "_ViewerHttpServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            html = _INDEX_HTML.replace("%POLL_MS%", str(self.server.viewer.poll_ms))
            self._send_bytes(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/frame.json":
            payload = self.server.viewer.frame_payload()
            self._send_bytes(
                HTTPStatus.OK,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                "application/json",
            )
            return
        self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain")


class _ViewerHttpServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], viewer: "G1VisionWebViewer") -> None:
        super().__init__(address, _ViewerHandler)
        self.viewer = viewer


class G1VisionWebViewer:
    def __init__(
        self,
        *,
        host: str = DEFAULT_VIEWER_HOST,
        port: int = DEFAULT_VIEWER_PORT,
        max_depth_m: float = 3.0,
        poll_ms: int = 200,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.max_depth_m = float(max_depth_m)
        self.poll_ms = int(poll_ms)
        self._lock = threading.Lock()
        self._latest: FrameSnapshot | None = None
        self._latest_sam3: Sam3Snapshot | None = None
        self._frame_id = 0
        self._server: _ViewerHttpServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        port = self.port
        if self._server is not None:
            port = int(self._server.server_address[1])
        return f"http://{self.host}:{port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _ViewerHttpServer((self.host, self.port), self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="g1-vision-web-viewer")
        self._thread.start()

    def update_sample(self, sample: Any) -> None:
        with self._lock:
            self._frame_id += 1
            self._latest = snapshot_from_sample(sample, frame_id=self._frame_id)

    def latest_snapshot(self) -> FrameSnapshot | None:
        with self._lock:
            return self._latest

    def update_sam3_results(
        self,
        frame_snapshot: FrameSnapshot,
        *,
        prompt: str,
        results: list[dict[str, Any]],
    ) -> None:
        result_count, best_score = _sam3_result_summary(results)
        overlay = build_sam3_overlay(frame_snapshot.rgb, results) if results else frame_snapshot.rgb.copy()
        with self._lock:
            self._latest_sam3 = Sam3Snapshot(
                frame_id=frame_snapshot.frame_id,
                received_time=time.time(),
                prompt=prompt,
                result_count=result_count,
                best_score=best_score,
                overlay_rgb=overlay,
                error=None,
            )

    def update_sam3_error(self, frame_snapshot: FrameSnapshot | None, *, prompt: str, error: str) -> None:
        with self._lock:
            self._latest_sam3 = Sam3Snapshot(
                frame_id=0 if frame_snapshot is None else frame_snapshot.frame_id,
                received_time=time.time(),
                prompt=prompt,
                result_count=0,
                best_score=None,
                overlay_rgb=None,
                error=error,
            )

    def frame_payload(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._latest
            sam3_snapshot = self._latest_sam3
        return build_frame_payload(snapshot, max_depth_m=self.max_depth_m, sam3_snapshot=sam3_snapshot)

    def close(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
