from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

DEFAULT_G1_CAMERA_HOST = "192.168.123.164"
DEFAULT_G1_CAMERA_PORT = 5555
DEFAULT_G1_CAMERA_INTERFACE = "enx6c1ff7c1192d"
DEFAULT_G1_RGB_KEY = "ego_view"
DEFAULT_G1_DEPTH_KEY = "ego_view_depth_m"
DEFAULT_SAM3_URL = "http://127.0.0.1:8114"
DEFAULT_SAM3_CHECKPOINT_PATH = "capx/model_weights/sam3/sam3.pt"
DEFAULT_PROMPT = "displayer"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str) -> Any:
    module_path = _repo_root() / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_g1_vision_module() -> Any:
    return _load_module("_capx_g1_vision_api", "capx/integrations/g1/vision.py")


def _load_sam3_module() -> Any:
    return _load_module("_capx_sam3_api", "capx/integrations/vision/sam3.py")


def _host_port_from_url(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"SAM3 URL must include host and port, got {url!r}.")
    return parsed.hostname, int(parsed.port)


def _tcp_port_open(host: str, port: int, *, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def ensure_localhost_no_proxy() -> None:
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [entry.strip() for entry in current.split(",") if entry.strip()]
    for entry in ("127.0.0.1", "localhost"):
        if entry not in entries:
            entries.append(entry)
    value = ",".join(entries)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def build_sam3_server_command(
    *,
    host: str,
    port: int,
    device: str,
    checkpoint_path: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "capx.serving.launch_sam3_server",
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        device,
    ]
    if checkpoint_path:
        command.extend(["--checkpoint-path", checkpoint_path])
    return command


def ensure_sam3_server(
    *,
    sam3_url: str,
    start_sam3_server: bool,
    device: str,
    checkpoint_path: str | None,
    wait_timeout_s: float,
    output_dir: Path,
) -> subprocess.Popen[bytes] | None:
    host, port = _host_port_from_url(sam3_url)
    if _tcp_port_open(host, port):
        return None

    if not start_sam3_server:
        raise RuntimeError(
            f"SAM3 service is not reachable at {sam3_url}. Start it with:\n"
            f"  .venv/bin/python -m capx.serving.launch_sam3_server "
            f"--host {host} --port {port} --device {device} "
            f"--checkpoint-path {checkpoint_path or DEFAULT_SAM3_CHECKPOINT_PATH}\n"
            "or rerun this smoke test with --start-sam3-server."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sam3_server.log"
    log_file = log_path.open("ab")
    proc = subprocess.Popen(
        build_sam3_server_command(
            host=host,
            port=port,
            device=device,
            checkpoint_path=checkpoint_path,
        ),
        cwd=_repo_root(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + float(wait_timeout_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SAM3 server exited early. Check log: {log_path}")
        if _tcp_port_open(host, port):
            return proc
        time.sleep(1.0)

    proc.terminate()
    raise TimeoutError(f"SAM3 server did not become reachable within {wait_timeout_s:.1f}s. Log: {log_path}")


def summarize_sam3_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        mask = np.asarray(result.get("mask", []), dtype=bool)
        box = [float(v) for v in result.get("box", [])]
        summary.append(
            {
                "index": int(idx),
                "label": str(result.get("label", "")),
                "score": float(result.get("score", 0.0)),
                "box": box,
                "mask_area_px": int(mask.sum()) if mask.size else 0,
            }
        )
    return summary


def run(args: argparse.Namespace) -> None:
    ensure_localhost_no_proxy()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sam3_proc = ensure_sam3_server(
        sam3_url=args.sam3_url,
        start_sam3_server=args.start_sam3_server,
        device=args.sam3_device,
        checkpoint_path=args.sam3_checkpoint_path,
        wait_timeout_s=args.sam3_start_timeout_s,
        output_dir=output_dir,
    )

    vision = _load_g1_vision_module()
    camera_api = vision.G1CameraApi(
        config=vision.G1CameraConfig(
            host=args.robot_host,
            port=args.robot_port,
            interface=None if args.no_interface_bind else args.interface,
            topic=args.topic,
            rgb_key=args.camera_key,
            depth_key=args.depth_key,
            receive_timeout_ms=args.camera_timeout_ms,
        )
    )

    print(
        f"[g1-sam3-smoke] capturing camera sample from "
        f"tcp://{args.robot_host}:{args.robot_port} rgb_key={args.camera_key}"
    )
    sample = camera_api.capture_camera_sample()
    saved_camera = camera_api.save_sample(sample, output_dir)
    print(f"[g1-sam3-smoke] saved camera artifacts: {saved_camera}")

    sam3 = _load_sam3_module()
    sam3.SERVICE_URL = args.sam3_url.rstrip("/")
    sam3_fn = sam3.init_sam3(device=args.sam3_device)

    print(f"[g1-sam3-smoke] running SAM3 prompt={args.prompt!r} on rgb={sample.rgb.shape}")
    results = sam3_fn(sample.rgb, text_prompt=args.prompt)
    result_summary = summarize_sam3_results(results)
    print(f"[g1-sam3-smoke] SAM3 returned {len(result_summary)} result(s)")

    pil_image = Image.fromarray(sample.rgb).convert("RGB")
    sam3.visualize_sam3_results(
        pil_image,
        args.prompt,
        results,
        output_dir=output_dir,
        show=args.show,
    )

    summary_path = output_dir / f"sam3_{args.prompt.replace(' ', '_')}_summary.json"
    summary = {
        "prompt": args.prompt,
        "sam3_url": args.sam3_url,
        "camera": {
            "host": args.robot_host,
            "port": args.robot_port,
            "interface": None if args.no_interface_bind else args.interface,
            "rgb_key": sample.rgb_key,
            "depth_key": sample.depth_key,
            "timestamp": sample.timestamp,
            "rgb_shape": list(sample.rgb.shape),
            "has_metric_depth_m": sample.depth_m is not None,
        },
        "saved_camera": saved_camera,
        "result_count": len(result_summary),
        "results": result_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[g1-sam3-smoke] saved summary: {summary_path}")

    if sam3_proc is not None and args.stop_started_sam3_server:
        sam3_proc.terminate()
        print("[g1-sam3-smoke] stopped SAM3 server started by this smoke test")

    if not result_summary and not args.allow_no_detections:
        raise SystemExit(
            f"SAM3 returned no detections for prompt {args.prompt!r}. "
            "Artifacts were still saved for inspection."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CaP-X smoke test: capture G1 camera RGB-D and run SAM3 text segmentation."
    )
    parser.add_argument("--robot-host", default=DEFAULT_G1_CAMERA_HOST)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_G1_CAMERA_PORT)
    parser.add_argument("--interface", default=DEFAULT_G1_CAMERA_INTERFACE)
    parser.add_argument("--no-interface-bind", action="store_true")
    parser.add_argument("--camera-key", default=DEFAULT_G1_RGB_KEY)
    parser.add_argument("--depth-key", default=DEFAULT_G1_DEPTH_KEY)
    parser.add_argument("--topic", default="")
    parser.add_argument("--camera-timeout-ms", type=int, default=5000)
    parser.add_argument("--sam3-url", default=DEFAULT_SAM3_URL)
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-checkpoint-path", default=DEFAULT_SAM3_CHECKPOINT_PATH)
    parser.add_argument("--start-sam3-server", action="store_true")
    parser.add_argument("--stop-started-sam3-server", action="store_true")
    parser.add_argument("--sam3-start-timeout-s", type=float, default=180.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="./output/g1_sam3_smoke")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--allow-no-detections", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[g1-sam3-smoke] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
