"""Run a recorded model trial paired with the isolated Route 2 launcher."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import select
import socket
import socketserver
import threading
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlsplit


_VISUAL_CONFIG = "tron2_cube_lift_visual_cgn_multiturn.yaml"
_PROFILES = {
    "S1": "tron2_cube_lift_privileged.yaml",
    "S2": "tron2_cube_lift_visual_cgn_only.yaml",
    "S3": "tron2_cube_lift_visual_reduced.yaml",
    "S4": "tron2_cube_lift_visual_reduced_exampleless.yaml",
    "M1": _VISUAL_CONFIG,
    "M2": _VISUAL_CONFIG,
    "M3": _VISUAL_CONFIG,
    "M4": "tron2_cube_lift_visual_reduced_multiturn.yaml",
}
_VDM_TIERS = ("M3", "M4")
_CGN_REPLAY_TIERS = ("S2", "M1", "M2", "M3")


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        # Keep both HTTP halves intact, including chunked responses and
        # half-closed clients. Neither side interprets nor logs request data.
        try:
            with self.server.connect_upstream() as upstream:
                upstream.settimeout(220)
                self.request.settimeout(220)
                peers = {self.request: upstream, upstream: self.request}
                while peers:
                    readable, _, _ = select.select(tuple(peers), (), (), 220)
                    if not readable:
                        break
                    for source in readable:
                        destination = peers[source]
                        data = source.recv(65536)
                        if data:
                            destination.sendall(data)
                        else:
                            destination.shutdown(socket.SHUT_WR)
                            del peers[source]
        except OSError:
            return


class _HostModelRelay(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, port: int) -> None:
        self.port = port
        super().__init__(str(path), _RelayHandler)

    def connect_upstream(self) -> socket.socket:
        return socket.create_connection(("127.0.0.1", self.port), timeout=5)


class _IsolatedModelRelay(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(("127.0.0.1", 0), _RelayHandler)

    def connect_upstream(self) -> socket.socket:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(5)
            client.connect(str(self.path))
        except BaseException:
            client.close()
            raise
        return client


def _model_endpoint(url: str) -> tuple[int, str]:
    address = urlsplit(url)
    if (address.scheme != "http" or address.hostname != "127.0.0.1"
            or address.username or address.password or address.query or address.fragment
            or not address.path):
        raise ValueError("Model endpoints must be host loopback URLs, e.g. http://127.0.0.1:18089/v1/chat/completions")
    return address.port or 80, address.path


@contextmanager
def _relayed_endpoint(socket_path: Path, host_url: str):
    _, path = _model_endpoint(host_url)
    with _IsolatedModelRelay(socket_path) as relay:
        listener = threading.Thread(target=relay.serve_forever, daemon=True)
        listener.start()
        try:
            yield f"http://127.0.0.1:{relay.server_address[1]}{path}"
        finally:
            relay.shutdown()
            listener.join()


def _check_vision_endpoint(model: str, url: str, api_key: str | None, declared_vlm: bool) -> None:
    """Verify the endpoint handles inline RGB before enabling CaP-X vision feedback."""
    import requests
    from PIL import Image

    from capx.llm import VLM_MODELS

    if model not in VLM_MODELS and not declared_vlm:
        raise ValueError(f"Declare the vision-capable endpoint for {model!r} with --model-is-vlm or --vdm-model-is-vlm")
    with io.BytesIO() as image:
        Image.new("RGB", (112, 112), color=(0, 210, 0)).save(image, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.post(
        url, headers=headers, timeout=120,
        json={"model": model, "max_tokens": 512, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color fills this image? Answer using one English color word."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}]},
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or "green" not in answer.lower():
        raise ValueError(f"Vision endpoint for {model!r} did not identify the RGB probe image")
    if model not in VLM_MODELS:
        VLM_MODELS.append(model)


def _wait_for_controller(env, startup_dir: Path) -> None:
    low = env.low_level_env
    low.start_motor_loop()
    (startup_dir / "motor.ready").touch()
    deadline = time.monotonic() + 75
    while not (startup_dir / "controller.ready").is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("The isolated TRON2 controller did not signal ready")
        low._check_physics()
        time.sleep(0.05)


def _run_recorded_trial(env, trial, seed, launch, config, multi_turn_prompt, output):
    from imageio_ffmpeg import count_frames_and_secs

    from capx.envs.runner import TRIAL_TIMEOUT_SECONDS, _run_single_trial_with_timeout
    from capx.utils.video_utils import _write_video

    output.mkdir()
    env.enable_video_capture(False, clear=True)
    summary = None
    error = None
    try:
        summary = _run_single_trial_with_timeout(
            env, seed, launch, {**config, "output_dir": str(output), "record_turn_metrics": True},
            multi_turn_prompt, TRIAL_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        error = str(exc)
        traceback.print_exc()

    trial_artifacts = Path(summary.code_path).parent if summary and summary.code_path else None
    video = (trial_artifacts / "video_combined.mp4") if trial_artifacts else None
    frames = 0
    video_error = None
    try:
        if video is None or not video.is_file():
            video = None
            if env.get_video_frame_count():
                _write_video(env.get_video_frames(clear=True), str(output), suffix="partial")
                video = output / "video_partial.mp4"
        if video is not None:
            frames = count_frames_and_secs(str(video))[0]
    except (OSError, RuntimeError, ValueError) as exc:
        video_error = str(exc)
        traceback.print_exc()
    turn_metrics_path = output / "turn_metrics.jsonl"
    turns = []
    if turn_metrics_path.is_file():
        turns = [json.loads(line) for line in turn_metrics_path.read_text().splitlines()]
    return {
        "trial": trial,
        "seed": seed,
        "initial_cube_world_m": env.low_level_env._cube_init_world.tolist() if summary else None,
        "sandbox_rc": summary.sandbox_rc if summary else None,
        "code_blocks": summary.num_code_blocks if summary else 0,
        "code_blocks_in_final_plan": summary.num_code_blocks if summary else 0,
        "executed_code_blocks": len(turns),
        "turns": turns,
        "regenerations": summary.num_regenerations if summary else 0,
        "model_finishes": summary.num_finishes if summary else 0,
        "reward": summary.reward if summary else None,
        "task_completed": summary.task_completed if summary else False,
        "video": str(video.relative_to(output.parent)) if video else None,
        "video_frames": frames,
        "video_error": video_error,
        "trial_artifacts": str(trial_artifacts.relative_to(output.parent)) if trial_artifacts else None,
        "error": error,
    }


def _evaluate(cli: argparse.Namespace, model_url: str, vdm_url: str | None = None) -> int:
    from capx.envs import trial as trial_module
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.launch import LaunchArgs
    from capx.envs.runner import _start_api_servers, _stop_api_servers
    from capx.llm.client import is_openrouter_model
    from capx.utils.launch_utils import _load_config

    profile = f"env_configs/tron2/{_PROFILES[cli.tier]}"
    coding_api_key = os.getenv("TRON2_MODEL_API_KEY")
    vdm_model = (cli.vdm_model or cli.model) if cli.tier in _VDM_TIERS else None
    vdm_api_key = os.getenv("TRON2_VDM_API_KEY")
    if vdm_api_key is None and (not cli.vdm_server_url or cli.vdm_server_url == cli.server_url):
        vdm_api_key = coding_api_key
    visual_feedback = cli.tier == "M2"
    if is_openrouter_model(cli.model) or (vdm_model and is_openrouter_model(vdm_model)):
        raise ValueError("Model IDs routed to the upstream OpenRouter proxy cannot use the isolated host endpoint")
    launch = LaunchArgs(
        config_path=profile, model=cli.model, server_url=model_url, api_key=coding_api_key,
        temperature=cli.temperature, max_tokens=cli.max_tokens,
        use_oracle_code=False, use_legacy_multi_turn_decision_prompt=True,
        use_visual_feedback=visual_feedback, use_img_differencing=vdm_model is not None,
        use_video_differencing=False, use_wrist_camera=False,
        visual_differencing_model=vdm_model,
        visual_differencing_model_server_url=vdm_url or model_url,
        visual_differencing_model_api_key=vdm_api_key,
        total_trials=cli.trials, num_workers=1, record_video=True,
    )
    env_factory, config, api_servers = _load_config(launch)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt")
    if bool(multi_turn_prompt) != cli.tier.startswith("M"):
        raise ValueError(f"The {cli.tier} config has an unexpected multi-turn setting")
    privileged = env_factory["cfg"].get("privileged", False)
    if privileged != (cli.tier == "S1"):
        raise ValueError(f"The {cli.tier} config has an unexpected privileged-observation setting")
    if cli.max_code_blocks is not None:
        trial_module.MULTITURN_LIMIT = cli.max_code_blocks - 1
    cli.output_dir.mkdir()
    os.environ.pop("CAPX_TRON2_CGN_RANK", None)
    os.environ.pop("CAPX_TRON2_CGN_JAW_BRANCH", None)
    os.environ.pop("CAPX_TRON2_CGN_CANDIDATES", None)
    if cli.candidate_file is not None:
        os.environ["CAPX_TRON2_CGN_CANDIDATES"] = str(cli.candidate_file)

    results = []
    report = {
        "tier": cli.tier,
        "model": cli.model,
        "model_server_url": cli.server_url,
        "visual_differencing_model": launch.visual_differencing_model,
        "visual_differencing_server_url": (cli.vdm_server_url or cli.server_url) if vdm_model else None,
        "profile": profile,
        "apis": env_factory["cfg"].get("apis", []),
        "privileged": privileged,
        "multi_turn": bool(multi_turn_prompt),
        "use_visual_feedback": config["use_visual_feedback"],
        "use_img_differencing": config["use_img_differencing"],
        "use_video_differencing": config["use_video_differencing"],
        "record_video": config["record_video"],
        "left_motion_backend": os.environ.get("CAPX_TRON2_LEFT_MOTION", "movej").lower(),
        "first_seed": cli.first_seed,
        "cube_xy_jitter_m": cli.cube_xy_jitter_m,
        "temperature": launch.temperature,
        "max_tokens": launch.max_tokens,
        "max_code_blocks": trial_module.MULTITURN_LIMIT + 1,
        "candidate_file": str(cli.candidate_file) if cli.candidate_file else None,
        "verified_vision_endpoints": [],
        "trials": results,
    }

    def save_report() -> str:
        text = json.dumps(report, indent=2)
        temporary = cli.output_dir / "results.tmp"
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(cli.output_dir / "results.json")
        return text

    save_report()
    server_procs = _start_api_servers(api_servers)
    env = None
    try:
        env = instantiate(env_factory)
        env.low_level_env.cube_xy_jitter_m = cli.cube_xy_jitter_m
        _wait_for_controller(env, cli.route2_startup_dir)
        if visual_feedback:
            _check_vision_endpoint(launch.model, launch.server_url, coding_api_key, cli.model_is_vlm)
            report["verified_vision_endpoints"].append("coding")
        if vdm_model:
            _check_vision_endpoint(vdm_model, launch.visual_differencing_model_server_url, vdm_api_key,
                                   cli.vdm_model_is_vlm or (vdm_model == cli.model and cli.model_is_vlm))
            report["verified_vision_endpoints"].append("visual_differencing")
        save_report()
        for trial in range(1, cli.trials + 1):
            seed = cli.first_seed + trial - 1
            result = _run_recorded_trial(
                env, trial, seed, launch, config, multi_turn_prompt, cli.output_dir / f"episode_{trial:02d}",
            )
            results.append(result)
            print(save_report(), flush=True)
    finally:
        try:
            if env is not None:
                env.low_level_env.close()
        finally:
            _stop_api_servers(server_procs)
    return int(any(row["error"] or not row["video_frames"] for row in results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-relay", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--relay-socket", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--vdm-relay-socket", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--tier", choices=tuple(_PROFILES))
    parser.add_argument("--model")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-is-vlm", action="store_true", help="Declare a custom served coding model as image-capable.")
    parser.add_argument("--vdm-model", help="Image differencing model for M3/M4. Defaults to --model.")
    parser.add_argument("--vdm-server-url", help="Host loopback VDM endpoint. Defaults to --server-url.")
    parser.add_argument("--vdm-model-is-vlm", action="store_true", help="Declare a custom served VDM model as image-capable.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--route2-startup-dir", type=Path)
    parser.add_argument("--candidate-file", type=Path, help="Replay a saved CGN batch for diagnostics.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--first-seed", type=int, default=1, help="First episode seed. Subsequent episodes increment by one.")
    parser.add_argument("--cube-xy-jitter-m", type=float, default=0.0,
                        help="Uniform cube reset jitter in each horizontal axis, in meters.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-code-blocks", type=int)
    parser.add_argument("--temperature", type=float, default=0)
    cli = parser.parse_args()
    try:
        port, _ = _model_endpoint(cli.server_url)
        if cli.vdm_server_url:
            _model_endpoint(cli.vdm_server_url)
    except ValueError as exc:
        parser.error(str(exc))

    if cli.serve_relay:
        with _HostModelRelay(cli.serve_relay, port) as relay:
            cli.serve_relay.chmod(0o600)
            relay.serve_forever()
        return 0

    if not all((cli.tier, cli.model, cli.output_dir, cli.route2_startup_dir, cli.relay_socket)):
        parser.error("Use scripts/run_tron2_route2.sh --tier S1–S4 or M1–M4 with --model")
    if min(cli.trials, cli.max_tokens) <= 0 or (cli.max_code_blocks is not None and cli.max_code_blocks <= 0):
        parser.error("--trials, --max-tokens and --max-code-blocks must be positive")
    if cli.candidate_file is not None and not cli.candidate_file.is_file():
        parser.error("--candidate-file must point to a saved CGN proposal batch")
    if cli.candidate_file and cli.tier not in _CGN_REPLAY_TIERS:
        parser.error("--candidate-file is supported for S2 and M1–M3")
    if cli.first_seed < 0:
        parser.error("--first-seed must be nonnegative")
    if not math.isfinite(cli.cube_xy_jitter_m) or cli.cube_xy_jitter_m < 0:
        parser.error("--cube-xy-jitter-m must be finite and nonnegative")
    if cli.tier not in _VDM_TIERS and any((cli.vdm_model, cli.vdm_server_url, cli.vdm_model_is_vlm)):
        parser.error("VDM settings are available for M3 and M4")
    if cli.vdm_server_url and not cli.vdm_relay_socket:
        parser.error("The namespace launcher must supply a relay for --vdm-server-url")
    with ExitStack() as stack:
        model_url = stack.enter_context(_relayed_endpoint(cli.relay_socket, cli.server_url))
        vdm_url = None
        if cli.vdm_server_url and cli.vdm_server_url != cli.server_url:
            vdm_url = stack.enter_context(_relayed_endpoint(cli.vdm_relay_socket, cli.vdm_server_url))
        return _evaluate(cli, model_url, vdm_url)


if __name__ == "__main__":
    raise SystemExit(main())
