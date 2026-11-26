"""Run a recorded model trial paired with the isolated Route 2 launcher."""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import socketserver
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit


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
        raise ValueError("--server-url must be a host loopback URL, e.g. http://127.0.0.1:18089/v1/chat/completions")
    return address.port or 80, address.path


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


def _run_recorded_trial(env, trial, launch, config, multi_turn_prompt, output):
    from imageio_ffmpeg import count_frames_and_secs

    from capx.envs.runner import TRIAL_TIMEOUT_SECONDS, _run_single_trial_with_timeout
    from capx.utils.video_utils import _write_video

    output.mkdir()
    env.enable_video_capture(False, clear=True)
    summary = None
    error = None
    try:
        summary = _run_single_trial_with_timeout(
            env, trial, launch, {**config, "output_dir": str(output)},
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
    return {
        "trial": trial,
        "sandbox_rc": summary.sandbox_rc if summary else None,
        "code_blocks": summary.num_code_blocks if summary else 0,
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


def _evaluate(cli: argparse.Namespace, model_url: str) -> int:
    from capx.envs import trial as trial_module
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.launch import LaunchArgs
    from capx.envs.runner import _start_api_servers, _stop_api_servers
    from capx.utils.launch_utils import _load_config

    profile = {
        "S2": "tron2_cube_lift_visual_cgn_only.yaml",
        "M1": "tron2_cube_lift_visual_cgn_multiturn.yaml",
    }[cli.tier]
    launch = LaunchArgs(
        config_path=f"env_configs/tron2/{profile}", model=cli.model, server_url=model_url,
        api_key=os.getenv("TRON2_MODEL_API_KEY"),
        temperature=cli.temperature, max_tokens=cli.max_tokens,
        use_oracle_code=False, use_legacy_multi_turn_decision_prompt=True,
        total_trials=cli.trials, num_workers=1, record_video=True,
    )
    env_factory, config, api_servers = _load_config(launch)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt")
    if bool(multi_turn_prompt) != (cli.tier == "M1"):
        raise ValueError(f"The {cli.tier} config has an unexpected multi-turn setting")
    if cli.max_code_blocks is not None:
        trial_module.MULTITURN_LIMIT = cli.max_code_blocks - 1
    cli.output_dir.mkdir()
    for key in ("CAPX_TRON2_CGN_RANK", "CAPX_TRON2_CGN_CANDIDATES"):
        os.environ.pop(key, None)

    server_procs = _start_api_servers(api_servers)
    env = None
    results = []
    try:
        env = instantiate(env_factory)
        _wait_for_controller(env, cli.route2_startup_dir)
        for trial in range(1, cli.trials + 1):
            result = _run_recorded_trial(
                env, trial, launch, config, multi_turn_prompt, cli.output_dir / f"episode_{trial:02d}",
            )
            results.append(result)
            text = json.dumps({"tier": cli.tier, "model": cli.model, "trials": results}, indent=2)
            temporary = cli.output_dir / "results.tmp"
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.replace(cli.output_dir / "results.json")
            print(text, flush=True)
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
    parser.add_argument("--tier", choices=("S2", "M1"))
    parser.add_argument("--model")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--route2-startup-dir", type=Path)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-code-blocks", type=int)
    parser.add_argument("--temperature", type=float, default=0)
    cli = parser.parse_args()
    try:
        port, path = _model_endpoint(cli.server_url)
    except ValueError as exc:
        parser.error(str(exc))

    if cli.serve_relay:
        with _HostModelRelay(cli.serve_relay, port) as relay:
            cli.serve_relay.chmod(0o600)
            relay.serve_forever()
        return 0

    if not all((cli.tier, cli.model, cli.output_dir, cli.route2_startup_dir, cli.relay_socket)):
        parser.error("Use scripts/run_tron2_route2.sh --tier S2 or M1 with --model")
    if min(cli.trials, cli.max_tokens) <= 0 or (cli.max_code_blocks is not None and cli.max_code_blocks <= 0):
        parser.error("--trials, --max-tokens and --max-code-blocks must be positive")
    with _IsolatedModelRelay(cli.relay_socket) as relay:
        listener = threading.Thread(target=relay.serve_forever, daemon=True)
        listener.start()
        try:
            return _evaluate(cli, f"http://127.0.0.1:{relay.server_address[1]}{path}")
        finally:
            relay.shutdown()
            listener.join()


if __name__ == "__main__":
    raise SystemExit(main())
