"""Run one isolated Route 2 episode and forward its dashboard events over SSH."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


PREFIX = "CAPX_EVENT\t"
ACTION_LOG_MARKERS = ("[TRON2", "Code block", "budget_exhausted", "Traceback")


def complete_lines(stream):
    while True:
        start = stream.tell()
        line = stream.readline()
        if not line.endswith("\n"):
            stream.seek(start)
            return
        yield line


def forward(events, log):
    for line in complete_lines(events):
        print(line, end="", flush=True)
    if log is not None:
        for line in complete_lines(log):
            if line.startswith(ACTION_LOG_MARKERS):
                event = {"type": "note", "model": "sim", "title": "TRON2", "text": line.rstrip()}
                print(PREFIX + json.dumps(event), flush=True)


def idle_gpu(selection):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    for row in rows.splitlines():
        gpu, memory, utilization = (int(value) for value in row.split(","))
        if (selection == "auto" or gpu == int(selection)) and memory <= 256 and utilization == 0:
            return str(gpu)
    raise RuntimeError("No idle simulation GPU matches the selection. Resume this run when a card is free.")


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except ProcessLookupError:
        process.wait()
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(args):
    attempt = args.attempt
    attempt.mkdir(parents=True)
    status = {"exit_code": 1, "interrupted": False}
    process = None
    log = None
    try:
        gpu = idle_gpu(args.gpu)
        status["gpu"] = gpu
        bundle = Path(__file__).resolve().parent
        event_path = attempt / "dashboard-events.log"
        event_path.touch()
        output = attempt / "data"
        tabletop = output / "tabletop.log"
        vdm_token = sys.stdin.buffer.readline().decode().rstrip("\n")
        env = dict(os.environ)
        env.pop("TRON2_MODEL_API_KEY", None)
        env.pop("CAPX_TRON2_CGN_CANDIDATES", None)
        env.update(
            TRON2_VDM_API_KEY=vdm_token, TRON2_DASHBOARD_EVENTS=str(event_path),
            TRON2_PYTHON=str(bundle / "run_python"), TRON2_SDK_DIR=str(args.runtime / "sdk"),
            CAPX_PROMPT_VARIANT=args.variant, CAPX_THINKING_CODE_MODEL=args.model, CAPX_THINKING_MODE="thinking_on",
            CAPX_THINKING_REQUEST_TIMEOUT_S="540", CAPX_THINKING_TRIAL_TIMEOUT_S="3600",
            CAPX_TRON2_LEFT_MOTION="servoj", HF_HUB_OFFLINE="1", IMAGEIO_FFMPEG_NO_PREVENT_SIGINT="1",
            CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu,
            OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
            NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
        )
        url = "http://127.0.0.1:{}/v1/chat/completions"
        command = [
            "bash", "scripts/run_tron2_route2.sh", str(args.vendor), str(args.runtime / "opt"), str(output),
            "--tier", args.tier, "--model", args.model, "--server-url", url.format(args.model_port),
            "--first-seed", str(args.seed), "--trials", "1", "--max-code-blocks", str(args.blocks),
            "--max-tokens", "8192", "--temperature", "0", "--cube-xy-jitter-m", str(args.jitter),
        ]
        if args.tier == "M2":
            command.append("--model-is-vlm")
        if args.tier == "M3":
            command.extend([
                "--vdm-model", "openai/codex-cli-vdm", "--vdm-model-is-vlm", "--vdm-server-url", url.format(args.vdm_port),
            ])
        if args.reuse_verified_vision:
            command.append("--reuse-verified-vision")
        print("Simulation GPU {} | {} {} seed {}".format(gpu, args.variant, args.tier, args.seed), flush=True)
        with event_path.open(encoding="utf-8") as events, (attempt / "launcher.log").open("w") as launcher_log:
            process = subprocess.Popen(
                command, cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                stdout=launcher_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while process.poll() is None:
                if log is None and tabletop.is_file():
                    log = tabletop.open(encoding="utf-8", errors="replace")
                forward(events, log)
                # Disconnects close this pipe. Stop this attempt even if the Mac is gone.
                if select.select([sys.stdin], [], [], 0.25)[0] and not os.read(sys.stdin.fileno(), 4096):
                    raise KeyboardInterrupt
            forward(events, log)
            status["exit_code"] = process.returncode
    except KeyboardInterrupt:
        status.update(exit_code=130, interrupted=True)
    except BrokenPipeError:
        status.update(exit_code=130, interrupted=True)
    except Exception as error:
        status["error"] = str(error)
        print(str(error), file=sys.stderr, flush=True)
    finally:
        if process is not None and process.poll() is None:
            terminate(process)
        if log is not None:
            log.close()
        temporary = attempt / "launch_status.tmp"
        temporary.write_text(json.dumps(status) + "\n")
        temporary.replace(attempt / "launch_status.json")
    return status["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attempt", "repo", "vendor", "runtime"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("variant", "tier", "model", "gpu"):
        parser.add_argument("--" + name, required=True)
    for name in ("model-port", "vdm-port", "seed", "blocks"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--jitter", type=float, required=True)
    parser.add_argument("--reuse-verified-vision", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    args.attempt.parent.mkdir(parents=True, exist_ok=True)
    with (args.attempt.parent / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("The previous attempt is still stopping on 4090w. Resume this run shortly.", file=sys.stderr)
            return 1
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
