#!/usr/bin/env python3
"""Launch and resume recorded TRON2 evaluations from the Mac."""

import argparse
import base64
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, build_opener

from tron2_paths import TRON2_REPO


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "tron2_runner"
RUNS = TRON2_REPO / "outputs" / "tron2_dashboard_runs"
MODEL = "Qwen/Qwen3.8-27B"
VDM_MODEL = "openai/codex-cli-vdm"
BUNDLED_CODEX = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"
EVENT_PREFIX = "CAPX_EVENT\t"
CONDITIONS = tuple(f"{variant}:{tier}" for variant in ("v1", "v2") for tier in ("M1", "M2", "M3"))
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
HTTP = build_opener(ProxyHandler({}))


def save_bytes(path, data):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def save_manifest(root, manifest):
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    save_bytes(root / "manifest.json", text.encode())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class SSH:
    def __init__(self, host, control):
        self.host = host
        self.options = ["ssh", "-S", str(control), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

    def __enter__(self):
        subprocess.run(self.options + [
            "-fNTM", "-o", "ControlPersist=no", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", self.host,
        ], check=True)
        return self

    def __exit__(self, *_):
        subprocess.run(self.options + ["-O", "exit", self.host], capture_output=True)

    def command(self, args):
        return self.options + ["-T", self.host, shlex.join(map(str, args))]

    def run(self, *args, **options):
        return subprocess.run(self.command(args), check=True, **options)

    def forward(self, option, address):
        result = subprocess.run(self.options + [
            "-O", "forward", "-o", "ExitOnForwardFailure=yes", option, address, self.host,
        ], check=True, text=True, stdout=subprocess.PIPE)
        return result.stdout.strip()


def loopback_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def from_http(url):
    with HTTP.open(url, timeout=3) as response:
        return json.load(response)


def stop_local(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=6)
    except ProcessLookupError:
        process.wait()
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def episode_url(port, root, episode):
    run = f"tron2:{root.name}:{episode['id']}"
    return f"http://127.0.0.1:{port}/?run=" + quote(run, safe=":")


def dashboard(root, port, first_episode):
    identity = f"tron2:{root.name}:{first_episode['id']}"
    url = f"http://127.0.0.1:{port}/api/runs"
    try:
        entries = from_http(url)
    except (ConnectionError, URLError, TimeoutError, ValueError):
        entries = None
    if isinstance(entries, list) and any(entry.get("id") == identity for entry in entries):
        return
    if entries is not None:
        raise RuntimeError(f"Port {port} is serving another dashboard version. Use --dashboard-port {port + 1}.")
    with (root / "dashboard.log").open("ab") as log:
        process = subprocess.Popen([
            sys.executable, str(HERE / "dashboard.py"), "serve", "--port", str(port),
        ], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if process.poll() is not None:
            break
        try:
            if any(row.get("id") == identity for row in from_http(url)):
                return
        except (ConnectionError, URLError, TimeoutError, ValueError):
            pass
        time.sleep(0.1)
    stop_local(process)
    raise RuntimeError(f"Dashboard could not register this run. See {root / 'dashboard.log'}.")


def find_vdm_cli():
    configured = os.environ.get("CODEX_BIN")
    if configured:
        cli = shutil.which(os.path.expanduser(configured))
        if not cli:
            raise RuntimeError(f"CODEX_BIN is not executable: {configured}")
    else:
        cli = shutil.which("codex") or shutil.which(BUNDLED_CODEX)
    if not cli:
        raise RuntimeError("Codex CLI was not found. Add it to PATH or set CODEX_BIN.")
    account = subprocess.run([cli, "login", "status"], capture_output=True, text=True, timeout=10)
    if account.returncode != 0 or "Logged in using ChatGPT" not in account.stdout + account.stderr:
        raise RuntimeError("Run codex login on the Mac before running M3.")
    return cli


def open_vdm(stack, sim, root, cli):
    local = loopback_port()
    token = secrets.token_urlsafe(32)
    env = dict(os.environ)
    env.setdefault("CAPX_CODEX_VDM_TIMEOUT", "180")
    with (root / "vdm.log").open("ab") as log:
        process = subprocess.Popen([
            sys.executable, str(HERE / "codex_vdm_server.py"), cli, str(local),
        ], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    stack.callback(stop_local, process)
    process.stdin.write((token + "\n").encode())
    process.stdin.close()
    for _ in range(40):
        if process.poll() is not None:
            break
        try:
            if from_http(f"http://127.0.0.1:{local}/health")["model"] == VDM_MODEL:
                remote_port = sim.forward("-R", f"127.0.0.1:0:127.0.0.1:{local}")
                return int(remote_port), token
        except (ConnectionError, URLError, TimeoutError, ValueError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Mac VDM bridge failed to start. See {root / 'vdm.log'}.")


def stop_owned_a800(a800, owned):
    with (BUNDLE / "stop_a800.py").open("rb") as script:
        result = subprocess.run(a800.command(["python3", "-", *owned]), stdin=script)
    if result.returncode:
        print(f"A800 cleanup returned {result.returncode}. Owned run directory {owned[0]}", file=sys.stderr)


def open_model(stack, sim, a800, args, root):
    port = args.a800_port
    if port is None:
        print("Starting an owned A800 Graph + MTP endpoint.", flush=True)
        command = ["bash", "-s", "--", args.a800_gpu]
        if args.a800_base is not None:
            command.append(args.a800_base)
        with (BUNDLE / "start_a800.sh").open("rb") as script:
            result = a800.run(*command, stdin=script, stdout=subprocess.PIPE)
        owned = re.search(rb"^RUN_DIR=([^\r\n]+) PID=(\d+) PORT=(\d+) GPU=(\d+)$", result.stdout, re.MULTILINE)
        if not owned:
            raise RuntimeError("A800 starter did not return ownership information.")
        metadata = [value.decode() for value in owned.groups()]
        stack.callback(stop_owned_a800, a800, metadata)
        save_bytes(root / "a800.json", json.dumps(dict(zip(("run_dir", "pid", "port", "gpu"), metadata))).encode())
        port = int(metadata[2])
    local = loopback_port()
    a800.forward("-L", f"127.0.0.1:{local}:127.0.0.1:{port}")
    for _ in range(100):
        try:
            if MODEL in {item["id"] for item in from_http(f"http://127.0.0.1:{local}/v1/models")["data"]}:
                return int(sim.forward("-R", f"127.0.0.1:0:127.0.0.1:{local}"))
        except (ConnectionError, URLError, TimeoutError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError(f"Qwen endpoint did not become ready on A800 loopback port {port}.")


def stage_bundle(sim, root, manifest):
    if "remote_root" not in manifest:
        remote = sim.run("mktemp", "-d", manifest["settings"]["runtime"] + "/mac-tron2.XXXXXXXX", capture_output=True, text=True)
        manifest["remote_root"] = remote.stdout.strip()
        save_manifest(root, manifest)
    remote = manifest["remote_root"]
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for path in sorted(BUNDLE.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    tar.add(path, arcname=str(Path("bundle") / path.relative_to(BUNDLE)))
            tar.add(HERE / "dashboard_instrumentation.py", arcname="bundle/dashboard_instrumentation.py")
            # Resolve scripts.run_tron2_model from this batch's bundle before the shared checkout.
            tar.addfile(tarfile.TarInfo("bundle/scripts/__init__.py"))
            stock_runner = TRON2_REPO / "scripts" / "run_tron2_model.py"
            tar.add(stock_runner, arcname="bundle/scripts/run_tron2_model.py")
        archive.seek(0)
        sim.run("tar", "-xf", "-", "-C", remote, stdin=archive)


def record_event(attempt, index, event):
    if event["type"] == "frame":
        image = base64.b64decode(event["jpeg"], validate=True)
        if image.startswith(b"\xff\xd8") and image.endswith(b"\xff\xd9"):
            save_bytes(attempt / "latest.jpg", image)
        return index
    payload = json.dumps(event, ensure_ascii=False).encode()
    save_bytes(attempt / "events" / f"{index:08d}.json", payload)
    if event["type"] in ("request", "reply"):
        print(f"  {event.get('model')} {event['type']} {event.get('status', '')}".rstrip(), flush=True)
    return index + 1


def download(sim, remote, attempt):
    check = subprocess.run(sim.command(["test", "-d", remote]), capture_output=True)
    if check.returncode == 1:
        return
    check.check_returncode()
    with tempfile.TemporaryFile() as archive:
        sim.run("tar", "-cf", "-", "-C", remote, ".", stdout=archive)
        archive.seek(0)
        with tempfile.TemporaryDirectory(dir=attempt) as directory:
            staging = Path(directory)
            with tarfile.open(fileobj=archive, mode="r:") as tar:
                tar.extractall(staging, filter="data")
            data = staging / "data"
            if data.is_dir():
                # Rename complete artifacts so the dashboard reads only stable files.
                for path in ("controller.log", "joystick.log", "tabletop.log", "evaluation"):
                    source = data / path
                    target = attempt / path
                    if source.is_dir() and target.is_dir():
                        shutil.copytree(source, target, dirs_exist_ok=True)
                    elif source.exists():
                        source.replace(target)
            for name in ("launcher.log", "launch_status.json", "dashboard-events.log"):
                if (staging / name).is_file():
                    (staging / name).replace(attempt / name)


def completed(attempt):
    try:
        status = read_json(attempt / "launch_status.json")
        result = read_json(attempt / "evaluation" / "results.json")
        row, = result["trials"]
        if status["exit_code"] or row["error"] or row["video_error"] or row["video_frames"] < 1:
            return None
        path = (attempt / "evaluation" / row["video"]).resolve()
        if path.is_relative_to(attempt.resolve()) and path.is_file() and path.stat().st_size > 0:
            return row
    except (OSError, TypeError, KeyError, ValueError):
        pass
    return None


def run_episode(sim, root, manifest, episode, model_port, vdm_port, token, gpu, *, reuse_verified_vision):
    old = episode.get("active_attempt")
    if old:
        remote_old = PurePosixPath(manifest["remote_root"]) / old
        download(sim, remote_old, root / old)
        if completed(root / old):
            episode["state"] = "complete"
            save_manifest(root, manifest)
            return ()

    attempts = root / "episodes" / episode["id"]
    attempts.mkdir(parents=True, exist_ok=True)
    index = max((int(path.name.split("-")[1]) for path in attempts.glob("attempt-[0-9]*") if path.is_dir()), default=0) + 1
    attempt = attempts / f"attempt-{index:03d}"
    (attempt / "events").mkdir(parents=True)
    relative = attempt.relative_to(root).as_posix()
    episode.update(state="running", active_attempt=relative)
    save_manifest(root, manifest)
    settings = manifest["settings"]
    remote = PurePosixPath(manifest["remote_root"]) / relative
    entry = PurePosixPath(manifest["remote_root"]) / "bundle" / "remote_episode.py"
    command = [
        "python3", entry, "--attempt", remote, "--repo", settings["repo"], "--vendor", settings["vendor"],
        "--runtime", settings["runtime"], "--variant", episode["variant"], "--tier", episode["tier"],
        "--seed", episode["seed"], "--blocks", settings["blocks"], "--jitter", settings["jitter"],
        "--model", MODEL, "--model-port", model_port, "--vdm-port", vdm_port, "--gpu", gpu,
    ]
    if reuse_verified_vision:
        command.append("--reuse-verified-vision")
    print(f"{episode['id']} · attempt {index}", flush=True)
    process = subprocess.Popen(sim.command(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               text=True, start_new_session=True)
    interrupted = False
    count = 0
    try:
        process.stdin.write(token + "\n")
        process.stdin.flush()
        for line in process.stdout:
            if line.startswith(EVENT_PREFIX):
                count = record_event(attempt, count, json.loads(line[len(EVENT_PREFIX):]))
            else:
                print(line, end="", flush=True)
        process.wait()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # EOF tells the remote supervisor to stop its owned namespace and flush status.
        process.stdin.close()
        process.stdin = None
        try:
            remaining, _ = process.communicate(timeout=12)
            for line in remaining.splitlines():
                if line.startswith(EVENT_PREFIX):
                    count = record_event(attempt, count, json.loads(line[len(EVENT_PREFIX):]))
        except subprocess.TimeoutExpired:
            stop_local(process)
        episode["state"] = "interrupted" if interrupted else "error"
        save_manifest(root, manifest)
        download(sim, remote, attempt)
        if completed(attempt):
            episode["state"] = "complete"
        save_manifest(root, manifest)
    if interrupted:
        raise KeyboardInterrupt
    try:
        return tuple(read_json(attempt / "evaluation" / "results.json")["verified_vision_endpoints"])
    except (OSError, TypeError, KeyError, ValueError):
        return ()


def write_report(root, manifest):
    lines = ["# TRON2 批次结果", "", "| Episode | 状态 | 抬升 | 最终抬升 cm | 桌面峰值 N | 录像 |", "| --- | --- | --- | ---: | ---: | --- |"]
    for episode in manifest["episodes"]:
        active = episode.get("active_attempt")
        row = completed(root / active) if active and episode["state"] == "complete" else None
        if row:
            video = f"{active}/evaluation/{row['video']}"
            lines.append(f"| {episode['id']} | complete | {'是' if row['task_completed'] else '否'} | {row['final_lift_cm']:.2f} | {row['table_peak_normal_n']:.2f} | [录像]({video}) |")
        else:
            lines.append(f"| {episode['id']} | {episode['state']} | — | — | — | — |")
    save_bytes(root / "recordings.md", ("\n".join(lines) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    identifier = parser.add_mutually_exclusive_group()
    identifier.add_argument("--run-id")
    identifier.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-code-blocks", type=int)
    parser.add_argument("--sim-gpu", default="auto")
    parser.add_argument("--a800-gpu", default="auto")
    parser.add_argument("--a800-port", type=int, help="Attach to a running A800 loopback endpoint instead of starting one.")
    parser.add_argument("--dashboard-port", type=int, default=8120)
    parser.add_argument("--a800-host", default="A800")
    parser.add_argument("--sim-host", default="4090w")
    parser.add_argument("--sim-base", type=PurePosixPath, default=os.environ.get("CAPX_TRON2_SIM_BASE") or None,
                        help="Absolute simulation host workspace path (or CAPX_TRON2_SIM_BASE).")
    parser.add_argument("--a800-base", type=PurePosixPath, default=os.environ.get("CAPX_A800_BASE") or None,
                        help="Absolute inference host workspace path (or CAPX_A800_BASE). Defaults to its home directory.")
    args = parser.parse_args()
    budgets = args.max_code_blocks if args.max_code_blocks is not None else 4
    seeds = args.seeds if args.seeds is not None else [4]
    if budgets < 1 or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error("Use positive code block budgets and distinct nonnegative seeds.")
    if args.resume and any(value is not None for value in (args.conditions, args.seeds, args.max_code_blocks)):
        parser.error("A resumed run uses its saved conditions, seeds, and code budget.")
    for value in (args.sim_gpu, args.a800_gpu):
        if value != "auto" and not value.isdecimal():
            parser.error("GPU values must be auto or a nonnegative integer.")
    for port in (args.dashboard_port, args.a800_port):
        if port is not None and not 1 <= port <= 65535:
            parser.error("Ports must be between 1 and 65535.")
    for host in (args.sim_host, args.a800_host):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
            parser.error("Hosts must be SSH aliases or user@host.")
    for option, path in (("--sim-base", args.sim_base), ("--a800-base", args.a800_base)):
        if path is not None and not path.is_absolute():
            parser.error(f"{option} must be an absolute path on its remote host.")
    if args.resume is None and args.sim_base is None:
        parser.error("Set --sim-base or CAPX_TRON2_SIM_BASE for a new run.")
    identifier = args.resume or args.run_id or (datetime.now().strftime("tron2_%Y%m%d_%H%M%S_") + secrets.token_hex(3))
    if not RUN_ID.fullmatch(identifier):
        parser.error("Use a run ID containing letters, numbers, dashes or underscores.")
    root = RUNS / identifier
    if not args.resume:
        root.mkdir(mode=0o700, parents=True)
        base = args.sim_base
        manifest = {"run_id": identifier, "settings": {
            "repo": str(base / "tron2-sim"), "vendor": str(base / "tron2_manipulation_controller"),
            "runtime": str(base / "tron2-route2-runtime"), "blocks": budgets, "jitter": 0.02,
            "sim_host": args.sim_host, "a800_host": args.a800_host,
            "a800_base": str(args.a800_base) if args.a800_base is not None else None,
        }, "episodes": []}
        for condition in dict.fromkeys(args.conditions or ["v1:M3"]):
            variant, tier = condition.split(":")
            for seed in seeds:
                manifest["episodes"].append({
                    "id": f"{variant}-{tier}-seed-{seed:03d}", "tier": tier, "variant": variant,
                    "seed": seed, "state": "pending", "active_attempt": None,
                })
        save_manifest(root, manifest)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Run {identifier} is already active.") from error
        manifest = read_json(root / "manifest.json")
        if args.a800_base is None and (saved_base := manifest["settings"].get("a800_base")):
            args.a800_base = PurePosixPath(saved_base)
        run(root, manifest, args)
    return int(any(episode["state"] != "complete" for episode in manifest["episodes"]))


def run(root, manifest, args):
    episodes = manifest["episodes"]
    pending = []
    for episode in episodes:
        if episode.get("active_attempt") and completed(root / episode["active_attempt"]):
            episode["state"] = "complete"
        else:
            pending.append(episode)
    save_manifest(root, manifest)
    seen = pending[0] if pending else episodes[0]
    dashboard(root, args.dashboard_port, seen)
    print(f"Run {root.name}\nDashboard {episode_url(args.dashboard_port, root, seen)}", flush=True)
    if not pending:
        write_report(root, manifest)
        print("All episodes have archived results and videos. No remote work started.", flush=True)
        return
    try:
        cli = find_vdm_cli() if any(row["tier"] == "M3" for row in pending) else None
        with ExitStack() as stack:
            control = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="capx-tron2-", dir="/tmp")))
            sim = stack.enter_context(SSH(manifest["settings"]["sim_host"], control / "sim"))
            stage_bundle(sim, root, manifest)
            vdm_port, token = open_vdm(stack, sim, root, cli) if cli else (0, "")
            a800 = stack.enter_context(SSH(manifest["settings"]["a800_host"], control / "a800"))
            model_port = open_model(stack, sim, a800, args, root)
            vision_endpoints = {
                "M2": ((model_port, MODEL), "coding"),
                "M3": ((vdm_port, VDM_MODEL), "visual_differencing"),
            }
            verified_vision = set()
            for episode in pending:
                endpoint, role = vision_endpoints.get(episode["tier"], (None, None))
                roles = run_episode(sim, root, manifest, episode, model_port, vdm_port, token, args.sim_gpu,
                                    reuse_verified_vision=endpoint in verified_vision)
                if endpoint is not None and role in roles:
                    verified_vision.add(endpoint)
                write_report(root, manifest)
                print(f"  {episode['state']} · {episode_url(args.dashboard_port, root, episode)}", flush=True)
    finally:
        write_report(root, manifest)
        print(f"Results {root / 'recordings.md'}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Run again with --resume and the same run ID.", file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
