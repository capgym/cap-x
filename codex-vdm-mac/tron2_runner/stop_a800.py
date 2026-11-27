"""Stop and verify release of a vLLM service owned by one experiment."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def command(*args):
    return subprocess.check_output(args, text=True, timeout=8).strip()


def live(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def owned(pid, marker):
    try:
        return marker in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def terminate(pid, marker):
    if not owned(pid, marker):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def owned_compute_pids(gpu, marker):
    output = command("nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    return {int(line) for line in output.splitlines() if line.isdigit() and owned(int(line), marker)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run_dir", type=Path)
    cli.add_argument("pid", type=positive)
    cli.add_argument("port", type=positive)
    cli.add_argument("gpu", type=int)
    cli.add_argument("--timeout", type=positive, default=60)
    args = cli.parse_args()
    if args.pid < 2 or args.port > 65535 or args.gpu < 0:
        cli.error("Invalid service PID, port or GPU")

    run = args.run_dir.resolve(strict=True)
    recorded_pid = int((run / "server.pid").read_text().strip())
    if recorded_pid != args.pid:
        raise RuntimeError(f"PID does not match {run / 'server.pid'}")
    marker = b"CAPX_QWEN27B_OWNER=" + os.fsencode(run)
    if live(args.pid) and not owned(args.pid, marker):
        raise RuntimeError(f"PID {args.pid} does not belong to {run}")

    terminate(args.pid, marker)
    signaled_children = set()
    start = time.monotonic()
    while time.monotonic() - start < args.timeout:
        listener = command("ss", "-H", "-ltn", f"sport = :{args.port}")
        children = owned_compute_pids(args.gpu, marker)
        parent_alive = live(args.pid) and owned(args.pid, marker)
        if not listener and not children and not parent_alive:
            print(f"STOPPED RUN_DIR={run} PID={args.pid} PORT={args.port} GPU={args.gpu}")
            return
        if not parent_alive and time.monotonic() - start >= 4:
            for child in children - signaled_children:
                terminate(child, marker)
                signaled_children.add(child)
        time.sleep(1)
    raise RuntimeError(f"Timed out verifying port {args.port} and owned GPU {args.gpu} process release")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
