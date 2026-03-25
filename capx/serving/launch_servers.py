"""Unified server launcher for CaP-X.

Start all API servers needed for an environment config with one command:

    uv run --no-sync --active capx/serving/launch_servers.py \
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml

Or start a predefined server profile:

    uv run --no-sync --active capx/serving/launch_servers.py --profile default
    uv run --no-sync --active capx/serving/launch_servers.py --profile full
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launch_servers")

# ---------------------------------------------------------------------------
# Server Registry
# ---------------------------------------------------------------------------

# Maps a short server name to its launch module and resource requirements.
# The "target" field corresponds to the ``_target_`` values found in YAML
# configs (minus the trailing ``.main``).
SERVER_REGISTRY: dict[str, dict[str, Any]] = {
    "sam3": {
        "target": "capx.serving.launch_sam3_server",
        "default_port": 8114,
        "gpu_required": True,
        "gpu_memory_mb": 3000,
        "extra_args": {"device": "cuda"},
    },
    "graspnet": {
        "target": "capx.serving.launch_contact_graspnet_server",
        "default_port": 8115,
        "gpu_required": True,
        "gpu_memory_mb": 2000,
        "extra_args": {"device": "cuda"},
    },
    "pyroki": {
        "target": "capx.serving.launch_pyroki_server",
        "default_port": 8116,
        "gpu_required": False,
        "gpu_memory_mb": 0,
        "extra_args": {},
    },
    "sam2": {
        "target": "capx.serving.launch_sam2_server",
        "default_port": 8113,
        "gpu_required": True,
        "gpu_memory_mb": 6000,
        "extra_args": {"device": "cuda"},
    },
    "owlvit": {
        "target": "capx.serving.launch_owlvit_server",
        "default_port": 8118,
        "gpu_required": True,
        "gpu_memory_mb": 3000,
        "extra_args": {"device": "cuda"},
    },
    "curobo": {
        "target": "capx.serving.launch_curobo_server",
        "default_port": 8117,
        "gpu_required": True,
        "gpu_memory_mb": 2000,
        "extra_args": {},
    },
}

# Reverse lookup: _target_ module string -> short name
_TARGET_TO_NAME: dict[str, str] = {
    info["target"]: name for name, info in SERVER_REGISTRY.items()
}

# ---------------------------------------------------------------------------
# Predefined Profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, list[dict[str, Any]]] = {
    "default": [
        {"server": "sam3", "port": 8114},
        {"server": "graspnet", "port": 8115},
        {"server": "pyroki", "port": 8116},
    ],
    "full": [
        {"server": "sam3", "port": 8114},
        {"server": "graspnet", "port": 8115},
        {"server": "pyroki", "port": 8116},
        {"server": "owlvit", "port": 8117},
        {"server": "sam2", "port": 8113},
    ],
    "minimal": [
        {"server": "pyroki", "port": 8116},
    ],
}

# ---------------------------------------------------------------------------
# GPU Detection & Allocation
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    index: int
    name: str = "unknown"
    memory_total_mb: int = 0
    memory_free_mb: int = 0


def detect_gpus() -> list[GpuInfo]:
    """Detect available GPUs via ``nvidia-smi``.

    Returns a list of :class:`GpuInfo` with memory stats. Falls back to a
    single entry ``GpuInfo(index=0)`` when ``nvidia-smi`` is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi returned non-zero; falling back to GPU 0")
            return [GpuInfo(index=0)]

        gpus: list[GpuInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=int(parts[2]),
                    memory_free_mb=int(parts[3]),
                )
            )
        if not gpus:
            return [GpuInfo(index=0)]
        return gpus

    except FileNotFoundError:
        logger.warning("nvidia-smi not found; assuming single GPU at index 0")
        return [GpuInfo(index=0)]
    except Exception as exc:
        logger.warning("GPU detection failed (%s); assuming single GPU at index 0", exc)
        return [GpuInfo(index=0)]


def allocate_gpus(
    servers: list[dict[str, Any]],
    gpus: list[GpuInfo],
) -> list[dict[str, Any]]:
    """Assign GPU indices to servers using greedy bin-packing.

    Sorts GPU-requiring servers by memory requirement (descending) and assigns
    each to the GPU with the most remaining free memory. CPU-only servers
    receive ``gpu_index=None``.

    Returns the *same* list with ``gpu_index`` populated in-place.
    """
    # Track remaining free memory per GPU
    free: dict[int, int] = {g.index: g.memory_free_mb for g in gpus}

    # Separate GPU vs CPU servers
    gpu_servers = [s for s in servers if SERVER_REGISTRY[s["server"]]["gpu_required"]]
    cpu_servers = [s for s in servers if not SERVER_REGISTRY[s["server"]]["gpu_required"]]

    # Sort GPU servers by memory requirement descending (largest first)
    gpu_servers.sort(
        key=lambda s: SERVER_REGISTRY[s["server"]]["gpu_memory_mb"],
        reverse=True,
    )

    for srv in gpu_servers:
        needed = SERVER_REGISTRY[srv["server"]]["gpu_memory_mb"]
        # Pick the GPU with the most free memory
        best_gpu = max(free, key=lambda idx: free[idx])
        if free[best_gpu] < needed:
            logger.warning(
                "GPU %d has only %d MB free but %s needs %d MB; assigning anyway",
                best_gpu,
                free[best_gpu],
                srv["server"],
                needed,
            )
        srv["gpu_index"] = best_gpu
        free[best_gpu] -= needed

    for srv in cpu_servers:
        srv["gpu_index"] = None

    return servers


# ---------------------------------------------------------------------------
# YAML Config Parsing
# ---------------------------------------------------------------------------


def parse_servers_from_yaml(config_path: str) -> list[dict[str, Any]]:
    """Extract server definitions from a CaP-X YAML config.

    The YAML ``api_servers`` key contains entries like::

        api_servers:
          - _target_: capx.serving.launch_sam3_server.main
            device: cuda
            port: 8114
            host: 127.0.0.1

    This function maps each ``_target_`` to the corresponding short name in
    :data:`SERVER_REGISTRY` and collects any extra keyword arguments from the
    YAML entry.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    api_servers_raw = cfg.get("api_servers")
    if not api_servers_raw:
        raise ValueError(f"No 'api_servers' key found in {config_path}")

    servers: list[dict[str, Any]] = []
    for entry in api_servers_raw:
        target = entry.get("_target_", "")
        # Strip trailing ".main" to get the module path
        module = target.removesuffix(".main")
        name = _TARGET_TO_NAME.get(module)
        if name is None:
            logger.warning(
                "Unknown server target '%s' in YAML; skipping. "
                "Known targets: %s",
                target,
                ", ".join(_TARGET_TO_NAME.keys()),
            )
            continue

        srv: dict[str, Any] = {
            "server": name,
            "port": entry.get("port", SERVER_REGISTRY[name]["default_port"]),
            "host": entry.get("host", "127.0.0.1"),
        }

        # Carry over extra args from the YAML (e.g. device, robot, target_link)
        extra: dict[str, Any] = {}
        skip_keys = {"_target_", "port", "host"}
        for k, v in entry.items():
            if k not in skip_keys:
                extra[k] = v
        if extra:
            srv["extra_args"] = extra

        servers.append(srv)

    if not servers:
        raise ValueError(f"No recognised servers in {config_path}")

    return servers


# ---------------------------------------------------------------------------
# Server Process Management
# ---------------------------------------------------------------------------


def _build_cmd(server_config: dict[str, Any], workers: int) -> list[str]:
    """Build the subprocess command list for a single server."""
    name = server_config["server"]
    reg = SERVER_REGISTRY[name]
    module = reg["target"]

    port = server_config.get("port", reg["default_port"])
    host = server_config.get("host", "127.0.0.1")

    cmd = [
        sys.executable,
        "-m",
        module,
        "--port",
        str(port),
        "--host",
        host,
    ]

    # Merge extra args: registry defaults overridden by YAML/config values
    merged_extra: dict[str, Any] = {}
    merged_extra.update(reg.get("extra_args", {}))
    merged_extra.update(server_config.get("extra_args", {}))

    # If the server is GPU-required and has a gpu_index, set device to cuda
    gpu_idx = server_config.get("gpu_index")
    if gpu_idx is not None and reg["gpu_required"]:
        # The device arg should be "cuda" (we control CUDA_VISIBLE_DEVICES
        # externally), but keep any explicit override from config.
        merged_extra.setdefault("device", "cuda")

    for k, v in merged_extra.items():
        # Convert Python values to CLI-style args: --key value
        arg_name = f"--{k.replace('_', '-')}"
        cmd.extend([arg_name, str(v)])

    return cmd


def start_server(
    server_config: dict[str, Any],
    workers: int = 1,
    log_dir: str | None = None,
) -> subprocess.Popen[str]:
    """Start a single server as a subprocess.

    Uses :func:`subprocess.Popen` with ``CUDA_VISIBLE_DEVICES`` set according
    to the GPU allocation.
    """
    name = server_config["server"]
    gpu_idx = server_config.get("gpu_index")

    env = os.environ.copy()
    if gpu_idx is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    else:
        # CPU-only server: hide all GPUs to avoid accidental usage
        if not SERVER_REGISTRY[name]["gpu_required"]:
            env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = _build_cmd(server_config, workers)

    # Log file handling
    stdout_target: int | Any = subprocess.PIPE
    stderr_target: int | Any = subprocess.STDOUT
    log_file_handle = None

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        port = server_config.get("port", SERVER_REGISTRY[name]["default_port"])
        log_file = log_path / f"{name}_{port}.log"
        log_file_handle = open(log_file, "w")  # noqa: SIM115
        stdout_target = log_file_handle
        stderr_target = subprocess.STDOUT
        logger.info("  Logging %s to %s", name, log_file)

    logger.info("  CMD: %s", " ".join(cmd))
    if gpu_idx is not None:
        logger.info("  CUDA_VISIBLE_DEVICES=%s", gpu_idx)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout_target,
        stderr=stderr_target,
        text=True,
    )

    # Attach log file handle so we can close it on shutdown
    proc._log_file_handle = log_file_handle  # type: ignore[attr-defined]
    return proc


def wait_for_ready(
    servers: list[dict[str, Any]],
    processes: list[tuple[dict[str, Any], subprocess.Popen[str]]],
    timeout: float = 120.0,
) -> bool:
    """Wait for all servers to accept TCP connections.

    Uses exponential backoff per server. Returns ``True`` if all servers
    became ready within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    pending: dict[str, dict[str, Any]] = {}
    for srv in servers:
        key = f"{srv['server']}:{srv.get('port', SERVER_REGISTRY[srv['server']]['default_port'])}"
        pending[key] = srv

    # Map server keys to their process for early failure detection
    proc_map: dict[str, subprocess.Popen[str]] = {}
    for srv, proc in processes:
        key = f"{srv['server']}:{srv.get('port', SERVER_REGISTRY[srv['server']]['default_port'])}"
        proc_map[key] = proc

    interval = 1.0
    max_interval = 5.0

    while pending and time.monotonic() < deadline:
        still_pending: dict[str, dict[str, Any]] = {}
        for key, srv in pending.items():
            # Check if process died
            proc = proc_map.get(key)
            if proc is not None and proc.poll() is not None:
                logger.error(
                    "  [FAILED] %s (port %s) - process exited with code %d",
                    srv["server"],
                    srv.get("port"),
                    proc.returncode,
                )
                # Don't keep waiting for a dead process
                continue

            host = srv.get("host", "127.0.0.1")
            port = srv.get("port", SERVER_REGISTRY[srv["server"]]["default_port"])
            if _tcp_check(host, port):
                logger.info("  [READY]  %s (port %s)", srv["server"], port)
            else:
                still_pending[key] = srv
        pending = still_pending
        if pending:
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
            interval = min(interval * 1.5, max_interval)

    if pending:
        for _key, srv in pending.items():
            logger.warning(
                "  [TIMEOUT] %s (port %s) did not become ready in %.0fs",
                srv["server"],
                srv.get("port"),
                timeout,
            )
        return False
    return True


def _tcp_check(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return ``True`` if a TCP connection to *host*:*port* succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_STATUS_SYMBOLS = {
    "starting": "...",
    "ready": "OK",
    "failed": "FAIL",
    "timeout": "TIMEOUT",
}


def print_allocation_table(servers: list[dict[str, Any]]) -> None:
    """Print a summary table of the planned server allocation."""
    header = f"{'Server':<12} {'Port':<7} {'GPU':<6} {'VRAM (MB)':<10}"
    sep = "-" * len(header)
    logger.info("Server allocation plan:")
    logger.info("  %s", header)
    logger.info("  %s", sep)
    for srv in servers:
        name = srv["server"]
        port = srv.get("port", SERVER_REGISTRY[name]["default_port"])
        gpu = srv.get("gpu_index")
        gpu_str = str(gpu) if gpu is not None else "CPU"
        vram = SERVER_REGISTRY[name]["gpu_memory_mb"]
        vram_str = str(vram) if vram > 0 else "-"
        logger.info("  %-12s %-7s %-6s %-10s", name, port, gpu_str, vram_str)
    logger.info("  %s", sep)


def print_status_table(
    processes: list[tuple[dict[str, Any], subprocess.Popen[str]]],
) -> None:
    """Print the final status of all server processes."""
    header = f"{'Server':<12} {'Port':<7} {'PID':<8} {'Status':<10}"
    sep = "-" * len(header)
    logger.info("Final server status:")
    logger.info("  %s", header)
    logger.info("  %s", sep)
    for srv, proc in processes:
        name = srv["server"]
        port = srv.get("port", SERVER_REGISTRY[name]["default_port"])
        pid = proc.pid
        rc = proc.poll()
        if rc is None:
            status = "running"
        elif rc == 0:
            status = "stopped"
        else:
            status = f"exit({rc})"
        logger.info("  %-12s %-7s %-8s %-10s", name, port, pid, status)
    logger.info("  %s", sep)


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------


@dataclass
class LaunchServersArgs:
    """Unified server launcher for CaP-X."""

    config_path: str | None = None
    """YAML config to read api_servers from."""

    profile: str | None = None
    """Use a predefined server profile: default, full, minimal."""

    gpus: str | None = None
    """Comma-separated GPU indices to use (e.g. '0,1'). Default: auto-detect all."""

    workers: int = 1
    """Number of uvicorn workers per server (for throughput scaling)."""

    log_dir: str | None = None
    """Directory to write per-server log files. Default: print to stdout with prefix."""

    timeout: float = 120.0
    """Seconds to wait for all servers to become ready."""

    host: str = "127.0.0.1"
    """Default host for servers (overridden by YAML/profile values)."""

    dry_run: bool = False
    """Print the allocation plan and exit without starting servers."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_gpu_list(gpus_str: str) -> list[int]:
    """Parse a comma-separated list of GPU indices."""
    indices: list[int] = []
    for part in gpus_str.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def _resolve_servers(args: LaunchServersArgs) -> list[dict[str, Any]]:
    """Determine the list of servers to start from CLI arguments."""
    if args.config_path:
        servers = parse_servers_from_yaml(args.config_path)
    elif args.profile:
        if args.profile not in PROFILES:
            raise ValueError(
                f"Unknown profile '{args.profile}'. "
                f"Available: {', '.join(PROFILES.keys())}"
            )
        # Deep-copy the profile so we don't mutate the template
        servers = [dict(s) for s in PROFILES[args.profile]]
    else:
        logger.info("No --config-path or --profile specified; using 'default' profile")
        servers = [dict(s) for s in PROFILES["default"]]

    # Apply default host if not set
    for srv in servers:
        srv.setdefault("host", args.host)

    return servers


# ---------------------------------------------------------------------------
# Graceful Shutdown
# ---------------------------------------------------------------------------


def _shutdown(
    processes: list[tuple[dict[str, Any], subprocess.Popen[str]]],
    grace_period: float = 10.0,
) -> None:
    """Terminate all server processes, waiting up to *grace_period* seconds."""
    logger.info("Shutting down %d server(s)...", len(processes))

    # Send SIGTERM to all
    for srv, proc in processes:
        if proc.poll() is None:
            logger.info("  Terminating %s (PID %d)", srv["server"], proc.pid)
            proc.terminate()

    # Wait for graceful exit
    deadline = time.monotonic() + grace_period
    for srv, proc in processes:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            logger.warning(
                "  %s (PID %d) did not exit in time; sending SIGKILL",
                srv["server"],
                proc.pid,
            )
            proc.kill()
            proc.wait(timeout=5)

    # Close any open log file handles
    for _srv, proc in processes:
        handle = getattr(proc, "_log_file_handle", None)
        if handle is not None:
            handle.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: LaunchServersArgs) -> None:
    # 1. Determine which servers to start
    servers = _resolve_servers(args)

    # 2. Detect GPUs and optionally filter
    gpus = detect_gpus()
    if args.gpus:
        allowed = set(_parse_gpu_list(args.gpus))
        gpus = [g for g in gpus if g.index in allowed]
        if not gpus:
            logger.error(
                "No GPUs remaining after filtering with --gpus=%s", args.gpus
            )
            sys.exit(1)

    if gpus:
        gpu_summary = ", ".join(
            f"GPU {g.index}: {g.name} ({g.memory_free_mb}/{g.memory_total_mb} MB free)"
            for g in gpus
        )
        logger.info("Detected GPUs: %s", gpu_summary)

    # 3. Allocate GPUs
    servers = allocate_gpus(servers, gpus)

    # 4. Print allocation plan
    print_allocation_table(servers)

    if args.dry_run:
        logger.info("Dry run; exiting without starting servers.")
        return

    # 5. Start all servers
    processes: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
    for srv in servers:
        logger.info("Starting %s on port %s...", srv["server"], srv.get("port"))
        try:
            proc = start_server(srv, workers=args.workers, log_dir=args.log_dir)
            processes.append((srv, proc))
        except Exception:
            logger.exception("Failed to start %s", srv["server"])
            _shutdown(processes)
            sys.exit(1)

    logger.info("All %d server(s) launched. Waiting for readiness...", len(processes))

    # 6. Wait for readiness
    all_ready = wait_for_ready(servers, processes, timeout=args.timeout)
    if all_ready:
        logger.info("All servers are ready.")
    else:
        logger.warning("Some servers did not become ready within %.0fs.", args.timeout)

    # 7. Block until Ctrl-C or SIGTERM, then graceful shutdown
    shutdown_requested = False

    def _signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            # Second signal: force kill
            logger.warning("Received second signal; forcing shutdown.")
            for _srv, proc in processes:
                if proc.poll() is None:
                    proc.kill()
            sys.exit(1)
        shutdown_requested = True
        sig_name = signal.Signals(signum).name
        logger.info("Received %s; initiating shutdown...", sig_name)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("Press Ctrl-C to stop all servers.")

    try:
        # Poll processes to detect unexpected exits
        while not shutdown_requested:
            all_dead = True
            for srv, proc in processes:
                rc = proc.poll()
                if rc is None:
                    all_dead = False
                elif rc != 0 and not getattr(srv, "_reported_exit", False):
                    logger.error(
                        "%s (PID %d) exited unexpectedly with code %d",
                        srv["server"],
                        proc.pid,
                        rc,
                    )
                    srv["_reported_exit"] = True  # type: ignore[typeddict-unknown-key]
            if all_dead:
                logger.error("All server processes have exited.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(processes)
        print_status_table(processes)

    logger.info("All servers stopped.")


if __name__ == "__main__":
    main(tyro.cli(LaunchServersArgs))
