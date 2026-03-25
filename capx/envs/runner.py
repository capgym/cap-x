"""Trial orchestration for CaP-X environments.

Handles batch execution, parallel worker dispatch, retry logic, and
wall-clock timeouts for running code-generation trials.  The actual
single-trial execution lives in :mod:`capx.envs.trial`.
"""

from __future__ import annotations

import functools
import os
import signal
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trial import (
    _annotate_code_blocks,
    _build_log_lines,
    _run_single_trial,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _print_and_save_summary,
    _save_trial_artifacts,
    run_server_proc,
)
from capx.utils.parallel_eval import run_parallel_with_setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIAL_TIMEOUT_SECONDS = 1000
MAX_TRIAL_RETRIES = 3


# ---------------------------------------------------------------------------
# API server helpers
# ---------------------------------------------------------------------------

def _start_api_servers(
    api_servers: list | None, wait_timeout: float = 120.0
) -> list:
    """Launch any API server sub-processes defined in the config.

    Skips servers whose port is already in use (e.g. started externally).
    After launching, waits until all servers are accepting connections.
    """
    import socket
    import time

    procs = []
    ports_to_wait: list[tuple[str, int]] = []
    if api_servers is not None:
        for api_server in api_servers:
            port = api_server.get("port")
            host = api_server.get("host", "127.0.0.1")
            if port is not None:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((host, int(port))) == 0:
                        print(f"API server on {host}:{port} already running, skipping")
                        continue
            proc = run_server_proc(api_server)
            procs.append(proc)
            print(f"API server {api_server} started")
            if port is not None:
                ports_to_wait.append((host, int(port)))

    # Wait for all launched servers to accept connections
    if ports_to_wait:
        deadline = time.time() + wait_timeout
        for host, port in ports_to_wait:
            while time.time() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((host, port)) == 0:
                        print(f"API server on {host}:{port} is ready")
                        break
                time.sleep(1.0)
            else:
                print(f"WARNING: API server on {host}:{port} not ready after {wait_timeout}s")

    return procs


def _stop_api_servers(server_procs: list) -> None:
    """Terminate API server sub-processes."""
    for proc in server_procs:
        proc.terminate()
        proc.join()


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def _setup_output_dir(args, config: dict[str, Any]) -> None:
    """Create and normalize the output directory path.

    Inserts the model name into the output path so that results from
    different models are stored separately.
    """
    if args.use_oracle_code:
        args.model = "oracle"
    if config["output_dir"]:
        parts = config["output_dir"].split("/")
        parts.insert(-1, str(args.model).replace("/", "_"))
        new_out_dir = "/".join(parts)
        Path(new_out_dir).mkdir(parents=True, exist_ok=True)
        config["output_dir"] = new_out_dir


# ---------------------------------------------------------------------------
# Headless trial runner
# ---------------------------------------------------------------------------

def _run_headless_trials(
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
    start_time: float,
) -> None:
    """Run all trials in headless CLI mode, then print a summary.

    Dispatches to parallel or sequential execution depending on ``num_workers``.
    """
    if config["total_trials"] <= 0:
        print("No trials requested; exiting.")
        return

    if config["record_video"] and not config["output_dir"]:
        raise ValueError("record_video requires --output-dir to save generated videos")

    _setup_output_dir(args, config)

    # Determine which trials to run (supports resume)
    trial_ids = list(range(1, config["total_trials"] + 1))
    if config.get("resume_idx") is not None:
        trial_ids = list(range(config["resume_idx"], config["total_trials"] + 1))

    # Execute trials
    if config["num_workers"] > 1:
        setup_fn = functools.partial(_worker_setup, env_factory=env_factory)
        trial_fn = functools.partial(_run_single_trial_worker, args=args, config=config)
        summaries = run_parallel_with_setup(
            trial_ids,
            num_workers=config["num_workers"],
            setup_fn=setup_fn,
            trial_fn=trial_fn,
        )
    else:
        summaries = _run_trial_batch(trial_ids, args=args, env_factory=env_factory, config=config)

    summaries.sort(key=lambda s: s.trial)
    _print_and_save_summary(summaries, args, config, start_time)

    # Write completion flag
    os.makedirs(os.path.join(config["output_dir"], "aaa_done_flag"), exist_ok=True)
    with open(os.path.join(config["output_dir"], "aaa_done_flag", "aaa_done_flag.txt"), "w") as f:
        f.write("1")


# ---------------------------------------------------------------------------
# Worker setup and trial dispatch
# ---------------------------------------------------------------------------

def _worker_setup(*, env_factory: dict[str, Any]) -> tuple[Any, str | None]:
    """Create the environment once per parallel worker.

    Returns:
        Tuple of (environment, multi_turn_prompt).
    """
    env = instantiate(env_factory)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)
    return (env, multi_turn_prompt)


def _run_single_trial_worker(
    state: tuple[Any, str | None],
    trial: int,
    *,
    args,
    config: dict[str, Any],
) -> TrialSummary:
    """Run a single trial using pre-initialized worker state (for parallel mode)."""
    env, multi_turn_prompt = state
    return _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)


def _run_trial_with_retries(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
) -> TrialSummary:
    """Attempt a trial up to MAX_TRIAL_RETRIES times, retrying on timeout."""
    for attempt in range(MAX_TRIAL_RETRIES):
        try:
            is_last_attempt = attempt == MAX_TRIAL_RETRIES - 1
            return _run_single_trial_with_timeout(
                env=env,
                trial=trial,
                args=args,
                config=config,
                multi_turn_prompt=multi_turn_prompt,
                timeout_s=TRIAL_TIMEOUT_SECONDS,
                raise_on_timeout=not is_last_attempt,
            )
        except TimeoutError:
            print(f"Trial {trial} timed out (attempt {attempt + 1}/{MAX_TRIAL_RETRIES}). Retrying...")

    # All retries exhausted
    return TrialSummary(
        trial=trial,
        success=False,
        reward=0.0,
        terminated=False,
        truncated=True,
        sandbox_rc=1,
        log=f"Trial {trial} failed after {MAX_TRIAL_RETRIES} timeout retries",
        task_completed=False,
        code_path=None,
        num_regenerations=0,
        num_finishes=0,
        num_code_blocks=0,
    )


def _run_trial_batch(
    trial_ids: Iterable[int],
    *,
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
) -> list[TrialSummary]:
    """Run a batch of trials sequentially (single-worker mode).

    Creates one environment instance and reuses it for all trials.
    """
    trial_indices = list(trial_ids)
    if not trial_indices:
        return []

    # OmniGibson / Isaac Sim reads sys.argv and can crash on unknown flags.
    # Strip our CLI args while instantiating the env.
    original_sys_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(env_factory)
    finally:
        sys.argv = original_sys_argv

    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)

    summaries: list[TrialSummary] = []
    for trial in tqdm(trial_indices, desc="Running Trials"):
        summary = _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)
        summaries.append(summary)

    summaries.sort(key=lambda s: s.trial)
    return summaries


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------

def _run_single_trial_with_timeout(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    timeout_s: float,
    raise_on_timeout: bool = False,
) -> TrialSummary:
    """Run a single trial with a wall-clock timeout via SIGALRM."""
    timeout_seconds = max(1, int(timeout_s))
    timed_out = False

    def _timeout_handler(signum: int, frame) -> None:  # type: ignore[override]
        nonlocal timed_out
        timed_out = True
        raise TimeoutError(f"Trial {trial} exceeded {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    partial_artifacts: dict[str, Any] = {}
    try:
        return _run_single_trial(
            env, trial, args, config, multi_turn_prompt, partial_artifacts=partial_artifacts
        )
    except BaseException as exc:
        is_timeout = timed_out or isinstance(exc, TimeoutError)
        try:
            import requests as _requests
            is_timeout = is_timeout or isinstance(exc, _requests.exceptions.Timeout)
        except Exception:
            pass

        if not is_timeout:
            raise
        if raise_on_timeout:
            raise TimeoutError(f"Trial {trial} timed out") from exc

        print(f"Trial {trial} timed out after {timeout_seconds} seconds")
        return _build_timeout_summary(trial, timeout_seconds, partial_artifacts, config, exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_timeout_summary(
    trial: int,
    timeout_seconds: int,
    pa: dict[str, Any],
    config: dict[str, Any],
    exc: BaseException,
) -> TrialSummary:
    """Build a TrialSummary from partial artifacts after a timeout."""
    raw_code = pa.get("raw_code", "")
    code_blocks = pa.get("code_blocks", [])
    code_block_metadata = pa.get("code_block_metadata", [])
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)

    info_step = pa.get("info_step", {"sandbox_rc": 1, "stdout": "", "stderr": str(exc)})
    if info_step.get("stderr") == "":
        info_step["stderr"] = str(exc)
    else:
        info_step["stderr"] += f"\n\nTimeout Error: {exc}"

    reward = pa.get("reward", 0.0)
    terminated = pa.get("terminated", False)
    truncated = pa.get("truncated", False)
    num_regenerations = pa.get("num_regenerations", 0)
    num_finishes = pa.get("num_finishes", 0)
    num_code_blocks = pa.get("num_code_blocks", len(code_blocks))

    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        prefix=f"Trial {trial} timed out after {timeout_seconds} seconds.",
    )

    code_path = _save_trial_artifacts(
        config, trial,
        sandbox_rc=1,
        reward=reward,
        task_completed=info_step.get("task_completed", False),
        final_code=final_code,
        raw_code=raw_code,
        all_responses=pa.get("all_responses", []),
        log_lines=log_lines,
        visual_feedback_imgs=pa.get("visual_feedback_imgs", []),
        ensemble_data=pa.get("ensemble_data"),
        multiturn_ensemble_data=pa.get("multiturn_ensemble_data", []),
    )

    return TrialSummary(
        trial=trial,
        success=False,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=1,
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", False),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )