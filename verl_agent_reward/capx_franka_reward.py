from __future__ import annotations

import os

# Ensure headless MuJoCo rendering in Ray workers
os.environ.setdefault("MUJOCO_GL", "osmesa")

import ast
import hashlib
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from capx.envs.tasks import get_config, get_exec_env
from capx.security import validate_generated_program

initialized_envs = {}

_ROBOT_API_NAMES = {
    "close_gripper",
    "compose_pose",
    "get_active_nut_types",
    "get_object_pose",
    "goto_home_joint_position",
    "goto_pose",
    "open_gripper",
    "relative_pose",
    "sample_grasp_pose",
}


def _extract_code(content: str) -> str:
    # blocks = []
    fence_start = "```python\n"
    fence_end = "```"
    start_idx = content.find(fence_start)
    end_idx = content.rfind(fence_end)
    if start_idx == -1 or end_idx == -1:
        return content
    content = content[start_idx + len(fence_start) : end_idx]
    return content.strip()


def _api_structure_bonus(source: str) -> float:
    """Give a small curriculum signal for executable use of documented APIs."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0.0
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _ROBOT_API_NAMES
    }
    return min(len(calls) * 0.005, 0.025)


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Raise TimeoutError if the with-block exceeds the given seconds.

    Uses POSIX timers; only effective in the main thread on Unix-like systems.
    """
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:  # type: ignore[override]
        raise TimeoutError(f"Environment step exceeded {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        # Cancel timer and restore previous handler
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | str,
    extra_info: dict[str, Any] | None = None,
) -> float | dict[str, Any]:
    """Compute reward for CaP-X Franka pick-and-place programs.

    This function is intended to be loaded by VeRL via
    custom_reward_function.path/name. It runs the candidate program inside
    the Franka environment and returns the obtained reward.

    Args:
        data_source: Dataset identifier; should match CaP-X Franka sources.
        solution_str: Generated program text to execute in the env.
        ground_truth: Unused here; may contain a reference program.
        extra_info: Metadata dict; expects a 'seed' used to reset the env.

    Returns:
        A float reward or a dict containing at least a 'score' key.
    """
    # need to make env persistent
    if data_source not in initialized_envs:
        cfg = get_config(data_source)
        if cfg.privileged:
            cfg.enable_render = False
        initialized_envs[data_source] = get_exec_env(data_source)(cfg)
    env = initialized_envs[data_source]
    extra: dict[str, Any] = extra_info or {}
    seed: int = int(extra.get("seed", 0))

    solution_str = _extract_code(solution_str)
    program_sha256 = hashlib.sha256(solution_str.encode()).hexdigest()
    print("Solution code: ", solution_str)
    guard = validate_generated_program(solution_str)
    if not guard.allowed:
        return {
            "score": 0.0,
            "won": False,
            "terminated": False,
            "truncated": False,
            "error": guard.reason,
            "program_sha256": program_sha256,
        }
    try:
        env.reset(seed=seed)
        # Execute the whole program in one env step; the env internally
        # interprets the program string and returns a terminal reward.
        start_time = time.time()
        with _time_limit(90.0):
            _, reward, terminated, truncated, info = env.step(solution_str)
        # print(f"Environment step time: {time.time() - start_time:.4f} seconds")
        endtime = time.time()
        print("Time taken for eval: ", endtime - start_time)
        environment_reward = float(max(reward, 0.0))
        structure_bonus = _api_structure_bonus(solution_str) if info["sandbox_rc"] == 0 else 0.0
        score = max(environment_reward, structure_bonus)
        print("Score: ", score)
        # Always return a consistent schema so reward_extra_info lists align with batch size.
        # If the episode did not terminate, we still return the immediate reward—VeRL will
        # place it on the last token.
        task_completed = bool(info.get("task_completed", False))
        return {
            "score": score,
            "won": task_completed,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "error": "",
            "environment_reward": environment_reward,
            "structure_bonus": structure_bonus,
            "program_sha256": program_sha256,
        }
    except TimeoutError as e:
        return {
            "score": 0.0,
            "won": False,
            "terminated": False,
            "truncated": True,
            "error": repr(e),
            "program_sha256": program_sha256,
        }
    except Exception as e:  # noqa: BLE001
        # Fail-safe: return zero with a consistent schema; include error string for inspection.
        return {
            "score": 0.0,
            "won": False,
            "terminated": False,
            "truncated": False,
            "error": repr(e),
            "program_sha256": program_sha256,
        }
