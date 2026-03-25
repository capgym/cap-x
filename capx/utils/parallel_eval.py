from __future__ import annotations

import math
import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

BatchResult = TypeVar("BatchResult")


def chunk_into_batches(items: Sequence[int], batch_size: int) -> list[list[int]]:
    """Split ``items`` into ``batch_size`` chunks."""

    if batch_size <= 0:
        return [list(items)]
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def run_parallel_batches(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    batch_fn: Callable[[Sequence[int]], Iterable[BatchResult]],
    batch_size: int | None = None,
    mp_start_method: str = "spawn",
) -> list[BatchResult]:
    """Execute ``batch_fn`` across ``trial_ids`` with optional multiprocessing.

    Args:
        trial_ids: Ordered trial identifiers to evaluate.
        num_workers: Number of worker processes. Values <= 1 run sequentially.
        batch_fn: Callable that consumes a batch of trial ids and returns an iterable
            of results. Must be picklable when ``num_workers > 1``.
        batch_size: Optional explicit batch size. Defaults to an even split across
            ``num_workers``.
        mp_start_method: Multiprocessing start method (defaults to ``"spawn"`` for
            compatibility with CUDA + Mujoco setups).

    Returns:
        List of aggregated ``batch_fn`` results (concatenated).
    """

    if not trial_ids:
        return []

    if num_workers <= 1 or len(trial_ids) == 1:
        return list(batch_fn(trial_ids))

    if batch_size is None:
        batch_size = max(1, math.ceil(len(trial_ids) / num_workers))

    batches = chunk_into_batches(trial_ids, batch_size)
    ctx = multiprocessing.get_context(mp_start_method)

    results: list[BatchResult] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [executor.submit(batch_fn, batch) for batch in batches]
        # Collect all futures, even if some fail
        for future in as_completed(futures):
            # try:
            results.extend(future.result())
            # except Exception as e:
            #     print(f"Worker failed with exception: {e}")
            #     # Continue processing other workers instead of crashing
            #     continue

    return results


def _worker_loop(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    single_trial_fn: Callable[[int], BatchResult],
) -> None:
    """Worker loop that pulls trial IDs from a shared queue until exhausted.

    Args:
        worker_id: ID of this worker (for logging).
        task_queue: Queue of trial IDs to process.
        result_queue: Queue to put results into.
        single_trial_fn: Function that processes a single trial ID and returns a result.
    """
    while True:
        try:
            # Non-blocking get with timeout to allow clean shutdown
            trial_id = task_queue.get(timeout=1.0)
        except Exception:
            # Queue is empty or closed
            break

        if trial_id is None:
            # Sentinel value signals worker to exit
            break

        try:
            result = single_trial_fn(trial_id)
            result_queue.put(("success", trial_id, result))
        except Exception as e:
            result_queue.put(("error", trial_id, str(e)))


def _worker_loop_with_setup(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    setup_fn: Callable[[], Any],
    trial_fn: Callable[[Any, int], BatchResult],
) -> None:
    """Worker loop that initializes state once, then pulls trials from queue.

    This is more efficient than _worker_loop when setup is expensive (e.g.,
    creating simulation environments).

    Args:
        worker_id: ID of this worker (for logging).
        task_queue: Queue of trial IDs to process.
        result_queue: Queue to put results into.
        setup_fn: Function called once to create worker state (e.g., environment).
        trial_fn: Function that takes (state, trial_id) and returns a result.
    """
    # Initialize worker state once
    try:
        state = setup_fn()
    except Exception as e:
        # Report setup failure and exit
        result_queue.put(("setup_error", worker_id, str(e)))
        return

    while True:
        try:
            # Non-blocking get with timeout to allow clean shutdown
            trial_id = task_queue.get(timeout=1.0)
        except Exception:
            # Queue is empty or closed
            break

        if trial_id is None:
            # Sentinel value signals worker to exit
            break

        try:
            result = trial_fn(state, trial_id)
            result_queue.put(("success", trial_id, result))
        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            result_queue.put(("error", trial_id, error_msg))


def run_parallel_dynamic(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    single_trial_fn: Callable[[int], BatchResult],
    mp_start_method: str = "spawn",
) -> list[BatchResult]:
    """Execute trials dynamically across workers using a shared queue.

    Unlike ``run_parallel_batches`` which pre-divides trials, this function
    uses a work-stealing pattern where workers pull trials from a shared queue
    as they become available. This provides better load balancing when trial
    execution times vary.

    Args:
        trial_ids: Ordered trial identifiers to evaluate.
        num_workers: Number of worker processes. Values <= 1 run sequentially.
        single_trial_fn: Callable that processes a single trial ID and returns
            a result. Must be picklable when ``num_workers > 1``.
        mp_start_method: Multiprocessing start method (defaults to ``"spawn"`` for
            compatibility with CUDA + Mujoco setups).

    Returns:
        List of results (one per trial, order may differ from input).
    """
    if not trial_ids:
        return []

    if num_workers <= 1:
        return [single_trial_fn(tid) for tid in trial_ids]

    ctx = multiprocessing.get_context(mp_start_method)
    task_queue: multiprocessing.Queue = ctx.Queue()
    result_queue: multiprocessing.Queue = ctx.Queue()

    # Populate task queue
    for trial_id in trial_ids:
        task_queue.put(trial_id)

    # Add sentinel values to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Start worker processes
    workers = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(worker_id, task_queue, result_queue, single_trial_fn),
        )
        p.start()
        workers.append(p)

    # Collect results
    results: list[BatchResult] = []
    errors: list[tuple[int, str]] = []
    for _ in range(len(trial_ids)):
        status, trial_id, data = result_queue.get()
        if status == "success":
            results.append(data)
        else:
            errors.append((trial_id, data))
            print(f"Trial {trial_id} failed with error: {data}")

    # Wait for all workers to finish
    for p in workers:
        p.join()

    if errors:
        print(f"WARNING: {len(errors)} trial(s) failed")

    return results


def run_parallel_with_setup(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    setup_fn: Callable[[], Any],
    trial_fn: Callable[[Any, int], BatchResult],
    mp_start_method: str = "spawn",
) -> list[BatchResult]:
    """Execute trials dynamically with per-worker setup (e.g., environment creation).

    Each worker calls ``setup_fn`` once to create its state (e.g., a simulation
    environment), then loops pulling trials from a shared queue and calling
    ``trial_fn(state, trial_id)`` for each. This provides:
    - Efficient environment reuse (no re-creation per trial)
    - Dynamic load balancing (workers grab trials as they become available)

    Args:
        trial_ids: Ordered trial identifiers to evaluate.
        num_workers: Number of worker processes. Values <= 1 run sequentially.
        setup_fn: Callable that creates per-worker state (called once per worker).
            Must be picklable.
        trial_fn: Callable that takes (state, trial_id) and returns a result.
            Must be picklable.
        mp_start_method: Multiprocessing start method (defaults to ``"spawn"`` for
            compatibility with CUDA + Mujoco setups).

    Returns:
        List of results (one per trial, order may differ from input).
    """
    if not trial_ids:
        return []

    if num_workers <= 1:
        state = setup_fn()
        return [trial_fn(state, tid) for tid in trial_ids]

    ctx = multiprocessing.get_context(mp_start_method)
    task_queue: multiprocessing.Queue = ctx.Queue()
    result_queue: multiprocessing.Queue = ctx.Queue()

    # Populate task queue
    for trial_id in trial_ids:
        task_queue.put(trial_id)

    # Add sentinel values to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Start worker processes
    workers = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=_worker_loop_with_setup,
            args=(worker_id, task_queue, result_queue, setup_fn, trial_fn),
        )
        p.start()
        workers.append(p)

    # Collect results
    results: list[BatchResult] = []
    errors: list[tuple[int, str]] = []
    setup_errors = 0

    for _ in range(len(trial_ids)):
        status, identifier, data = result_queue.get()
        if status == "success":
            results.append(data)
        elif status == "setup_error":
            setup_errors += 1
            print(f"Worker {identifier} failed during setup: {data}")
        else:
            errors.append((identifier, data))
            print(f"Trial {identifier} failed with error: {data}")

    # Wait for all workers to finish
    for p in workers:
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join()

    if setup_errors:
        print(f"WARNING: {setup_errors} worker(s) failed during setup")
    if errors:
        print(f"WARNING: {len(errors)} trial(s) failed")

    return results


__all__ = [
    "chunk_into_batches",
    "run_parallel_batches",
    "run_parallel_dynamic",
    "run_parallel_with_setup",
]
