"""Backward-compatible re-export of execution_logger.

The execution logger has been moved to capx.utils.execution_logger.
This shim provides backward compatibility for existing imports.

New code should import from:
    from capx.utils.execution_logger import log_step, log_step_update
"""

# Re-export all public symbols from the new location
from capx.utils.execution_logger import (
    ExecutionHistory,
    ExecutionStep,
    clear_all_histories,
    finalize_execution_context,
    get_all_histories,
    get_current_history,
    get_execution_steps_with_images,
    get_execution_summary_for_vlm,
    init_execution_context,
    log_step,
    log_step_update,
    set_auto_init,
)

__all__ = [
    "ExecutionStep",
    "ExecutionHistory",
    "init_execution_context",
    "finalize_execution_context",
    "clear_all_histories",
    "get_all_histories",
    "get_current_history",
    "set_auto_init",
    "log_step",
    "log_step_update",
    "get_execution_summary_for_vlm",
    "get_execution_steps_with_images",
]
