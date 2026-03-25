"""Execution logger for detailed code execution feedback.

This module provides functions to log execution steps during code execution.
It works in both web UI mode (with WebSocket emission) and CLI mode (standalone).
Steps can be retrieved for VLM analysis or saved to disk.

Usage in API code (e.g., agibot_api.py):
    from capx.utils.execution_logger import log_step, log_step_update

    # Before an operation
    log_step("SAM3 Segmentation", "Querying SAM3 for 'red cube'...")

    # After operation completes
    result_image = ...
    log_step_update(images=[result_image], text="Found 3 instances")

    # Or in one call with images
    log_step("IK Planning", "Planning trajectory...", images=[before_img])

For VLM analysis:
    from capx.utils.execution_logger import get_execution_steps_with_images

    steps = get_execution_steps_with_images()
    # Returns list of dicts with 'tool', 'description', 'images', 'num_images'
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Get current UTC time as ISO format string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionStep:
    """A single execution step with tool name, text, and optional images."""

    tool_name: str
    text: str
    images: list[str] = field(default_factory=list)  # Base64 encoded images
    timestamp: str = field(default_factory=_utc_now_iso)
    step_index: int = 0
    highlight: bool = False  # If True, display with highlighted color scheme

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "text": self.text,
            "images": self.images,
            "timestamp": self.timestamp,
            "step_index": self.step_index,
            "highlight": self.highlight,
        }

    def to_vlm_format(self, include_images: bool = True) -> dict[str, Any]:
        """Format for VLM analysis - can include images as base64."""
        result: dict[str, Any] = {
            "tool": self.tool_name,
            "description": self.text,
            "timestamp": self.timestamp,
        }
        if include_images and self.images:
            result["images"] = self.images
            result["num_images"] = len(self.images)
        return result


@dataclass
class ExecutionHistory:
    """Collection of execution steps for a code execution."""

    steps: list[ExecutionStep] = field(default_factory=list)
    code_block_index: int = 0
    start_time: str = field(default_factory=_utc_now_iso)
    end_time: str | None = None

    def add_step(self, step: ExecutionStep) -> None:
        step.step_index = len(self.steps)
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_block_index": self.code_block_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_vlm_summary(self, include_image_counts: bool = True) -> str:
        """Generate a text summary suitable for VLM analysis.

        Args:
            include_image_counts: Whether to mention image counts in the summary
        """
        lines = [f"## Execution History (Code Block {self.code_block_index})"]
        lines.append(f"Started: {self.start_time}")
        if self.end_time:
            lines.append(f"Ended: {self.end_time}")
        lines.append("")

        for i, step in enumerate(self.steps):
            lines.append(f"### Step {i + 1}: {step.tool_name}")
            lines.append(step.text)
            if include_image_counts and step.images:
                lines.append(f"*({len(step.images)} image(s) captured)*")
            lines.append("")

        return "\n".join(lines)

    def get_steps_for_vlm(self, include_images: bool = True) -> list[dict[str, Any]]:
        """Get steps in a format suitable for VLM message construction."""
        return [s.to_vlm_format(include_images) for s in self.steps]

    def save_to_directory(self, output_dir: Path | str) -> None:
        """Save execution history to a directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON metadata
        history_file = output_dir / f"execution_history_block_{self.code_block_index}.json"
        # Save without images in JSON (too large)
        history_data = {
            "code_block_index": self.code_block_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps": [{
                "tool_name": s.tool_name,
                "text": s.text,
                "num_images": len(s.images),
                "timestamp": s.timestamp,
                "step_index": s.step_index,
            } for s in self.steps],
        }
        history_file.write_text(json.dumps(history_data, indent=2))

        # Save images separately
        for step in self.steps:
            for img_idx, img_b64 in enumerate(step.images):
                img_file = output_dir / f"block_{self.code_block_index}_step_{step.step_index}_img_{img_idx}.jpg"
                try:
                    img_data = base64.b64decode(img_b64)
                    img_file.write_bytes(img_data)
                except Exception as e:
                    logger.warning(f"Failed to save image: {e}")


# Global storage for all execution histories in current session
_all_histories: list[ExecutionHistory] = []
_current_history: ExecutionHistory | None = None
_emit_callback: Callable[[ExecutionStep], None] | None = None
_emit_callback_async: Callable[[ExecutionStep], Any] | None = None
_auto_init_enabled: bool = True  # Auto-initialize context on first log_step


def _encode_image(image: np.ndarray | Image.Image | str) -> str:
    """Encode image to base64 string."""
    if isinstance(image, str):
        # Already base64 or file path
        if image.startswith("data:") or len(image) > 1000:
            # Likely already base64
            if image.startswith("data:"):
                return image.split(",", 1)[-1]
            return image
        # File path
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # PIL Image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def init_execution_context(
    code_block_index: int = 0,
    emit_callback: Callable[[ExecutionStep], None] | None = None,
    emit_callback_async: Callable[[ExecutionStep], Any] | None = None,
) -> None:
    """Initialize execution context for a new code block execution.

    Args:
        code_block_index: Index of the code block being executed
        emit_callback: Sync callback to emit steps (called in thread, e.g., for WebSocket)
        emit_callback_async: Async callback to emit steps
    """
    global _current_history, _emit_callback, _emit_callback_async

    _current_history = ExecutionHistory(code_block_index=code_block_index)
    _emit_callback = emit_callback
    _emit_callback_async = emit_callback_async

    logger.debug(f"Initialized execution context for block {code_block_index}")


def finalize_execution_context() -> ExecutionHistory | None:
    """Finalize current execution context and return the history.

    Returns:
        The completed ExecutionHistory, or None if no context was active.
    """
    global _current_history, _emit_callback, _emit_callback_async, _all_histories

    if _current_history is None:
        return None

    _current_history.end_time = _utc_now_iso()
    history = _current_history
    _all_histories.append(history)

    _current_history = None
    _emit_callback = None
    _emit_callback_async = None

    logger.debug(f"Finalized execution context with {len(history.steps)} steps")
    return history


def clear_all_histories() -> None:
    """Clear all stored execution histories and reset state."""
    global _all_histories, _current_history, _emit_callback, _emit_callback_async
    _all_histories = []
    _current_history = None
    _emit_callback = None
    _emit_callback_async = None


def get_all_histories() -> list[ExecutionHistory]:
    """Get all execution histories from current session."""
    return _all_histories.copy()


def get_current_history() -> ExecutionHistory | None:
    """Get the current active execution history."""
    return _current_history


def set_auto_init(enabled: bool) -> None:
    """Enable or disable auto-initialization of execution context.

    When enabled (default), calling log_step without an active context
    will automatically create one. Disable this if you want strict control.
    """
    global _auto_init_enabled
    _auto_init_enabled = enabled


def log_step(
    tool_name: str,
    text: str,
    images: list[np.ndarray | Image.Image | str] | np.ndarray | Image.Image | str | None = None,
    highlight: bool = False,
) -> None:
    """Log an execution step.

    This function is designed to be called from within code execution
    (e.g., from agibot_api.py methods). It works in both web UI mode
    (with WebSocket callbacks) and standalone CLI mode.

    In standalone mode (no init_execution_context called), steps are still
    recorded and can be retrieved via get_all_histories().

    Args:
        tool_name: Name of the tool/API being called (e.g., "SAM3 Segmentation")
        text: Description text (supports markdown)
        images: Optional image(s) to display. Can be:
            - numpy array (H, W, 3) RGB
            - PIL Image
            - Base64 string
            - File path
            - List of any of the above
        highlight: If True, display with highlighted color scheme in UI

    Example:
        log_step("IK Planning", "Planning trajectory to target pose...")
        log_step("Camera Capture", "Captured image:", images=rgb_image)
        log_step("Detection", "Found objects:", images=[img1, img2, img3])
        log_step("Result", "Task completed!", highlight=True)
    """
    global _current_history, _emit_callback, _emit_callback_async, _auto_init_enabled

    # Auto-initialize context if not active and auto-init is enabled
    if _current_history is None and _auto_init_enabled:
        init_execution_context(code_block_index=0)
        logger.debug("Auto-initialized execution context for standalone mode")

    # Encode images
    encoded_images: list[str] = []
    if images is not None:
        if not isinstance(images, list):
            images = [images]
        for img in images:
            try:
                encoded_images.append(_encode_image(img))
            except Exception as e:
                logger.warning(f"Failed to encode image: {e}")

    step = ExecutionStep(
        tool_name=tool_name,
        text=text,
        images=encoded_images,
        highlight=highlight,
    )

    # Add to history if context is active
    if _current_history is not None:
        _current_history.add_step(step)

    # Emit via callback (for web UI WebSocket)
    if _emit_callback is not None:
        try:
            _emit_callback(step)
        except Exception as e:
            logger.warning(f"Failed to emit step via sync callback: {e}")

    if _emit_callback_async is not None:
        try:
            # Schedule async callback - it will be awaited by the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_emit_callback_async(step), loop)
        except Exception as e:
            logger.warning(f"Failed to emit step via async callback: {e}")

    logger.debug(f"Logged step: {tool_name} - {text[:50]}...")


def log_step_update(
    text: str | None = None,
    images: list[np.ndarray | Image.Image | str] | np.ndarray | Image.Image | str | None = None,
) -> None:
    """Update the last logged step with additional text or images.

    Useful for adding results after an operation completes.

    Args:
        text: Additional text to append (will be added on new line)
        images: Additional images to add
    """
    global _current_history, _emit_callback

    if _current_history is None or not _current_history.steps:
        logger.debug("log_step_update called but no active step - ignoring")
        return

    last_step = _current_history.steps[-1]

    if text:
        last_step.text = f"{last_step.text}\n\n{text}"

    if images is not None:
        if not isinstance(images, list):
            images = [images]
        for img in images:
            try:
                last_step.images.append(_encode_image(img))
            except Exception as e:
                logger.warning(f"Failed to encode image: {e}")

    # Re-emit the updated step
    if _emit_callback is not None:
        try:
            _emit_callback(last_step)
        except Exception as e:
            logger.warning(f"Failed to emit updated step: {e}")


def get_execution_summary_for_vlm(
    include_image_counts: bool = True,
    max_history_blocks: int | None = None,
) -> str:
    """Get a formatted execution summary suitable for VLM prompts.

    Args:
        include_image_counts: Whether to mention image counts in the summary
        max_history_blocks: Maximum number of history blocks to include (None = all)

    Returns:
        Markdown-formatted execution summary
    """
    histories = _all_histories
    if max_history_blocks is not None:
        histories = histories[-max_history_blocks:]

    if not histories:
        return "No execution history available."

    summaries = [h.to_vlm_summary(include_image_counts) for h in histories]
    return "\n\n---\n\n".join(summaries)


def get_execution_steps_with_images(
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Get execution steps with images for VLM analysis.

    This returns a list of steps that can be used to construct
    VLM messages with interleaved text and images.

    Args:
        max_steps: Maximum number of steps to return (None = all)

    Returns:
        List of step dicts with 'tool', 'description', 'images', 'num_images'
    """
    all_steps = []
    for history in _all_histories:
        all_steps.extend(history.get_steps_for_vlm(include_images=True))

    if max_steps is not None:
        all_steps = all_steps[-max_steps:]

    return all_steps
