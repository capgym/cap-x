"""SAM3 mask popup display helpers."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np


def _bool_from_string(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sam3_mask_popup_enabled(env: Any) -> bool:
    """Return whether SAM3 mask popups should be shown for an environment."""

    env_value = os.environ.get("CAPX_SAM3_MASK_POPUP")
    if env_value is not None:
        return _bool_from_string(env_value)
    return bool(getattr(env, "sam3_mask_popup", False))


def _has_gui_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class Sam3MaskPopupDisplay:
    """Small OpenCV HighGUI wrapper that never raises into robot control."""

    def __init__(self, window_name: str = "CaP-X SAM3 Mask") -> None:
        self.window_name = window_name
        self._warned_no_display = False
        self._warned_failure = False

    def show(self, image: np.ndarray, *, title: str, wait_ms: int = 0) -> bool:
        """Show the popup with an RGB image overlay.

        Args:
            image: RGB uint8 image or image-like array.
            title: Text drawn on the image before display.
            wait_ms: OpenCV event-loop delay. Use 0 to block until any key is pressed.

        Returns:
            True when the image was handed to the GUI backend, False otherwise.
        """

        if not _has_gui_display():
            if not self._warned_no_display:
                print("[sam3-mask-popup] no desktop DISPLAY/WAYLAND_DISPLAY; skipping popup.")
                self._warned_no_display = True
            return False

        try:
            import cv2

            frame = np.asarray(image)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError(f"Expected HxWx3 image, got shape {frame.shape}.")
            frame = frame[:, :, :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            canvas = frame.copy()
            if title:
                label = title[:160]
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
                y1 = max(0, min(canvas.shape[0] - 1, 6 + text_h + baseline))
                x2 = min(canvas.shape[1], text_w + 16)
                if y1 > 0 and x2 > 0:
                    canvas[: y1 + 6, :x2] = (0.35 * canvas[: y1 + 6, :x2]).astype(np.uint8)
                    cv2.putText(
                        canvas,
                        label,
                        (8, min(canvas.shape[0] - 1, 8 + text_h)),
                        font,
                        scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

            cv2.imshow(self.window_name, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            delay = int(wait_ms)
            cv2.waitKey(0 if delay <= 0 else max(1, delay))
            if delay <= 0:
                cv2.destroyWindow(self.window_name)
            return True
        except Exception as exc:  # pragma: no cover - depends on local GUI backend
            if not self._warned_failure:
                print(f"[sam3-mask-popup] failed to show popup: {exc}")
                self._warned_failure = True
            return False
