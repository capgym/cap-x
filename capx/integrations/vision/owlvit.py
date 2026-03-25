from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import requests
from PIL import Image

from capx.utils.serve_utils import post_with_retries

"""OWL-ViT integration via FastAPI service."""

# Configuration
SERVICE_URL = "http://127.0.0.1:8117"


def _encode_image(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        image_u8 = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
        pil_image = Image.fromarray(image_u8).convert("RGB")
    else:
        pil_image = image

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def init_owlvit(
    model_name: str = "google/owlv2-large-patch14-ensemble",
    device: str = "cpu",
    threshold: float = 0.05,
) -> Any:
    """Initialize OWL-ViT or OWL-v2 object detector client.

    Note: This function now uses server-based inference only and does not load
    models locally to save GPU memory when running multiple workers.

    Args:
        model_name: Name of the model to use (ignored in server mode, kept for compatibility).
            Examples:
            - "google/owlvit-large-patch14" (OWL-ViT)
            - "google/owlvit-base-patch32" (OWL-ViT)
            - "google/owlv2-large-patch14-ensemble" (OWL-v2)
            - "google/owlv2-base-patch16-ensemble" (OWL-v2)
        device: Device argument (ignored in server mode, kept for compatibility).
        threshold: Confidence threshold for detections (ignored in server mode, kept for compatibility).
    Returns:
        Detection function that takes (rgb, texts) and returns list of detections
    """
    # Server-based mode: Don't load models locally to save GPU memory
    # The det_fn below will use the HTTP service instead

    def det_fn(rgb: np.ndarray, texts: list[list[str]] | None = None) -> list[dict[str, Any]]:
        encoded_image = _encode_image(rgb)
        payload = {
            "image_base64": encoded_image,
            "texts": texts,
            "threshold": threshold,
        }

        try:
            data = post_with_retries(f"{SERVICE_URL}/detect", payload=payload)
            # resp.raise_for_status()
            # data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to communicate with OWL-ViT service at {SERVICE_URL}: {e}")

        # Convert response to the expected format
        detections = []
        for det in data["detections"]:
            detections.append(
                {
                    "label": det["label"],
                    "score": det["score"],
                    "box": det["box"],
                }
            )
        return detections

    return det_fn
