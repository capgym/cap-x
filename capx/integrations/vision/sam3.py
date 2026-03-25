from __future__ import annotations

import base64
import io
import pathlib
from collections.abc import Sequence
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from capx.utils.serve_utils import post_with_retries

"""SAM 3 integration via FastAPI service."""

# Configuration
SERVICE_URL = "http://127.0.0.1:8114"


def _encode_image(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        image_u8 = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
        pil_image = Image.fromarray(image_u8).convert("RGB")
    else:
        pil_image = image

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _decode_mask(mask_b64: str, shape: tuple[int, ...], dtype=np.uint8) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    # Using np.frombuffer directly on the decoded bytes
    return np.frombuffer(mask_bytes, dtype=dtype).reshape(shape)


def init_sam3(
    checkpoint_path: str | None = None,
    device: str = "cuda",
    model_type: str = "vit_l",
) -> Any:
    """Initialize SAM3 segmentation client.

    Returns a callable `segment_fn(image: np.ndarray | Image.Image, text_prompt: str) -> list[dict]`.

    Note: checkpoint_path, device, and model_type are ignored in client mode.
    """

    def segment_fn(
        image: np.ndarray | Image.Image,
        text_prompt: str,
        box_prompt: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run SAM3 inference with a text prompt via service.
        """
        encoded_image = _encode_image(image)
        payload = {"image_base64": encoded_image, "text_prompt": text_prompt}

        try:
            resp = post_with_retries(f"{SERVICE_URL}/segment", payload)
            results_data = resp["results"]
        except Exception as e:
            print(f"Failed to communicate with SAM3 service at {SERVICE_URL}: {e}")
            return []

        if not results_data:
            print(f"SAM3 returned no results for prompt: '{text_prompt}'")
            return []

        results = []
        for item in results_data:
            mask_shape = tuple(item["shape"])
            mask = _decode_mask(item["mask_base64"], mask_shape, dtype=np.uint8).astype(bool)

            results.append(
                {"mask": mask, "box": item["box"], "score": item["score"], "label": item["label"]}
            )

        return results

    return segment_fn


def visualize_sam3_results(
    image: Image.Image,
    prompt: str,
    results: list[dict[str, Any]],
    output_dir: pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Visualize SAM3 masks and boxes on the image. Adapted from visualize_sam3.py"""
    if not results:
        print(f"No results found for prompt: '{prompt}'")
        return

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup plot: Original + Individual detections
    # Limit to top 3 results to avoid overcrowding
    top_results = results[:3]

    fig, axes = plt.subplots(1, len(top_results) + 1, figsize=(4 * (len(top_results) + 1), 4))
    if len(top_results) == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = np.array([axes]) if not hasattr(axes, "__len__") else axes

    # Column 1: Original Image with all boxes
    ax_main = axes[0]
    ax_main.imshow(image)
    ax_main.set_title(f"Prompt: '{prompt}'")
    ax_main.axis("off")

    # Draw all boxes on main image
    for res in top_results:
        box = res["box"]
        score = res["score"]
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax_main.add_patch(rect)
        ax_main.text(x1, y1, f"{score:.2f}", color="white", fontsize=8, backgroundcolor="red")

    # Subsequent columns: Individual Mask + Box
    image_np = np.array(image)

    for idx, res in enumerate(top_results, start=1):
        if idx >= len(axes):
            break
        ax = axes[idx]
        mask = res["mask"]
        box = res["box"]
        score = res["score"]

        # Overlay mask
        overlay = image_np.copy()
        color_mask = np.array([30, 144, 255], dtype=np.uint8)  # Dodger Blue

        # mask is boolean (H, W)
        if mask.shape[:2] == overlay.shape[:2]:
            overlay[mask] = overlay[mask] * 0.5 + color_mask * 0.5

        ax.imshow(overlay)

        # Draw box
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)

        ax.set_title(f"Score: {score:.2f}")
        ax.axis("off")

        # Save individual mask if needed
        if output_dir:
            mask_path = output_dir / f"mask_{prompt.replace(' ', '_')}_{idx}_{score:.2f}.png"
            overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8, copy=False)
            overlay_img = Image.fromarray(overlay_u8, mode="RGB")
            overlay_img.save(str(mask_path))

    plt.tight_layout()

    if output_dir:
        grid_path = output_dir / f"sam3_{prompt.replace(' ', '_')}.png"
        plt.savefig(str(grid_path), format="png")
        print(f"Saved visualization to: {grid_path}")

    if show:
        plt.show()

    plt.close()


def init_sam3_point_prompt(
    device: str = "cuda",
) -> Any:
    """Initialize SAM3 point prompt client.

    Returns a callable:
        point_prompt_fn(image: np.ndarray | Image.Image,
                        point_coords: tuple[float, float])
            -> tuple[list[float], np.ndarray]
    """

    def point_prompt_fn(
        image: np.ndarray | Image.Image, point_coords: tuple[float, float]
    ) -> tuple[list[float], np.ndarray]:
        encoded_image = _encode_image(image)
        payload = {"image_base64": encoded_image, "point_coords": point_coords}

        try:
            resp = post_with_retries(f"{SERVICE_URL}/segment_point", payload)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to communicate with SAM3 service at {SERVICE_URL}: {e}")

        scores = resp.get("scores", [])
        masks_shape = tuple(resp.get("masks_shape", (0, 0, 0)))
        masks_dtype = np.dtype(resp.get("masks_dtype", "float32"))
        masks = _decode_mask(resp.get("masks_base64", ""), masks_shape, dtype=masks_dtype)
        masks = masks.astype(bool)
        results = [{"mask": mask, "score": score} for mask, score in zip(masks, scores)]

        return results

    return point_prompt_fn
