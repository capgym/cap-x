from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from capx.integrations.vision.molmo import init_molmo
from capx.integrations.vision.sam2 import init_sam2_point_prompt


def _to_numpy_mask(mask_like: Any) -> np.ndarray:
    """Convert various mask types (torch, PIL, np) to a boolean numpy array."""
    if hasattr(mask_like, "detach"):
        arr = mask_like.detach().cpu().numpy()
    elif hasattr(mask_like, "cpu"):
        arr = mask_like.cpu().numpy()
    elif hasattr(mask_like, "numpy"):
        arr = mask_like.numpy()
    else:
        arr = np.asarray(mask_like)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {arr.shape}")
    return arr > 0


def _iter_mask_candidates(masks: Any) -> list[Any]:
    """Collect mask-like objects from the nested SAM2 outputs."""
    if masks is None:
        return []

    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return [masks]
        if masks.ndim >= 3:
            return [masks[idx] for idx in range(masks.shape[0])]
        raise ValueError(f"Unexpected mask array shape {masks.shape}")

    if isinstance(masks, Sequence) and not isinstance(masks, (bytes, str)):
        candidates: list[Any] = []
        for entry in masks:
            candidates.extend(_iter_mask_candidates(entry))
        return candidates

    return [masks]


def _visualize_masks(
    image: Image.Image,
    point_coords: tuple[float, float],
    scores: Sequence[float],
    masks: Sequence[Any],
    max_masks: int = 3,
    save_dir: str | Path | None = None,
) -> None:
    """Overlay SAM2 masks on the image, show point prompt, and optionally save JPGs."""
    mask_candidates = _iter_mask_candidates(masks)
    if not mask_candidates:
        raise ValueError("No masks found in SAM2 output; cannot visualize.")

    masks_to_show = [_to_numpy_mask(mask_like) for mask_like in mask_candidates[:max_masks]]
    output_dir = Path(save_dir) if save_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(masks_to_show) + 1, figsize=(4 * (len(masks_to_show) + 1), 4))

    axes[0].imshow(image)
    axes[0].add_patch(
        Circle(point_coords, radius=8, edgecolor="red", facecolor="none", linewidth=2)
    )
    axes[0].set_title("Original Image")

    image_np = np.array(image)
    for idx, mask in enumerate(masks_to_show, start=1):
        overlay = image_np.copy()
        overlay[mask] = [255, 0, 0]
        axes[idx].imshow(overlay)

        if output_dir:
            mask_path = output_dir / f"mask_{idx}_{scores[idx - 1]:.2f}.jpg"
            Image.fromarray(overlay).save(mask_path)
            print(f"Saved mask {idx} (score {scores[idx - 1]:.2f}) to {mask_path}")

        score_txt = f"{scores[idx - 1]:.2f}" if idx - 1 < len(scores) else "n/a"
        axes[idx].set_title(f"Mask {idx} | score={score_txt}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if output_dir:
        grid_path = output_dir / "sam2_overlays.jpg"
        plt.savefig(grid_path)
        print(f"Saved visualization to: {grid_path}")
        plt.close()
    else:
        plt.show()


image_path = "first_frame.jpg"
image = Image.open(image_path)

molmo_det_fn = init_molmo()
points = molmo_det_fn(
    image, objects=["handle of the square nut", "square nut center", "square block"]
)
print(points)


sam2_det_fn = init_sam2_point_prompt()
for point_name, _point_coords in points.items():
    scores, masks = sam2_det_fn(image, point_coords=points[point_name])
    print(scores, masks)

    _visualize_masks(
        image=image,
        point_coords=points[point_name],
        scores=scores,
        masks=masks,
        save_dir=f"outputs/sam2_overlays/{point_name}",
    )
