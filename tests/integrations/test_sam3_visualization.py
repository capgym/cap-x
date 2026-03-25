from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tyro
from PIL import Image

from capx.integrations.vision.sam3 import init_sam3

@dataclass
class Config:
    """Command-line options for SAM3 segmentation visualization."""

    image_path: Path = Path("scripts/images/robosuite_spill2.png")
    prompts: tuple[str, ...] = ("brown spill",)
    device: str = "cuda"
    show: bool = True
    save_path: Path | None = None
    save_dir: Path = Path("scripts/images")


def _visualize_results(
    image: Image.Image,
    prompt: str,
    results: list[dict[str, Any]],
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Visualize SAM3 masks and boxes on the image."""
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
        # Should be caught by the "not results" check, but safe fallback
        axes = [axes] 
    elif not isinstance(axes, np.ndarray):
        # If only 1 result + original, subplots might return 1D array or single axes if squeezed
        # With 2 subplots it usually returns array. 
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
            (x1, y1), width, height, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax_main.add_patch(rect)
        ax_main.text(
            x1, y1, f"{score:.2f}", 
            color='white', fontsize=8, backgroundcolor='red'
        )

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
        color_mask = np.array([30, 144, 255], dtype=np.uint8) # Dodger Blue
        
        # mask is boolean (H, W)
        if mask.shape[:2] == overlay.shape[:2]:
            overlay[mask] = overlay[mask] * 0.5 + color_mask * 0.5
        
        ax.imshow(overlay)
        
        # Draw box
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h, 
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.set_title(f"Score: {score:.2f}")
        ax.axis("off")
        
        # Save individual mask if needed
        if output_dir:
            mask_path = output_dir / f"mask_{prompt.replace(' ', '_')}_{idx}_{score:.2f}.jpg"
            Image.fromarray(overlay).save(mask_path)

    plt.tight_layout()
    
    if output_dir:
        grid_path = output_dir / f"sam3_{prompt.replace(' ', '_')}.jpg"
        plt.savefig(grid_path)
        print(f"Saved visualization to: {grid_path}")
    
    if show:
        plt.show()
    
    plt.close()


def main(cfg: Config) -> None:
    # Initialize SAM3
    print(f"Initializing SAM3 on {cfg.device}...")
    try:
        sam3_fn = init_sam3(device=cfg.device)
    except Exception as e:
        print(f"Failed to initialize SAM3: {e}")
        return

    # Load Image
    if not cfg.image_path.exists():
        print(f"Image not found: {cfg.image_path}")
        # Attempt to find it relative to workspace root if current dir is different
        # Use an absolute path lookup as fallback if user provided relative path not working
        resolved_path = Path.cwd() / cfg.image_path
        if resolved_path.exists():
            cfg.image_path = resolved_path
        else:
             print(f"Could not resolve image path: {cfg.image_path}")
             return
    
    image = Image.open(cfg.image_path).convert("RGB")
    print(f"Loaded image: {cfg.image_path} {image.size}")

    print(f"Testing prompts: {cfg.prompts}")

    for prompt in cfg.prompts:
        print(f"\nRunning inference for: '{prompt}'")
        try:
            results = sam3_fn(image, prompt)
            print(f"Found {len(results)} matches.")
            
            if results:
                # Visualize
                save_dir = cfg.save_dir / cfg.image_path.stem
                _visualize_results(
                    image, 
                    prompt, 
                    results, 
                    output_dir=save_dir,
                    show=cfg.show
                )
                
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tyro.cli(main)
