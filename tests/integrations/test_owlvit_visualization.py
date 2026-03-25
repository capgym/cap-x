from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tyro
from PIL import Image


@dataclass
class Config:
    """Command-line options for OWL-ViT/OWL-v2 detection visualization."""

    image_path: Path = Path("scripts/images/robosuite_spill.png")
    texts: tuple[str, ...] = ("spill",)
    models: tuple[str, ...] = ("google/owlvit-large-patch14",)
    device: str = "cuda"
    threshold: float = 0.1
    show: bool = True
    save_path: Path | None = None
    compare: bool = True  # If True, compare OWL-ViT and OWL-v2 side by side


def _generate_colors(n: int) -> Sequence[tuple[float, float, float]]:
    """Generate distinct colors for different detection classes."""
    cmap = plt.get_cmap("tab10", max(n, 1))
    return [tuple(cmap(i)[:3]) for i in range(n)]


def init_detector(model_name: str, device: str, threshold: float):
    """Initialize OWL-ViT or OWL-v2 detector."""
    import torch
    from transformers import (
        Owlv2ForObjectDetection,
        Owlv2Processor,
        OwlViTForObjectDetection,
        OwlViTProcessor,
    )

    # Determine if this is OWL-v2 or OWL-ViT
    is_v2 = "owlv2" in model_name.lower()
    
    if is_v2:
        processor = Owlv2Processor.from_pretrained(model_name)
        model = Owlv2ForObjectDetection.from_pretrained(model_name)
    else:
        processor = OwlViTProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(model_name)
    
    model = model.to(device)
    model.eval()

    def detect_fn(rgb: np.ndarray, texts: list[str]) -> list[dict[str, Any]]:
        """Run detection on an image with text queries."""
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        
        # OWL models expect text queries as a list of lists
        text_queries = [texts]
        
        inputs = processor(text=text_queries, images=rgb_u8, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})

        target_sizes = torch.tensor([rgb_u8.shape[:2]], device=model.device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )
        
        detections: list[dict[str, Any]] = []
        i = 0
        labels = texts
        for box, score, label in zip(
            results[i]["boxes"], results[i]["scores"], results[i]["labels"], strict=False
        ):
            b = box.detach().to("cpu").numpy().tolist()
            detections.append(
                {
                    "label": labels[int(label)],
                    "score": float(score.item()),
                    "box": [round(x, 2) for x in b],  # [x_min, y_min, x_max, y_max]
                }
            )
        return detections

    return detect_fn


def _plot_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    colors: dict[str, tuple[float, float, float]],
    title: str = "Detections",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot detections on an image."""
    if ax is None:
        _, ax = plt.subplots()
    
    ax.imshow(image)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_axis_off()

    for det in detections:
        x_min, y_min, x_max, y_max = det["box"]
        label = det["label"]
        score = det["score"]
        color = colors.get(label, (1, 0, 0))
        
        # Draw bounding box
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        
        # Add label with score
        label_text = f"{label.replace('a photo of a ', '')}: {score:.2f}"
        ax.text(
            x_min,
            y_min - 5,
            label_text,
            color="white",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=(*color, 0.8), ec="none"),
        )
    
    # Add detection count
    ax.text(
        0.02,
        0.98,
        f"Detections: {len(detections)}",
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        weight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc=(0, 0, 0, 0.7), ec="none"),
    )
    
    return ax


def main(cfg: Config) -> None:
    """Main visualization function."""
    image = Image.open(cfg.image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    
    texts = list(cfg.texts)
    print(f"Text queries: {texts}")
    
    # Generate colors for each label
    colors = _generate_colors(len(texts))
    color_map = {label: colors[i] for i, label in enumerate(texts)}
    
    # Handle comparison mode
    if cfg.compare:
        # Use both OWL-ViT and OWL-v2
        models = [
            "google/owlvit-large-patch14",
            "google/owlv2-large-patch14-ensemble",
        ]
        model_titles = ["OWL-ViT", "OWL-v2"]
    else:
        models = list(cfg.models)
        model_titles = [m.split("/")[-1] for m in models]
    
    # Create figure
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    # Run detection for each model
    for idx, (model_name, title) in enumerate(zip(models, model_titles)):
        print(f"\nRunning {title} ({model_name})...")
        
        detect_fn = init_detector(model_name, cfg.device, cfg.threshold)
        detections = detect_fn(rgb, texts)
        
        print(f"Found {len(detections)} detections:")
        for det in detections:
            print(f"  {det['label']}: {det['score']:.3f} @ {det['box']}")
        
        _plot_detections(rgb, detections, color_map, title=title, ax=axes[idx])
    
    plt.tight_layout()
    
    # Save figure
    if cfg.save_path is None:
        suffix = "_owlvit_comparison" if cfg.compare else f"_owlvit_{model_titles[0]}"
        save_path = cfg.image_path.with_name(cfg.image_path.stem + suffix + ".png")
    else:
        save_path = cfg.save_path
    
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved visualization to {save_path}")
    
    if cfg.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    tyro.cli(main)