from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import requests
from PIL import Image

from capx.utils.serve_utils import post_with_retries

"""SAM 2 integration via FastAPI service."""

# Configuration
SERVICE_URL = "http://127.0.0.1:8113"


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


def _extract_masks_from_payload(
    payload: dict[str, Any], max_masks: int | None = None
) -> list[dict[str, Any]]:
    masks_raw = payload.get("masks", [])
    scores_raw = payload.get("scores")
    scores_list = _normalize_scores(scores_raw)

    parsed: list[dict[str, Any]] = []
    for idx, entry in enumerate(_ensure_iterable(masks_raw)):
        score: float | None = None
        mask_like: Any | None = None

        if isinstance(entry, dict):
            mask_like = entry.get("mask") or entry.get("segmentation")
            score = entry.get("score")
            if score is None and idx < len(scores_list):
                score = scores_list[idx]
        else:
            mask_like = (
                getattr(entry, "mask", None)
                or getattr(entry, "segmentation", None)
                or entry
            )
            if hasattr(entry, "score"):
                score = entry.score  # type: ignore[assignment]
            if idx < len(scores_list):
                score = scores_list[idx]

        mask_bool = _to_numpy_bool(mask_like)
        if mask_bool is None or mask_bool.size == 0:
            continue
        parsed.append({"mask": mask_bool, "score": float(score or 0.0)})

    parsed.sort(key=lambda item: float(item["score"]), reverse=True)
    if max_masks is not None:
        parsed = parsed[:max_masks]
    return parsed


def _parse_pipeline_outputs(outputs: Any, max_masks: int | None = None) -> list[dict[str, Any]]:
    if isinstance(outputs, dict):
        return _extract_masks_from_payload(outputs, max_masks)

    if hasattr(outputs, "to_dict"):
        return _extract_masks_from_payload(outputs.to_dict(), max_masks)  # type: ignore[no-untyped-call]

    if hasattr(outputs, "masks"):
        payload: dict[str, Any] = {"masks": outputs.masks}
        if hasattr(outputs, "scores"):
            payload["scores"] = outputs.scores
        return _extract_masks_from_payload(payload, max_masks)

    if isinstance(outputs, (list, tuple)):
        combined: list[dict[str, Any]] = []
        for item in outputs:
            combined.extend(_parse_pipeline_outputs(item, None))
        combined.sort(key=lambda elem: float(elem["score"]), reverse=True)
        if max_masks is not None:
            combined = combined[:max_masks]
        return combined

    raise TypeError(f"Unexpected output type from SAM2 pipeline: {type(outputs)!r}")


def init_sam2(
    model_name: str = "facebook/sam2.1-hiera-large",
    device: str | int = "cuda",
    points_per_batch: int = 64,
) -> Any:
    """Initialize SAM2 segmentation client.

    Returns a callable `segment_fn(rgb: np.ndarray)`.

    Note: This function now uses server-based inference only and does not load
    models locally to save GPU memory when running multiple workers.
    """

    global _GENERATOR, _POINTS_PER_BATCH, _PROCESSOR, _MODEL

    # Server-based mode: Don't load models locally to save GPU memory
    # The segment_fn below will use the HTTP service instead
    _GENERATOR = None
    _POINTS_PER_BATCH = points_per_batch
    _PROCESSOR = None
    _MODEL = None

    def _reshape_mask(arr: np.ndarray, height: int, width: int) -> np.ndarray:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 3 and arr.shape[-2:] == (height, width):
            # Shape is (N, H, W) - take first mask
            arr = arr[0]
        if arr.ndim == 1 and arr.size == height * width:
            return arr.reshape(height, width)
        if arr.ndim == 3:
            # Still 3D after all conversions - take first channel/mask
            return arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
        return arr

    def _run_box_prompt(
        pil_image: Image.Image, box: Sequence[float], max_masks: int | None
    ) -> list[dict[str, Any]]:
        assert _PROCESSOR is not None and _MODEL is not None, "SAM2 processor/model unavailable."

        # Process image WITH boxes using the processor
        # Format: [[[x_min, y_min, x_max, y_max]]]
        input_boxes = [[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]]

        # Use the exact same pattern as visualize_prompted_sam2.py
        inputs = _PROCESSOR(
            images=pil_image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(_MODEL.device)

        with torch.no_grad():
            outputs = _MODEL(**inputs)  # type: ignore[misc]

        pred_masks = getattr(outputs, "pred_masks", None)
        if pred_masks is None:
            pred_masks = outputs.get("pred_masks")
        if pred_masks is None:
            raise RuntimeError("SAM2 model did not return pred_masks.")

        # Use the same approach as visualize_prompted_sam2.py
        height, width = pil_image.size[1], pil_image.size[0]

        try:
            # Try using processor's post_process_masks
            masks_processed = _PROCESSOR.post_process_masks(
                pred_masks.cpu(),
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"],
            )[0]
            # Threshold to binary and convert to numpy
            parsed_masks = [
                (masks_processed[i] > 0.0).numpy() for i in range(masks_processed.shape[0])
            ]
        except (KeyError, RuntimeError, TypeError):
            # Fallback: manual interpolation (same as visualize_prompted_sam2.py)
            from torch.nn.functional import interpolate

            masks_raw = pred_masks.cpu()

            # Handle 5D tensor: (batch, boxes, num_masks, h_pred, w_pred)
            if masks_raw.ndim == 5:
                masks_raw = masks_raw[
                    0, 0
                ]  # Take first batch and box -> (num_masks, h_pred, w_pred)

            # Interpolate to original size
            if masks_raw.ndim == 3:
                masks_resized = interpolate(
                    masks_raw.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]  # Remove batch dim -> (num_masks, h, w)
            elif masks_raw.ndim == 4:
                masks_resized = interpolate(
                    masks_raw, size=(height, width), mode="bilinear", align_corners=False
                )[0]  # Take first batch -> (num_masks, h, w)
            else:
                raise ValueError(f"Unexpected mask shape: {masks_raw.shape}")

            # Threshold to binary
            parsed_masks = [(masks_resized[i] > 0.0).numpy() for i in range(masks_resized.shape[0])]

        # Get scores
        scores_obj = getattr(outputs, "iou_scores", None)
        if scores_obj is None and isinstance(outputs, dict):
            scores_obj = outputs.get("iou_scores")

        if scores_obj is not None:
            scores_raw = scores_obj.cpu().numpy()
            scores = scores_raw.flatten() if scores_raw.size > 0 else []
        else:
            scores = []

        result: list[dict[str, Any]] = []
        for idx, mask in enumerate(parsed_masks):
            score_val = float(scores[idx]) if idx < len(scores) else 0.0
            result.append({"mask": mask, "score": score_val})

        result.sort(key=lambda item: float(item["score"]), reverse=True)
        if max_masks is not None:
            result = result[:max_masks]
        return result

    def segment_fn(
        rgb: np.ndarray,
        *,
        max_masks: int | None = None,
        box: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        encoded_image = _encode_image(rgb)
        payload = {
            "image_base64": encoded_image,
            "box": box,
            "max_masks": max_masks,
            "points_per_batch": points_per_batch,
        }

        try:
            data = post_with_retries(f"{SERVICE_URL}/segment", payload=payload)
            # resp.raise_for_status()
            # data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to communicate with SAM2 service at {SERVICE_URL}: {e}")

        results = []
        for item in data["masks"]:
            mask_shape = tuple(item["shape"])
            # /segment masks are always boolean (serialized as uint8 or similar from service)
            # The service serializes them as uint8 bytes of boolean mask.
            mask = _decode_mask(item["mask_base64"], mask_shape, dtype=np.uint8).astype(bool)
            results.append({"mask": mask, "score": item["score"]})
        return results

    return segment_fn

def init_sam2_point_prompt(
    model_name: str = "facebook/sam2.1-hiera-large",
    device: str | int = "cuda",
) -> Any:
    """Initialize SAM2 point prompt client.

    Returns a callable `point_prompt_fn(rgb: np.ndarray,
    point_coords: tuple(float, float)) -> tuple[list[float], list[np.ndarray]]`.
    """

    def segment_from_point_prompt(
        image: np.ndarray | Image.Image, point_coords: tuple[float, float]
    ) -> tuple[list[float], list[np.ndarray]]:
        encoded_image = _encode_image(image)
        payload = {"image_base64": encoded_image, "point_coords": point_coords}

        try:
            data = post_with_retries(f"{SERVICE_URL}/segment_point", payload=payload)
            # resp.raise_for_status()
            # data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to communicate with SAM2 service at {SERVICE_URL}: {e}")

        scores = data["scores"]
        masks_shape = tuple(data["masks_shape"])
        masks_dtype_str = data["masks_dtype"]

        masks = _decode_mask(data["masks_base64"], masks_shape, dtype=np.dtype(masks_dtype_str))

        return scores, masks

        # we need to get the scores as well
        masks_hf = (
            _PROCESSOR.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0][0]
            .cpu()
            .numpy()
        )
        iou_scores_hf = outputs.iou_scores[0][0].cpu().numpy()
        mask_sort_idxs = np.argsort(iou_scores_hf)[::-1]
        masks_hf = masks_hf[mask_sort_idxs]
        iou_scores_hf = iou_scores_hf[mask_sort_idxs]
        return iou_scores_hf, masks_hf

    return segment_from_point_prompt