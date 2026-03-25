import base64
import io
import logging
from collections.abc import Sequence
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import Sam2Model, Sam2Processor, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global state
_GENERATOR: Any | None = None
_PROCESSOR: Any | None = None
_MODEL: Any | None = None
_POINTS_PER_BATCH: int = 64
_DEVICE: str = "cuda"

# --- Helper Functions (copied from sam2.py) ---


def _ensure_iterable(obj: Any) -> Any:  # Simplified return type annotation
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return [obj]
        if obj.ndim <= 2:
            return [obj]
        return [obj[i] for i in range(obj.shape[0])]

    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        tensor = obj
        arr = tensor.detach().cpu().numpy()  # type: ignore[union-attr]
        if arr.ndim == 0:
            return [arr]
        if arr.ndim <= 2:
            return [arr]
        return [arr[i] for i in range(arr.shape[0])]

    return [obj]


def _normalize_scores(scores: Any) -> list[float]:
    if scores is None:
        return []
    if hasattr(scores, "detach"):
        return list(scores.detach().cpu().numpy().tolist())  # type: ignore[no-untyped-call]
    if hasattr(scores, "cpu"):
        return list(scores.cpu().numpy().tolist())  # type: ignore[no-untyped-call]
    if hasattr(scores, "numpy"):
        return list(scores.numpy().tolist())  # type: ignore[no-untyped-call]
    if isinstance(scores, (list, tuple)):
        return [float(s) for s in scores]
    if np.isscalar(scores):
        return [float(scores)]
    return [float(s) for s in np.asarray(scores).ravel()]


def _to_numpy_bool(mask_like: Any) -> np.ndarray | None:
    if mask_like is None:
        return None
    if isinstance(mask_like, np.ndarray):
        arr = mask_like
    elif hasattr(mask_like, "detach"):
        arr = mask_like.detach().cpu().numpy()
    elif hasattr(mask_like, "cpu"):
        arr = mask_like.cpu().numpy()
    elif hasattr(mask_like, "numpy"):
        arr = mask_like.numpy()
    else:
        arr = np.asarray(mask_like)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(bool)


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
                getattr(entry, "mask", None) or getattr(entry, "segmentation", None) or entry
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


# --- Request/Response Models ---


class SegmentRequest(BaseModel):
    image_base64: str
    box: list[float] | None = None
    max_masks: int | None = None


class MaskData(BaseModel):
    mask_base64: str
    shape: tuple[int, int]
    score: float


class SegmentResponse(BaseModel):
    masks: list[MaskData]


class PointPromptRequest(BaseModel):
    image_base64: str
    point_coords: tuple[float, float]


class PointPromptResponse(BaseModel):
    scores: list[float]
    masks_base64: str
    masks_shape: tuple[int, int, int]  # N, H, W
    masks_dtype: str


# --- Core Logic ---


def decode_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_mask(mask: np.ndarray) -> str:
    # Pack boolean mask to bytes
    # To be safe and simple, we can cast to uint8 and tobytes, or packbits.
    # np.packbits is more compact but requires tracking original shape carefully if not byte-aligned.
    # Given we pass shape, packbits is good.
    # But for simplicity let's just use uint8 bytes + base64 for now, less prone to alignment bugs.
    # The client can just np.frombuffer(..., dtype=np.uint8).reshape(...)
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


def encode_array(arr: np.ndarray) -> str:
    # General array encoding
    # We'll stick to tobytes for float arrays too
    return base64.b64encode(arr.tobytes()).decode("utf-8")


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    if _GENERATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    pil_image = decode_image(req.image_base64)
    height, width = pil_image.size[1], pil_image.size[0]

    if req.box is not None:
        # Box Prompt Logic
        if _PROCESSOR is None or _MODEL is None:
            raise HTTPException(
                status_code=503, detail="SAM2 processor/model unavailable for box prompt."
            )

        box = req.box
        # Format: [[[x_min, y_min, x_max, y_max]]]
        input_boxes = [[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]]

        inputs = _PROCESSOR(
            images=pil_image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(_MODEL.device)

        with torch.no_grad():
            outputs = _MODEL(**inputs)

        pred_masks = getattr(outputs, "pred_masks", None)
        if pred_masks is None:
            pred_masks = outputs.get("pred_masks")
        if pred_masks is None:
            raise RuntimeError("SAM2 model did not return pred_masks.")

        try:
            masks_processed = _PROCESSOR.post_process_masks(
                pred_masks.cpu(),
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"],
            )[0]
            parsed_masks = [
                (masks_processed[i] > 0.0).numpy() for i in range(masks_processed.shape[0])
            ]
        except (KeyError, RuntimeError, TypeError):
            # Fallback interpolation
            from torch.nn.functional import interpolate

            masks_raw = pred_masks.cpu()
            if masks_raw.ndim == 5:
                masks_raw = masks_raw[0, 0]

            if masks_raw.ndim == 3:
                masks_resized = interpolate(
                    masks_raw.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            elif masks_raw.ndim == 4:
                masks_resized = interpolate(
                    masks_raw, size=(height, width), mode="bilinear", align_corners=False
                )[0]
            else:
                raise ValueError(f"Unexpected mask shape: {masks_raw.shape}")

            parsed_masks = [(masks_resized[i] > 0.0).numpy() for i in range(masks_resized.shape[0])]

        scores_obj = getattr(outputs, "iou_scores", None)
        if scores_obj is None and isinstance(outputs, dict):
            scores_obj = outputs.get("iou_scores")

        if scores_obj is not None:
            scores_raw = scores_obj.cpu().numpy()
            scores = scores_raw.flatten() if scores_raw.size > 0 else []
        else:
            scores = []

        result_masks = []
        for idx, mask in enumerate(parsed_masks):
            score_val = float(scores[idx]) if idx < len(scores) else 0.0
            result_masks.append({"mask": mask, "score": score_val})

        result_masks.sort(key=lambda item: float(item["score"]), reverse=True)
        if req.max_masks is not None:
            result_masks = result_masks[: req.max_masks]

    else:
        # Full Image Segmentation Logic
        outputs = _GENERATOR(pil_image, points_per_batch=_POINTS_PER_BATCH)
        parsed = _parse_pipeline_outputs(outputs, req.max_masks)

        result_masks = []
        for item in parsed:
            mask_val = item.get("mask")
            if hasattr(mask_val, "detach"):
                mask_arr = mask_val.detach().cpu().numpy()
            else:
                mask_arr = np.asarray(mask_val, dtype=np.float32)

            mask_arr = _reshape_mask(mask_arr, height, width)
            result_masks.append(
                {"mask": mask_arr.astype(bool), "score": float(item.get("score", 0.0))}
            )

    # Serialize Response
    response_data = []
    for item in result_masks:
        mask_np = item["mask"]
        response_data.append(
            MaskData(mask_base64=encode_mask(mask_np), shape=mask_np.shape, score=item["score"])
        )

    return SegmentResponse(masks=response_data)


@app.post("/segment_point", response_model=PointPromptResponse)
async def segment_point(req: PointPromptRequest):
    if _PROCESSOR is None or _MODEL is None:
        raise HTTPException(status_code=503, detail="SAM2 processor/model unavailable")

    pil_image = decode_image(req.image_base64)
    point_coords = [[[list(req.point_coords)]]]

    inputs = _PROCESSOR(pil_image, input_points=point_coords, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        outputs = _MODEL(**inputs)

    masks_hf = (
        _PROCESSOR.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0][0]
        .cpu()
        .numpy()
    )
    iou_scores_hf = outputs.iou_scores[0][0].cpu().numpy()

    mask_sort_idxs = np.argsort(iou_scores_hf)[::-1]
    masks_hf = masks_hf[mask_sort_idxs]
    iou_scores_hf = iou_scores_hf[mask_sort_idxs]

    return PointPromptResponse(
        scores=iou_scores_hf.tolist(),
        masks_base64=encode_array(masks_hf),
        masks_shape=masks_hf.shape,
        masks_dtype=str(masks_hf.dtype),
    )


def main(
    model_name: str = "facebook/sam2.1-hiera-large",
    device: str = "cuda",
    port: int = 8113,
    host: str = "127.0.0.1",
):
    global _GENERATOR, _PROCESSOR, _MODEL, _POINTS_PER_BATCH, _DEVICE

    _DEVICE = device
    if device.startswith("cuda") and ":" not in device:
        device_arg = f"{device}:0"
    else:
        device_arg = device

    logger.info(f"Loading SAM2 model: {model_name} on {device_arg}...")

    # Initialize Pipeline
    _GENERATOR = pipeline("mask-generation", model=model_name, device=device_arg)

    # Initialize Processor/Model for prompting
    _PROCESSOR = Sam2Processor.from_pretrained(model_name)
    _MODEL = getattr(_GENERATOR, "model", None)

    # If pipeline didn't expose model or we need to ensure it's on right device/mode:
    if _MODEL is None:
        _MODEL = Sam2Model.from_pretrained(model_name).to(device_arg)
    elif hasattr(_MODEL, "to"):
        _MODEL = _MODEL.to(device_arg)

    logger.info("Model loaded. Starting Server...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
