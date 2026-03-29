import asyncio
import base64
import functools
import io
import logging
from typing import Any, List, Tuple

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global state
_PROCESSOR: Any | None = None
_MODEL: Any | None = None
_DEVICE: str = "cuda"

# Semaphore to serialize GPU access (prevents OOM from concurrent inference)
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    """Run a blocking GPU function without blocking the event loop."""
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

# --- Helper Functions ---


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor-like object to numpy array."""
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "cpu"):
        t = tensor.cpu()
        if hasattr(t, "dtype") and t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def decode_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_mask(mask: np.ndarray) -> str:
    # Pack boolean mask to bytes (uint8) then base64
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


def encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 bytes."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("utf-8")


# --- Request/Response Models ---


class SegmentRequest(BaseModel):
    image_base64: str
    text_prompt: str


class PointPromptRequest(BaseModel):
    image_base64: str
    point_coords: list[float]  # [x, y] — JSON arrays, not tuples


class PointPromptResponse(BaseModel):
    scores: list[float]
    masks_base64: str
    masks_shape: list[int]  # [num_masks, H, W]
    masks_dtype: str


class MaskData(BaseModel):
    mask_base64: str
    shape: list[int]  # [H, W]
    box: list[float]  # [x1, y1, x2, y2]
    score: float
    label: str


class SegmentResponse(BaseModel):
    results: list[MaskData]


# --- Core Logic ---


def _do_segment(pil_image: Image.Image, text_prompt: str):
    """Blocking SAM3 text-prompt segmentation (runs on GPU thread)."""
    dtype_context = (
        torch.autocast(_DEVICE, dtype=torch.bfloat16)
        if "cuda" in _DEVICE
        else torch.autocast("cpu")
    )

    with dtype_context:
        inference_state = _PROCESSOR.set_image(pil_image)
        output = _PROCESSOR.set_text_prompt(state=inference_state, prompt=text_prompt)

    masks_tensor = output.get("masks")
    boxes_tensor = output.get("boxes")
    scores_tensor = output.get("scores")

    if masks_tensor is None or boxes_tensor is None:
        return SegmentResponse(results=[])

    masks_np = _to_numpy(masks_tensor)
    boxes_np = _to_numpy(boxes_tensor)
    scores_np = _to_numpy(scores_tensor)

    # Squeeze masks if needed: (N, 1, H, W) -> (N, H, W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    results_data = []
    num_preds = len(scores_np)

    for i in range(num_preds):
        mask = masks_np[i] > 0  # Boolean mask
        box = boxes_np[i].tolist()  # [x1, y1, x2, y2]
        score = float(scores_np[i])

        results_data.append(
            MaskData(
                mask_base64=encode_mask(mask),
                shape=mask.shape,
                box=box,
                score=score,
                label=text_prompt,
            )
        )

    # Sort by score descending
    results_data.sort(key=lambda x: x.score, reverse=True)

    return SegmentResponse(results=results_data)


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    if _PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment, pil_image, req.text_prompt)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


def _do_segment_point(pil_image: Image.Image, point_coords_tuple: tuple[float, float]):
    """Blocking SAM3 point-prompt segmentation (runs on GPU thread)."""
    dtype_context = (
        torch.autocast(_DEVICE, dtype=torch.bfloat16)
        if "cuda" in _DEVICE
        else torch.autocast("cpu")
    )

    with dtype_context:
        inference_state = _PROCESSOR.set_image(pil_image)
        point_coords = np.array([list(point_coords_tuple)], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int64)  # foreground point
        masks, scores, _ = _MODEL.predict_inst(
            inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    masks_np = np.asarray(masks)
    scores_np = np.asarray(scores)

    if masks_np.size == 0 or scores_np.size == 0:
        return PointPromptResponse(
            scores=[],
            masks_base64="",
            masks_shape=(0, 0, 0),
            masks_dtype="float32",
        )

    # Sort by score descending
    sort_idx = np.argsort(scores_np)[::-1]
    masks_np = masks_np[sort_idx]
    scores_np = scores_np[sort_idx]

    return PointPromptResponse(
        scores=scores_np.astype(float).tolist(),
        masks_base64=encode_array(masks_np),
        masks_shape=tuple(masks_np.shape),
        masks_dtype=str(masks_np.dtype),
    )


@app.post("/segment_point", response_model=PointPromptResponse)
async def segment_point(req: PointPromptRequest):
    if _PROCESSOR is None or _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if getattr(_MODEL, "inst_interactive_predictor", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Instance interactivity not enabled on SAM3 model",
        )

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment_point, pil_image, req.point_coords)
    except Exception as e:
        logger.error(f"Point prompt inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Point prompt inference failed: {e}")


def main(
    device: str = "cuda",
    port: int = 8114,
    host: str = "127.0.0.1",
):
    global _MODEL, _PROCESSOR, _DEVICE

    _DEVICE = device

    # Setup torch settings for Ampere+ GPUs as recommended
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("Loading SAM3 model...")
    try:
        # Assuming build_sam3_image_model loads default checkpoint
        _MODEL = build_sam3_image_model(enable_inst_interactivity=True)
    except Exception as e:
        logger.error(f"Error building SAM3 model: {e}")
        raise

    if hasattr(_MODEL, "to") and device:
        _MODEL = _MODEL.to(device)

    _PROCESSOR = Sam3Processor(_MODEL, confidence_threshold=0.0)
    logger.info(f"SAM3 model loaded on {device}. Starting Server...")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
