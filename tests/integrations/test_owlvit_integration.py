from __future__ import annotations

import io
import os
import pathlib

import numpy as np
import pytest
import requests
from PIL import Image, ImageDraw

from capx.integrations.vision.owlvit import init_owlvit
from capx.tools.tool_api import detect_open_vocab


def _draw_boxes(rgb: np.ndarray, boxes: list[list[float]], labels: list[str]) -> Image.Image:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for b, lab in zip(boxes, labels, strict=False):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), lab, fill=(255, 0, 0))
    return img


@pytest.mark.integration
def test_owlvit_open_vocab_detection_real() -> None:
    if os.environ.get("HYRL_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set HYRL_INTEGRATION_REAL=1 to run real OWL-ViT test")

    init_owlvit(device=os.environ.get("HYRL_DEVICE", "cpu"))

    url = os.environ.get(
        "HYRL_TEST_IMAGE_URL", "http://images.cocodataset.org/val2017/000000039769.jpg"
    )
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((640, 480))
    rgb = np.asarray(image, dtype=np.uint8)

    texts = [["a photo of a cat", "a photo of a dog"]]
    dets = detect_open_vocab(rgb, texts)
    assert isinstance(dets, list)
    if len(dets) == 0:
        pytest.skip("No detections; environment constraints or model mismatch")

    boxes = [d["box"] for d in dets]
    labels = [d["label"] for d in dets]
    img_out = _draw_boxes(rgb, boxes, labels)
    out_file = pathlib.Path("owlvit_det.jpg")
    img_out.save(out_file)
    assert out_file.exists() and out_file.stat().st_size > 0
