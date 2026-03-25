from __future__ import annotations

import io
import os

import numpy as np
import pytest
import requests
from PIL import Image

from capx.integrations import sam2 as sam2_mod
from capx.tools import tool_api


def test_sam2_registers_and_segments(monkeypatch: object) -> None:
    class FakePred:
        def __init__(self) -> None:
            self._called = False

        def generate(self, rgb: np.ndarray):
            h, w = rgb.shape[:2]
            return [{"segmentation": np.zeros((h, w), dtype=bool), "predicted_iou": 0.9}]

        def set_image(self, rgb: np.ndarray) -> None:
            self._called = True

        def predict(
            self, point_coords: np.ndarray, point_labels: np.ndarray, multimask_output: bool
        ):
            h, w = 8, 8
            masks = np.zeros((1, h, w), dtype=bool)
            scores = np.array([0.5], dtype=np.float32)
            return masks, scores, None

    def build(model_cfg: str, checkpoint: str, device: str):  # noqa: ARG001
        return object()

    class FakeImagePredictor:
        def __init__(self, model: object) -> None:  # noqa: ARG002
            self._pred = FakePred()

        def __getattr__(self, name: str):
            return getattr(self._pred, name)

    monkeypatch.setattr(
        sam2_mod,
        "_build_sam2_image_predictor",
        lambda a, b, c: FakeImagePredictor(object()),
        raising=True,
    )  # type: ignore[arg-type]

    sam2_mod.init_sam2("cfg.yaml", "ckpt.pt", device="cpu")

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    masks = tool_api.segment_anything(rgb)
    assert isinstance(masks, list) and len(masks) >= 1
    assert "mask" in masks[0] and "score" in masks[0]


@pytest.mark.integration
def test_sam2_real_init_and_segment() -> None:
    if os.environ.get("HYRL_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set HYRL_INTEGRATION_REAL=1 to run real SAM2 test")

    model_cfg = os.environ.get("HYRL_SAM2_CFG", "sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
    ckpt_path = os.environ.get("HYRL_SAM2_CKPT", "sam2.1_hiera_tiny.pt")

    sam2_mod.init_sam2(model_cfg, ckpt_path, device="cpu")

    url = os.environ.get(
        "HYRL_TEST_IMAGE_URL", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((320, 320))
    rgb = np.asarray(img, dtype=np.uint8)

    masks = tool_api.segment_anything(rgb)
    assert isinstance(masks, list)
