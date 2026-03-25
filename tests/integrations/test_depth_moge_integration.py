from __future__ import annotations

import io
import os

import numpy as np
import pytest
import requests
from PIL import Image

from capx.integrations import depth_moge as moge_mod
from capx.tools import tool_api


@pytest.mark.integration
def test_moge_registers_and_predicts_real() -> None:
    if os.environ.get("HYRL_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set HYRL_INTEGRATION_REAL=1 to run real model tests")
    # Initialize real MoGe model on CPU
    moge_mod.init_moge_depth(device="cpu")

    # Download a real image (small) for testing
    url = os.environ.get(
        "HYRL_TEST_IMAGE_URL", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((256, 256))
    rgb = np.asarray(img, dtype=np.uint8)

    depth = tool_api.depth_anything(rgb)
    assert depth.shape[:2] == rgb.shape[:2]

    K = np.array([[100.0, 0, 4.0], [0, 100.0, 4.0], [0, 0, 1]])
    pts = tool_api.backproject_depth_to_points(depth, K)
    assert pts.shape == (rgb.shape[0], rgb.shape[1], 3)

    # Grasp planner API exists (returns [] by default without real model)
    res = tool_api.grasp_plan(depth, K)
    assert isinstance(res, list)
