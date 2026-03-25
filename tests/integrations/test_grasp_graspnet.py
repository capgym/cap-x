from __future__ import annotations

import io
import os

import numpy as np
import pytest
import requests
from PIL import Image

from capx.integrations.vision.graspnet import init_graspnet
from capx.tools import tool_api


@pytest.mark.integration
def test_graspnet_real_depth_to_grasps() -> None:
    if os.environ.get("HYRL_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set HYRL_INTEGRATION_REAL=1 to run real GraspNet test")
    # Skip gracefully if the PointNet2 CUDA extension is not available
    try:
        import sys as _sys

        # try vendored path first
        _sys.path.append("capx/third_party/contact_graspnet_pytorch/pointnet2")
        import pointnet2._ext  # type: ignore  # noqa: F401
    except Exception:
        pytest.skip("pointnet2._ext not built for current env; skipping GraspNet test")

    # Init model (optional checkpoint path via env)
    ckpt = os.environ.get("HYRL_GRASPNET_CKPT", "") or None
    init_graspnet(device="cpu", checkpoint_path=ckpt)

    # Download an RGB image and synthesize a simple depth by grayscale
    url = os.environ.get(
        "HYRL_TEST_IMAGE_URL", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((160, 120))
    rgb = np.asarray(img, dtype=np.uint8)
    depth = np.mean(rgb, axis=2).astype(np.float32) / 255.0  # fake depth in meters ~ [0,1]

    # Simple pinhole intrinsics for backprojection; not used directly in graspnet baseline but required by our API
    K = np.array(
        [[120.0, 0, depth.shape[1] / 2], [0, 120.0, depth.shape[0] / 2], [0, 0, 1]],
        dtype=np.float32,
    )

    grasps = tool_api.grasp_plan(depth, K)
    assert isinstance(grasps, list)
    # We at least expect a list (maybe empty without a trained ckpt)
    if len(grasps) > 0:
        g0 = grasps[0]
        assert set(["pose", "width", "score"]).issubset(g0.keys())
