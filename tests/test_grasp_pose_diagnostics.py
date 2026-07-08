from __future__ import annotations

import numpy as np
import pytest


def test_sample_grasp_pose_reports_empty_grasp_candidates() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.full((4, 4, 1), 0.75, dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)

    class FakeEnv:
        def get_observation(self) -> dict:
            return {
                "robot0_robotview": {
                    "images": {
                        "rgb": rgb,
                        "depth": depth,
                    },
                    "intrinsics": np.array(
                        [
                            [100.0, 0.0, 2.0],
                            [0.0, 100.0, 2.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                    "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                }
            }

    api = G1RealControlApi(FakeEnv())
    api.sam3_seg_fn = lambda _rgb, text_prompt: [
        {"score": 0.95, "box": [0, 0, 3, 3], "mask": mask}
    ]
    api.grasp_net_plan_fn = lambda *args, **kwargs: (
        np.empty((0, 4, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="ContactGraspNet returned no grasp candidates") as exc:
        api.sample_grasp_pose("plastic water bottle")

    message = str(exc.value)
    assert "mask_pixels=16" in message
    assert "valid_mask_depth_pixels=16" in message
    assert "segment_points=16" in message


def test_sample_grasp_pose_pushes_sam3_overlay_to_popup_display() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 120
    depth = np.full((4, 4, 1), 0.75, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    class FakeDisplay:
        def __init__(self) -> None:
            self.calls = []

        def show(self, image, *, title: str, wait_ms: int = 1) -> bool:
            self.calls.append((image.copy(), title, wait_ms))
            return True

    class FakeEnv:
        sam3_mask_popup = True

        def __init__(self) -> None:
            self.sam3_mask_popup_display = FakeDisplay()

        def get_observation(self) -> dict:
            return {
                "robot0_robotview": {
                    "images": {
                        "rgb": rgb,
                        "depth": depth,
                    },
                    "intrinsics": np.array(
                        [
                            [100.0, 0.0, 2.0],
                            [0.0, 100.0, 2.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                    "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                }
            }

    env = FakeEnv()
    api = G1RealControlApi(env)
    api.sam3_seg_fn = lambda _rgb, text_prompt: [
        {"score": 0.95, "box": [0, 0, 3, 3], "mask": mask}
    ]
    api.grasp_net_plan_fn = lambda *args, **kwargs: (
        np.empty((0, 4, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="ContactGraspNet returned no grasp candidates"):
        api.sample_grasp_pose("plastic water bottle")

    assert len(env.sam3_mask_popup_display.calls) == 1
    overlay, title, wait_ms = env.sam3_mask_popup_display.calls[0]
    assert overlay.shape == rgb.shape
    assert title == "SAM3 mask: plastic water bottle score=0.950"
    assert wait_ms == 0

def test_sample_grasp_pose_popup_uses_best_mask_when_score_is_low() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.full((4, 4, 1), 0.75, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    class FakeDisplay:
        def __init__(self) -> None:
            self.calls = []

        def show(self, image, *, title: str, wait_ms: int = 1) -> bool:
            self.calls.append((image.copy(), title, wait_ms))
            return True

    class FakeEnv:
        sam3_mask_popup = True

        def __init__(self) -> None:
            self.sam3_mask_popup_display = FakeDisplay()

        def get_observation(self) -> dict:
            return {
                "robot0_robotview": {
                    "images": {"rgb": rgb, "depth": depth},
                    "intrinsics": np.eye(3, dtype=np.float32),
                    "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                }
            }

    env = FakeEnv()
    api = G1RealControlApi(env)
    api.sam3_seg_fn = lambda _rgb, text_prompt: [
        {"score": 0.01, "box": [0, 0, 3, 3], "mask": mask}
    ]
    api.grasp_net_plan_fn = lambda *args, **kwargs: (
        np.empty((0, 4, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="ContactGraspNet returned no grasp candidates"):
        api.sample_grasp_pose("plastic water bottle")

    assert len(env.sam3_mask_popup_display.calls) == 1
    assert env.sam3_mask_popup_display.calls[0][1] == "SAM3 mask: plastic water bottle score=0.010"
    assert env.sam3_mask_popup_display.calls[0][2] == 0

