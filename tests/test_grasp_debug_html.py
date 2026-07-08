from __future__ import annotations

from pathlib import Path

import numpy as np


def test_build_grasp_debug_html_embeds_point_cloud_and_grasp_frame() -> None:
    from capx.utils.grasp_debug_html import build_grasp_debug_html

    html = build_grasp_debug_html(
        object_name="plastic water bottle",
        points=np.array([[0.1, 0.0, 0.2], [0.2, 0.0, 0.25]], dtype=np.float64),
        colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        grasp_position=np.array([0.15, 0.0, 0.3], dtype=np.float64),
        grasp_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        approach_position=np.array([0.15, 0.0, 0.4], dtype=np.float64),
    )

    assert "G1 Grasp Debug: plastic water bottle" in html
    assert "pointCloud" in html
    assert "graspPoseMatrix" in html
    assert "approachPosition" in html
    assert "drawGraspPose" in html


def test_save_grasp_debug_html_writes_file(tmp_path: Path) -> None:
    from capx.utils.grasp_debug_html import save_grasp_debug_html

    path = save_grasp_debug_html(
        tmp_path / "scene.html",
        object_name="plastic water bottle",
        points=np.array([[0.1, 0.0, 0.2]], dtype=np.float64),
        colors=np.array([[255, 128, 0]], dtype=np.uint8),
        grasp_position=np.array([0.15, 0.0, 0.3], dtype=np.float64),
        grasp_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        approach_position=np.array([0.15, 0.0, 0.4], dtype=np.float64),
    )

    assert path.exists()
    assert "plastic water bottle" in path.read_text()
