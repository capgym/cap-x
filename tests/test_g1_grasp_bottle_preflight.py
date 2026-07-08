from pathlib import Path

import numpy as np


def test_deproject_masked_points_uses_metric_depth_and_intrinsics() -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import deproject_masked_points

    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[False, True], [False, False]])
    intrinsics = np.array([[2.0, 0.0, 0.5], [0.0, 4.0, 0.25], [0.0, 0.0, 1.0]])

    points, colors = deproject_masked_points(depth, rgb, mask, intrinsics)

    assert np.allclose(points, [[0.5, -0.125, 2.0]])
    assert np.allclose(colors, [[0.0, 1.0, 0.0]])


def test_transform_pose_matrix_to_wxyz_xyz() -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import transform_pose_matrix

    t_world_camera = np.eye(4)
    t_world_camera[:3, 3] = [0.1, 0.2, 0.3]
    t_camera_object = np.eye(4)
    t_camera_object[:3, 3] = [1.0, 2.0, 3.0]

    t_world_object, pose = transform_pose_matrix(t_world_camera, t_camera_object)

    assert np.allclose(t_world_object[:3, 3], [1.1, 2.2, 3.3])
    assert np.allclose(pose["position"], [1.1, 2.2, 3.3])
    assert np.allclose(pose["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0])


def test_extract_right_arm_ik_solution_from_with_hand_config() -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import extract_right_arm_ik_solution

    cfg = np.arange(43, dtype=np.float64)

    right_arm = extract_right_arm_ik_solution(cfg)

    assert np.array_equal(right_arm, np.arange(29, 36, dtype=np.float64))


def test_build_report_html_embeds_summary_and_artifact_paths(tmp_path: Path) -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import build_report_html

    summary = {
        "object_name": "plastic water bottle",
        "object_modal_pose_world": {
            "position": [0.1, 0.2, 0.3],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "grasp_pose_world": {
            "position": [0.4, 0.5, 0.6],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "grasp_pose_world_matrix": [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.6],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "ik": {"final_right_arm_joints": [0.0] * 7},
    }
    artifacts = {"sam3_overlay": "sam3.png", "segment_pointcloud_ply": "segment.ply"}

    html = build_report_html(summary, artifacts, point_preview={"points": [], "colors": []})

    assert "plastic water bottle" in html
    assert "sam3.png" in html
    assert "segment.ply" in html
    assert "final_right_arm_joints" in html


def test_build_report_html_overlays_final_grasp_pose_on_pointcloud(tmp_path: Path) -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import build_report_html

    summary = {
        "object_name": "plastic water bottle",
        "grasp_pose_world_matrix": [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.6],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "ik": {
            "approach_position_world": [0.4, 0.5, 0.75],
            "final_right_arm_joints": [0.0] * 7,
        },
    }

    html = build_report_html(
        summary,
        {},
        point_preview={
            "points": [[0.35, 0.45, 0.55], [0.45, 0.55, 0.65]],
            "colors": [[1.0, 0.0, 0.0], [0.0, 0.3, 1.0]],
        },
    )

    assert "3D Grasp Preview" in html
    assert "graspPoseMatrix" in html
    assert '"approachPosition":[0.4,0.5,0.75]' in html
    assert "drawGraspPose" in html
    assert "Final hand pose" in html


def test_scene_preview_includes_camera_pose_and_ik_stick_figure() -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import _scene_preview

    summary = {
        "camera": {"T_world_camera": np.eye(4).tolist()},
        "ik": {
            "final_stick_figure": {
                "frame": "torso_link",
                "links": [
                    {"name": "torso_link", "position": [0.0, 0.0, 0.0]},
                    {"name": "right_hand_palm_link", "position": [0.2, -0.1, 0.1]},
                ],
                "segments": [["torso_link", "right_hand_palm_link"]],
            }
        },
    }

    scene = _scene_preview(summary, {"points": [], "colors": []})

    assert scene["cameraPoseMatrix"] == summary["camera"]["T_world_camera"]
    assert scene["ikStickFigure"]["frame"] == "torso_link"
    assert scene["ikStickFigure"]["segments"] == [["torso_link", "right_hand_palm_link"]]


def test_build_report_html_draws_ik_right_arm_stick_figure(tmp_path: Path) -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import build_report_html

    summary = {
        "object_name": "plastic water bottle",
        "camera": {"T_world_camera": np.eye(4).tolist()},
        "grasp_pose_world_matrix": np.eye(4).tolist(),
        "ik": {
            "final_right_arm_joints": [0.0] * 7,
            "final_stick_figure": {
                "frame": "torso_link",
                "links": [
                    {"name": "torso_link", "position": [0.0, 0.0, 0.0]},
                    {"name": "right_shoulder_pitch_link", "position": [0.0, -0.1, 0.2]},
                    {"name": "right_hand_palm_link", "position": [0.25, -0.15, 0.05]},
                ],
                "segments": [
                    ["torso_link", "right_shoulder_pitch_link"],
                    ["right_shoulder_pitch_link", "right_hand_palm_link"],
                ],
            },
        },
    }

    html = build_report_html(summary, {}, point_preview={"points": [], "colors": []})

    assert "drawRobotStickFigure" in html
    assert "cameraPoseMatrix" in html
    assert "right_hand_palm_link" in html
    assert "IK right arm" in html


def test_build_report_html_adds_roll_view_control() -> None:
    from tools.g1_vision_bridge.g1_grasp_bottle_preflight import build_report_html

    html = build_report_html(
        {"object_name": "plastic water bottle"},
        {},
        point_preview={"points": [], "colors": []},
    )

    assert 'id="roll"' in html
    assert "const rollEl = document.getElementById('roll');" in html
    assert "const roll = Number(rollEl.value) * Math.PI / 180;" in html
    assert "cr: Math.cos(roll)" in html
    assert "sr: Math.sin(roll)" in html
    assert "rollEl.addEventListener('input', draw);" in html
