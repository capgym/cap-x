import socket
import base64
import os
from pathlib import Path

import msgpack
import numpy as np
import pytest


def _jpeg_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return bytes(buf)


def _g1_composed_camera_jpeg_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    assert ok
    return bytes(buf)


def test_realsense_publisher_uses_factory_depth_alignment(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    requested_streams: list[str] = []
    alignment_targets: list[str] = []

    class FakeVideoStreamProfile:
        def as_video_stream_profile(self):
            return self

        def get_intrinsics(self):
            return SimpleNamespace(fx=400.0, fy=401.0, ppx=319.0, ppy=239.0)

    class FakeDevice:
        def first_depth_sensor(self):
            return SimpleNamespace(get_depth_scale=lambda: 0.001)

    class FakeProfile:
        def get_device(self):
            return FakeDevice()

        def get_stream(self, stream):
            requested_streams.append(stream)
            return FakeVideoStreamProfile()

    class FakePipeline:
        def start(self, config):
            return FakeProfile()

    class FakeConfig:
        def enable_device(self, serial):
            pass

        def enable_stream(self, *args):
            pass

    def fake_align(target):
        alignment_targets.append(target)
        return SimpleNamespace()

    fake_rs = SimpleNamespace(
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        pipeline=FakePipeline,
        config=FakeConfig,
        align=fake_align,
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    from tools.g1_vision_bridge.publisher_realsense import RealSenseRgbdCamera

    camera = RealSenseRgbdCamera(serial=None, width=640, height=480, fps=30)

    assert alignment_targets == ["depth"]
    assert requested_streams == ["depth"]
    assert np.allclose(
        camera.intrinsics,
        [[400.0, 0.0, 319.0], [0.0, 401.0, 239.0], [0.0, 0.0, 1.0]],
    )


def test_build_frame_converts_to_capx_observation_with_pose_from_matrix() -> None:
    from tools.g1_vision_bridge.protocol import build_frame, frame_to_capx_observation

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    depth_m = np.full((2, 3), 1.25, dtype=np.float32)
    intrinsics = np.array(
        [[300.0, 0.0, 1.5], [0.0, 300.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose_mat = np.eye(4, dtype=np.float32)
    pose_mat[:3, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    frame = build_frame(
        camera_name="robot0_robotview",
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        pose_mat=pose_mat,
        timestamp=123.0,
    )
    obs = frame_to_capx_observation(frame)

    camera_obs = obs["robot0_robotview"]
    assert np.array_equal(camera_obs["images"]["rgb"], rgb)
    assert camera_obs["images"]["depth"].shape == (2, 3, 1)
    assert camera_obs["images"]["depth"].dtype == np.float32
    assert np.allclose(camera_obs["images"]["depth"][:, :, 0], depth_m)
    assert np.array_equal(camera_obs["intrinsics"], intrinsics)
    assert np.allclose(camera_obs["pose_mat"], pose_mat)
    assert np.allclose(camera_obs["pose"], np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]))
    assert obs["timestamp"] == 123.0


def test_protocol_round_trips_numpy_frame_over_socket() -> None:
    from tools.g1_vision_bridge.protocol import build_frame, recv_framed, send_framed

    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth_m = np.linspace(0.1, 0.6, 6, dtype=np.float32).reshape(2, 3)
    frame = build_frame(
        camera_name="robot0_robotview",
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=np.eye(3, dtype=np.float32),
        pose_mat=np.eye(4, dtype=np.float32),
        timestamp=456.0,
    )

    left, right = socket.socketpair()
    try:
        send_framed(left, frame)
        decoded = recv_framed(right)
    finally:
        left.close()
        right.close()

    assert decoded["version"] == 1
    assert decoded["camera_name"] == "robot0_robotview"
    assert np.array_equal(decoded["rgb"], rgb)
    assert np.allclose(decoded["depth_m"], depth_m)


def test_frame_to_g1_lowlevel_message_uses_camera_top_schema() -> None:
    from tools.g1_vision_bridge.protocol import build_frame, frame_to_g1_lowlevel_message

    frame = build_frame(
        camera_name="robot0_robotview",
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        pose_mat=np.eye(4, dtype=np.float32),
        timestamp=789.0,
    )

    message = frame_to_g1_lowlevel_message(frame)

    assert set(message) == {"camera_top", "timestamp"}
    assert np.array_equal(message["camera_top"]["images"]["rgb"], frame["rgb"])
    assert message["camera_top"]["images"]["depth"].shape == (2, 3, 1)
    assert np.array_equal(message["camera_top"]["intrinsics_matrix"], frame["intrinsics"])
    assert np.array_equal(message["camera_top"]["pose_mat"], frame["pose_mat"])


def test_g1_d435_calibration_files_load_real_intrinsics_and_torso_extrinsics() -> None:
    from tools.g1_vision_bridge.calibration import load_intrinsics, load_pose_mat

    repo_root = Path(__file__).resolve().parents[1]

    intrinsics = load_intrinsics(
        str(repo_root / "env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml")
    )
    assert intrinsics is not None
    assert np.allclose(
        intrinsics,
        np.array(
            [
                [388.523101806641, 0.0, 319.235015869141],
                [0.0, 388.523101806641, 236.498229980469],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    pose_mat = load_pose_mat(
        str(repo_root / "env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml")
    )
    expected_pose_mat = np.array(
        [
            [0.0, -0.738455340626, 0.674302387584, 0.0576235],
            [-1.0, 0.0, 0.0, 0.01753],
            [0.0, -0.674302387584, -0.738455340626, 0.41987],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(pose_mat, expected_pose_mat)
    rotation = pose_mat[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)



def test_load_depth_to_rgb_transform_uses_t_rgb_depth_key(tmp_path: Path) -> None:
    from tools.g1_vision_bridge.calibration import load_depth_to_rgb_mat

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = np.array([0.01, -0.02, 0.003], dtype=np.float32)
    path = tmp_path / "d435_depth_to_rgb.yaml"
    path.write_text(
        "T_rgb_depth:\n"
        "  - [1.0, 0.0, 0.0, 0.01]\n"
        "  - [0.0, 1.0, 0.0, -0.02]\n"
        "  - [0.0, 0.0, 1.0, 0.003]\n"
        "  - [0.0, 0.0, 0.0, 1.0]\n",
        encoding="utf-8",
    )

    assert np.allclose(load_depth_to_rgb_mat(str(path)), transform)

def test_build_frame_rejects_bad_shapes() -> None:
    from tools.g1_vision_bridge.protocol import build_frame

    rgb = np.zeros((2, 3), dtype=np.uint8)
    depth_m = np.ones((2, 3), dtype=np.float32)

    try:
        build_frame(
            camera_name="robot0_robotview",
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=np.eye(3, dtype=np.float32),
            pose_mat=np.eye(4, dtype=np.float32),
            timestamp=0.0,
        )
    except ValueError as exc:
        assert "rgb" in str(exc)
    else:
        raise AssertionError("Expected build_frame to reject non-HxWx3 RGB input")


def test_viewer_builds_rgb_depth_canvas() -> None:
    from tools.g1_vision_bridge.protocol import build_frame
    from tools.g1_vision_bridge.viewer import build_display_canvas, colorize_depth

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    depth_m = np.array([[0.0, 0.5, 1.0], [1.5, 2.0, np.nan]], dtype=np.float32)
    frame = build_frame(
        camera_name="robot0_robotview",
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=np.eye(3, dtype=np.float32),
        pose_mat=np.eye(4, dtype=np.float32),
        timestamp=1.0,
    )

    depth_color = colorize_depth(depth_m, max_depth_m=2.0)
    canvas = build_display_canvas(frame, max_depth_m=2.0)

    assert depth_color.shape == rgb.shape
    assert depth_color.dtype == np.uint8
    assert canvas.shape == (2, 6, 3)
    assert canvas.dtype == np.uint8


def test_bridge_viewer_sam3_loop_updates_latest_frame() -> None:
    import threading
    from types import SimpleNamespace

    from tools.g1_vision_bridge.client_zmq_to_capx import _run_viewer_sam3_loop

    stop_event = threading.Event()
    snapshot = SimpleNamespace(
        frame_id=4,
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
    )
    calls = []

    class FakeViewer:
        def latest_snapshot(self):
            return snapshot

        def update_sam3_results(self, frame_snapshot, *, prompt, results):
            calls.append((frame_snapshot.frame_id, prompt, results))
            stop_event.set()

    def fake_segment(rgb, text_prompt):
        assert rgb.shape == (4, 5, 3)
        return [{"mask": np.zeros((4, 5), dtype=bool), "box": [0, 0, 1, 1], "score": 0.8}]

    _run_viewer_sam3_loop(
        FakeViewer(),
        fake_segment,
        "plastic water bottle",
        stop_event,
        period_s=0.0,
        log_errors=False,
    )

    assert len(calls) == 1
    frame_id, prompt, results = calls[0]
    assert frame_id == 4
    assert prompt == "plastic water bottle"
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.8)
    assert np.array_equal(results[0]["mask"], np.zeros((4, 5), dtype=bool))


def test_bridge_viewer_sam3_loop_handles_manual_refresh_request() -> None:
    import threading
    from types import SimpleNamespace

    from tools.g1_vision_bridge.client_zmq_to_capx import _run_viewer_sam3_loop

    stop_event = threading.Event()
    snapshot = SimpleNamespace(
        frame_id=8,
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
    )
    calls = []

    class FakeViewer:
        def __init__(self) -> None:
            self.request = SimpleNamespace(request_id=3, requested_time=0.0, after_frame_id=8)
            self.latest_sam3 = None

        def take_sam3_refresh_request(self):
            request = self.request
            self.request = None
            return request

        def latest_snapshot(self):
            return snapshot

        def update_sam3_results(self, frame_snapshot, *, prompt, results):
            calls.append((frame_snapshot.frame_id, prompt, results))
            self._latest_sam3 = SimpleNamespace(grasp={"position_world": [0.1, 0.2, 0.3]})
            stop_event.set()

    def fake_segment(rgb, text_prompt):
        assert rgb.shape == (4, 5, 3)
        return [{"mask": np.zeros((4, 5), dtype=bool), "box": [0, 0, 1, 1], "score": 0.8}]

    _run_viewer_sam3_loop(
        FakeViewer(),
        fake_segment,
        "plastic water bottle",
        stop_event,
        period_s=0.0,
        log_errors=False,
    )

    assert len(calls) == 1
    assert calls[0][0] == 8


def test_bridge_viewer_sample_loop_updates_independently() -> None:
    import threading
    from types import SimpleNamespace

    from tools.g1_vision_bridge.client_zmq_to_capx import _run_viewer_sample_loop

    stop_event = threading.Event()
    samples = [
        SimpleNamespace(
            rgb_key="ego_view",
            depth_key="ego_view_depth_m",
            timestamp=float(idx),
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_m=np.ones((2, 2), dtype=np.float32),
            depth_image=None,
        )
        for idx in range(3)
    ]

    class FakeCameraApi:
        config = SimpleNamespace(
            align_rgb_to_depth=False,
            intrinsics=np.eye(3, dtype=np.float64),
            pose_mat=np.eye(4, dtype=np.float64),
        )

        def samples(self):
            yield from samples

    class FakeViewer:
        def __init__(self) -> None:
            self.seen = []

        def update_sample(self, sample) -> None:
            self.seen.append(sample)
            if len(self.seen) == len(samples):
                stop_event.set()

    viewer = FakeViewer()

    _run_viewer_sample_loop(
        FakeCameraApi(),
        viewer,
        stop_event,
        reconnect_sleep_s=0.0,
        log_errors=False,
    )

    assert [sample.rgb_key for sample in viewer.seen] == ["ego_view", "ego_view", "ego_view"]
    assert [sample.timestamp for sample in viewer.seen] == [0.0, 1.0, 2.0]
    assert np.array_equal(viewer.seen[0].intrinsics, np.eye(3))
    assert np.array_equal(viewer.seen[0].pose_mat, np.eye(4))



def test_bridge_viewer_displays_aligned_frame_when_alignment_is_enabled() -> None:
    from types import SimpleNamespace

    from tools.g1_vision_bridge.client_zmq_to_capx import _viewer_sample_for_display

    sample = SimpleNamespace(
        rgb_key="ego_view",
        depth_key="ego_view_depth_m",
        timestamp=2.0,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2), dtype=np.float32),
        depth_image=None,
    )
    aligned_rgb = np.full((3, 4, 3), 123, dtype=np.uint8)
    aligned_depth = np.full((3, 4, 1), 0.75, dtype=np.float32)

    intrinsics = np.eye(3, dtype=np.float64) * 2.0
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[:3, 3] = [0.1, 0.2, 0.3]

    class FakeCameraApi:
        config = SimpleNamespace(align_rgb_to_depth=True, intrinsics=np.eye(3), pose_mat=np.eye(4))

        def sample_to_capx_observation(self, camera_sample):
            assert camera_sample is sample
            return {
                "camera_top": {
                    "images": {
                        "rgb": aligned_rgb,
                        "depth": aligned_depth,
                    },
                    "intrinsics_matrix": intrinsics,
                    "pose_mat": pose_mat,
                }
            }

    displayed = _viewer_sample_for_display(FakeCameraApi(), sample)

    assert displayed.rgb_key == "ego_view:aligned_to_depth"
    assert np.array_equal(displayed.rgb, aligned_rgb)
    assert np.array_equal(displayed.depth_m, aligned_depth[:, :, 0])
    assert np.array_equal(displayed.intrinsics, intrinsics)
    assert np.array_equal(displayed.pose_mat, pose_mat)

def test_web_viewer_payload_includes_sam3_overlay_when_available() -> None:
    from types import SimpleNamespace

    from tools.g1_vision_bridge.web_viewer import (
        build_frame_payload,
        build_sam3_overlay,
        snapshot_from_sample,
        update_snapshot_sam3,
    )

    sample = SimpleNamespace(
        rgb_key="ego_view",
        depth_key=None,
        timestamp=1.0,
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        depth_m=None,
        depth_image=None,
    )
    sample.rgb[:, :, 0] = 120
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True

    snapshot = snapshot_from_sample(sample, frame_id=3, received_time=10.0)
    overlay = build_sam3_overlay(sample.rgb, [{"mask": mask, "box": [2, 1, 4, 3], "score": 0.91}])
    snapshot = update_snapshot_sam3(
        snapshot,
        prompt="plastic water bottle",
        results=[{"mask": mask, "box": [2, 1, 4, 3], "score": 0.91}],
        overlay_rgb=overlay,
        received_time=11.0,
    )

    payload = build_frame_payload(snapshot)

    assert payload["sam3"]["ready"] is True
    assert payload["sam3"]["prompt"] == "plastic water bottle"
    assert payload["sam3"]["result_count"] == 1
    assert payload["sam3"]["best_score"] == pytest.approx(0.91)
    assert payload["sam3"]["overlay_data_url"].startswith("data:image/jpeg;base64,")


def test_web_viewer_computes_grasp_point_from_sam3_mask_depth_and_camera_pose() -> None:
    from types import SimpleNamespace

    from tools.g1_vision_bridge.web_viewer import build_frame_payload, snapshot_from_sample, update_snapshot_sam3

    depth = np.ones((3, 3), dtype=np.float32)
    intrinsics = np.array(
        [
            [100.0, 0.0, 1.0],
            [0.0, 100.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[:3, 3] = [0.5, -0.2, 0.1]
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    sample = SimpleNamespace(
        rgb_key="ego_view",
        depth_key="ego_view_depth_m",
        timestamp=1.0,
        rgb=np.zeros((3, 3, 3), dtype=np.uint8),
        depth_m=depth,
        depth_image=None,
        intrinsics=intrinsics,
        pose_mat=pose_mat,
    )

    snapshot = snapshot_from_sample(sample, frame_id=11)
    snapshot = update_snapshot_sam3(
        snapshot,
        prompt="plastic water bottle",
        results=[{"mask": mask, "box": [1, 1, 2, 2], "score": 0.9}],
    )

    payload = build_frame_payload(snapshot)

    grasp = payload["sam3"]["grasp"]
    assert grasp["kind"] == "front_policy_pinch_center"
    assert grasp["position_camera"] == pytest.approx([0.0, 0.0, 1.0])
    assert grasp["position_world"] == pytest.approx([0.5, -0.2, 1.1])
    assert grasp["quaternion_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert grasp["valid_depth_px"] == 1


def test_web_viewer_builds_live_frame_payload_with_rgb_and_depth() -> None:
    from types import SimpleNamespace

    from tools.g1_vision_bridge.web_viewer import build_frame_payload, snapshot_from_sample

    sample = SimpleNamespace(
        rgb_key="ego_view",
        depth_key="ego_view_depth_m",
        timestamp=12.5,
        rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        depth_m=np.array(
            [
                [0.0, 0.5, 1.0, 1.5],
                [2.0, np.nan, 2.5, 3.0],
                [0.2, 0.4, 0.6, 0.8],
            ],
            dtype=np.float32,
        ),
        depth_image=None,
    )
    sample.rgb[:, :, 0] = 200

    snapshot = snapshot_from_sample(sample, frame_id=7, received_time=100.0)
    payload = build_frame_payload(snapshot, max_depth_m=3.0)

    assert payload["ready"] is True
    assert payload["frame_id"] == 7
    assert payload["rgb_key"] == "ego_view"
    assert payload["depth_key"] == "ego_view_depth_m"
    assert payload["rgb_shape"] == [3, 4, 3]
    assert payload["depth_shape"] == [3, 4]
    assert payload["rgb_data_url"].startswith("data:image/jpeg;base64,")
    assert payload["depth_data_url"].startswith("data:image/jpeg;base64,")
    assert payload["depth_stats"]["valid_px"] == 10
    assert payload["depth_stats"]["min_m"] == pytest.approx(0.2)
    assert payload["depth_stats"]["max_m"] == pytest.approx(3.0)


def test_zmq_camera_client_decodes_raw_jpeg_msgpack_image() -> None:
    from tools.g1_vision_bridge.zmq_camera_client import decode_zmq_camera_payload

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[:, :, 1] = 180
    payload = msgpack.packb(
        {
            "timestamps": {"head": 12.5},
            "images": {"head": _jpeg_bytes_from_rgb(rgb)},
        },
        use_bin_type=True,
    )

    frame = decode_zmq_camera_payload(payload, camera_key="head")

    assert frame.camera_name == "head"
    assert frame.rgb.shape == rgb.shape
    assert frame.rgb.dtype == np.uint8
    assert frame.timestamp == 12.5


def test_zmq_camera_client_decodes_base64_nested_image() -> None:
    from tools.g1_vision_bridge.zmq_camera_client import decode_zmq_camera_payload

    rgb = np.zeros((3, 6, 3), dtype=np.uint8)
    rgb[:, :, 2] = 220
    encoded = base64.b64encode(_jpeg_bytes_from_rgb(rgb)).decode("ascii")
    payload = msgpack.packb(
        {
            "timestamp": 99.0,
            "images": {
                "robot0_robotview": {
                    "encoding": "jpeg_base64",
                    "data": encoded,
                }
            },
        },
        use_bin_type=True,
    )

    frame = decode_zmq_camera_payload(payload, camera_key="robot0_robotview")

    assert frame.camera_name == "robot0_robotview"
    assert frame.rgb.shape == rgb.shape
    assert frame.timestamp == 99.0


def test_g1_camera_api_decodes_composed_camera_rgb_jpeg_without_channel_swap() -> None:
    from capx.integrations.g1.vision import G1CameraApi, G1CameraConfig

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :] = [190, 90, 25]
    payload = msgpack.packb(
        {
            "timestamps": {"ego_view": 10.5},
            "images": {"ego_view": _g1_composed_camera_jpeg_bytes_from_rgb(rgb)},
            "depth_m": np.ones((8, 8), dtype=np.float32),
        },
        use_bin_type=True,
    )

    api = G1CameraApi(
        config=G1CameraConfig(
            rgb_key="ego_view",
            intrinsics=np.eye(3, dtype=np.float32),
            pose_mat=np.eye(4, dtype=np.float32),
        )
    )

    sample = api.decode_payload(payload)

    assert float(sample.rgb[:, :, 0].mean()) > float(sample.rgb[:, :, 2].mean()) + 100.0



def test_g1_camera_api_aligns_rgb_to_depth_frame_with_depth_to_rgb_extrinsics() -> None:
    from capx.integrations.g1.vision import G1CameraApi, G1CameraConfig, G1CameraSample

    # Depth pixel (u, v, z=1) projects into RGB pixel (u + 1, v) because
    # T_rgb_depth translates depth-frame points by +1 m in X and fx=1.
    rgb = np.zeros((2, 5, 3), dtype=np.uint8)
    for col in range(rgb.shape[1]):
        rgb[:, col, :] = [col * 40, 10, 200 - col * 20]
    depth_m = np.ones((2, 3), dtype=np.float32)
    k_depth = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    k_rgb = k_depth.copy()
    t_rgb_depth = np.eye(4, dtype=np.float32)
    t_rgb_depth[0, 3] = 1.0

    api = G1CameraApi(
        config=G1CameraConfig(
            intrinsics=k_depth,
            depth_intrinsics=k_depth,
            rgb_intrinsics=k_rgb,
            depth_to_rgb_mat=t_rgb_depth,
            align_rgb_to_depth=True,
            pose_mat=np.eye(4, dtype=np.float32),
        )
    )
    sample = G1CameraSample(
        rgb_key="ego_view",
        rgb=rgb,
        timestamp=1.0,
        raw_message={},
        depth_key="ego_view_depth_m",
        depth_m=depth_m,
    )

    obs = api.sample_to_capx_observation(sample)
    camera = obs["camera_top"]
    aligned_rgb = camera["images"]["rgb"]

    assert aligned_rgb.shape == (2, 3, 3)
    assert np.array_equal(aligned_rgb[:, 0, :], rgb[:, 1, :])
    assert np.array_equal(aligned_rgb[:, 1, :], rgb[:, 2, :])
    assert np.array_equal(aligned_rgb[:, 2, :], rgb[:, 3, :])
    assert np.array_equal(camera["intrinsics_matrix"], k_depth)
    assert np.allclose(camera["images"]["depth"][:, :, 0], depth_m)

def test_g1_camera_recv_payload_reports_timeout_for_zmq_again() -> None:
    import zmq

    from capx.integrations.g1.vision import G1CameraApi, G1CameraConfig

    class TimeoutSocket:
        def recv_multipart(self) -> list[bytes]:
            raise zmq.Again()

    api = G1CameraApi(
        config=G1CameraConfig(
            host="192.168.123.164",
            port=5555,
            receive_timeout_ms=1234,
            topic="",
        )
    )

    with pytest.raises(TimeoutError) as exc_info:
        api.recv_payload(TimeoutSocket())

    message = str(exc_info.value)
    assert "No G1 camera message received within 1234 ms" in message
    assert "tcp://192.168.123.164:5555" in message


def test_sam3_smoke_summary_records_scores_boxes_and_mask_area() -> None:
    from tools.g1_vision_bridge.smoke_test_sam3_displayer import summarize_sam3_results

    mask = np.zeros((3, 4), dtype=bool)
    mask[1:, 2:] = True
    results = [
        {
            "label": "displayer",
            "score": 0.875,
            "box": [1, 2, 3, 4],
            "mask": mask,
        }
    ]

    summary = summarize_sam3_results(results)

    assert summary == [
        {
            "index": 0,
            "label": "displayer",
            "score": 0.875,
            "box": [1.0, 2.0, 3.0, 4.0],
            "mask_area_px": 4,
        }
    ]


def test_sam3_smoke_server_command_uses_local_checkpoint() -> None:
    from tools.g1_vision_bridge.smoke_test_sam3_displayer import build_sam3_server_command

    command = build_sam3_server_command(
        host="127.0.0.1",
        port=8114,
        device="cuda",
        checkpoint_path="capx/model_weights/sam3/sam3.pt",
    )

    assert "--checkpoint-path" in command
    assert "capx/model_weights/sam3/sam3.pt" in command


def test_sam3_smoke_sets_localhost_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.g1_vision_bridge.smoke_test_sam3_displayer import ensure_localhost_no_proxy

    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.delenv("no_proxy", raising=False)

    ensure_localhost_no_proxy()

    no_proxy = os.environ["NO_PROXY"]
    assert "example.com" in no_proxy
    assert "127.0.0.1" in no_proxy
    assert "localhost" in no_proxy
    assert os.environ["no_proxy"] == no_proxy
