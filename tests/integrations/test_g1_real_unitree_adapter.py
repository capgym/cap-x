from types import SimpleNamespace
import inspect
import os
import threading
import time

import msgpack
import numpy as np
import pytest


def _jpeg_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return bytes(buf)


def _fake_low_cmd(num_motors: int = 35) -> SimpleNamespace:
    return SimpleNamespace(
        motor_cmd=[
            SimpleNamespace(mode=0, q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0)
            for _ in range(num_motors)
        ],
        mode_pr=0,
        mode_machine=0,
        crc=0,
    )


def _fake_low_state(num_motors: int = 35, *, mode_machine: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        mode_machine=mode_machine,
        motor_state=[
            SimpleNamespace(q=float(i) / 10.0)
            for i in range(num_motors)
        ],
    )


def _fake_hand_cmd(num_motors: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        motor_cmd=[
            SimpleNamespace(mode=0, q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0)
            for _ in range(num_motors)
        ],
        reserve=[0, 0, 0, 0],
    )


def _fake_hand_state(joints: np.ndarray | None = None) -> SimpleNamespace:
    values = np.zeros(7, dtype=np.float64) if joints is None else np.asarray(joints, dtype=np.float64)
    return SimpleNamespace(motor_state=[SimpleNamespace(q=float(q)) for q in values])


class _FakeGatewayArmActionMsg:
    def __init__(self) -> None:
        self.act = [0.0] * 14

    def encode(self):
        return tuple(self.act)


class _FakeGatewayLcm:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, object]] = []
        self.published: list[tuple[str, object]] = []

    def subscribe(self, channel: str, handler):
        token = (channel, handler)
        self.subscriptions.append(token)
        return token

    def unsubscribe(self, subscription) -> None:
        pass

    def publish(self, channel: str, payload) -> None:
        self.published.append((channel, payload))

    def fileno(self) -> int:
        return -1


def test_dex3_grasp_profiles_match_isaaclab_right_hand_motion_controller() -> None:
    from capx.integrations.g1.sdk import (
        G1_DEX3_DDS_HAND_JOINT_NAMES,
        G1_DEX3_HAND_JOINT_NAMES,
        dex3_grasp_joints,
        dex3_semantic_to_dds_joints,
    )

    assert G1_DEX3_HAND_JOINT_NAMES == (
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
    )
    assert G1_DEX3_DDS_HAND_JOINT_NAMES == (
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    )

    assert np.allclose(dex3_grasp_joints(trigger=0.0, squeeze=0.0), np.zeros(7))
    assert np.allclose(
        dex3_grasp_joints(trigger=1.0, squeeze=0.0),
        [-0.5, -0.4, -0.7, 1.0, 1.0, 0.0, 0.0],
    )
    assert np.allclose(
        dex3_grasp_joints(trigger=1.0, squeeze=1.0),
        [0.0, -0.4, -0.7, 1.0, 1.0, 1.0, 1.0],
    )
    assert np.allclose(
        dex3_semantic_to_dds_joints(dex3_grasp_joints(trigger=1.0, squeeze=0.0)),
        [-0.5, -0.4, -0.7, 0.0, 0.0, 1.0, 1.0],
    )


def test_dex3_hand_command_maps_semantic_hand_joints_to_unitree_dds_order() -> None:
    from capx.integrations.g1.sdk import G1Dex3HandBridge, dex3_grasp_joints

    bridge = G1Dex3HandBridge(dry_run=True, hand_cmd_factory=_fake_hand_cmd)
    target = dex3_grasp_joints(trigger=1.0, squeeze=0.0)

    cmd = bridge.build_hand_cmd(target)

    assert bridge.publisher_topic == "rt/dex3/right/cmd"
    assert bridge.state_topic == "rt/lf/dex3/right/state"
    assert [motor.mode for motor in cmd.motor_cmd] == [16 + i for i in range(7)]
    assert [motor.q for motor in cmd.motor_cmd] == pytest.approx(
        [-0.5, -0.4, -0.7, 0.0, 0.0, 1.0, 1.0]
    )
    assert [motor.kp for motor in cmd.motor_cmd] == pytest.approx([1.5] * 7)
    assert [motor.kd for motor in cmd.motor_cmd] == pytest.approx([0.1] * 7)


def test_dex3_hand_state_is_reported_in_semantic_hand_joint_order() -> None:
    from capx.integrations.g1.sdk import G1Dex3HandBridge

    bridge = G1Dex3HandBridge(dry_run=False, hand_cmd_factory=_fake_hand_cmd)
    bridge._handle_hand_state(_fake_hand_state(np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)))

    assert np.allclose(bridge.get_hand_joint_positions(), [0, 1, 2, 5, 6, 3, 4])


def test_lowcmd_command_maps_7_joint_vector_and_keeps_other_g1_joints() -> None:
    from capx.integrations.g1.sdk import (
        G1_ARM_MOTOR_INDICES,
        G1_LOWCMD_CONTROLLED_MOTOR_INDICES,
        G1ArmSdkBridge,
    )

    bridge = G1ArmSdkBridge(dry_run=True, low_cmd_factory=_fake_low_cmd)
    bridge._handle_low_state(_fake_low_state(mode_machine=5))
    target = np.linspace(-0.7, 0.7, 7)

    cmd = bridge.build_low_cmd(target)

    assert bridge.publisher_topic == "rt/lowcmd"
    assert G1_ARM_MOTOR_INDICES == tuple(range(22, 29))
    assert cmd.mode_pr == 0
    assert cmd.mode_machine == 5
    for motor_idx in G1_LOWCMD_CONTROLLED_MOTOR_INDICES:
        assert cmd.motor_cmd[motor_idx].mode == 1

    for i, motor_idx in enumerate(G1_ARM_MOTOR_INDICES):
        motor = cmd.motor_cmd[motor_idx]
        assert motor.q == pytest.approx(target[i])
        assert motor.dq == 0.0
        assert motor.kp == 40.0
        assert motor.kd == 1.0
        assert motor.tau == 0.0

    assert cmd.motor_cmd[0].q == pytest.approx(0.0)
    assert cmd.motor_cmd[15].q == pytest.approx(1.5)
    assert cmd.motor_cmd[21].q == pytest.approx(2.1)
    assert cmd.motor_cmd[29].mode == 0


def test_lowcmd_command_rejects_non_7_joint_vectors() -> None:
    from capx.integrations.g1.sdk import G1ArmSdkBridge

    bridge = G1ArmSdkBridge(dry_run=True, low_cmd_factory=_fake_low_cmd)

    with pytest.raises(ValueError, match="7"):
        bridge.build_low_cmd(np.zeros(14))


def test_lowcmd_real_mode_refuses_to_publish_before_lowstate() -> None:
    from capx.integrations.g1.sdk import G1ArmSdkBridge

    class FakePublisher:
        def __init__(self) -> None:
            self.writes = 0

        def Write(self, cmd):
            self.writes += 1
            return True

    publisher = FakePublisher()
    bridge = G1ArmSdkBridge(dry_run=False, low_cmd_factory=_fake_low_cmd)
    bridge._connected = True
    bridge._publisher = publisher
    target = np.linspace(-0.3, 0.3, 7)

    with pytest.raises(RuntimeError, match="lowstate"):
        bridge.publish_joints(target)

    assert publisher.writes == 0


def test_gateway_arm_action_maps_right_arm_and_preserves_left_arm() -> None:
    from capx.integrations.g1.gateway import G1GatewayArmActionBridge

    bridge = G1GatewayArmActionBridge(
        dry_run=True,
        arm_message_type=_FakeGatewayArmActionMsg,
    )
    q = np.arange(29, dtype=np.float64) / 10.0
    bridge._handle_body_state(
        "body_control_data",
        SimpleNamespace(q=q, qd=np.zeros(29, dtype=np.float64), timestamp_us=123),
    )
    target = np.linspace(-0.7, 0.7, 7)

    msg = bridge.build_arm_message(target)

    assert bridge.has_low_state
    assert msg.act[:7] == pytest.approx(q[15:22])
    assert msg.act[7:14] == pytest.approx(target)
    assert np.allclose(bridge.get_arm_joint_positions(), q[22:29])


def test_gateway_arm_action_real_mode_refuses_before_body_control_data() -> None:
    from capx.integrations.g1.gateway import G1GatewayArmActionBridge

    fake_lcm = _FakeGatewayLcm()
    bridge = G1GatewayArmActionBridge(
        dry_run=False,
        lcm_client=fake_lcm,
        arm_message_type=_FakeGatewayArmActionMsg,
        body_state_message_type=SimpleNamespace,
        imu_message_type=SimpleNamespace,
    )

    with pytest.raises(RuntimeError, match="body_control_data"):
        bridge.publish_joints(np.linspace(-0.2, 0.2, 7))

    assert fake_lcm.published == []


def test_gateway_arm_action_publishes_lcm_message() -> None:
    from capx.integrations.g1.gateway import G1GatewayArmActionBridge

    fake_lcm = _FakeGatewayLcm()
    bridge = G1GatewayArmActionBridge(
        dry_run=False,
        lcm_client=fake_lcm,
        arm_message_type=_FakeGatewayArmActionMsg,
        body_state_message_type=SimpleNamespace,
        imu_message_type=SimpleNamespace,
    )
    q = np.arange(29, dtype=np.float64) / 10.0
    bridge._handle_body_state(
        "body_control_data",
        SimpleNamespace(q=q, qd=np.zeros(29, dtype=np.float64), timestamp_us=123),
    )
    target = np.linspace(-0.4, 0.4, 7)

    assert bridge.publish_joints(target) is True

    assert [item[0] for item in fake_lcm.subscriptions] == [
        "body_control_data",
        "state_estimator_data",
    ]
    assert fake_lcm.published == [("arm_action", tuple(bridge.last_command.act))]
    assert bridge.last_command.act[:7] == pytest.approx(q[15:22])
    assert bridge.last_command.act[7:14] == pytest.approx(target)


def test_gateway_arm_action_serializes_lcm_io_across_threads() -> None:
    from capx.integrations.g1.gateway import G1GatewayArmActionBridge

    class ConcurrencyCheckingLcm(_FakeGatewayLcm):
        def __init__(self) -> None:
            super().__init__()
            self.reader_fd, self.writer_fd = os.pipe()
            os.write(self.writer_fd, b"x")
            self.handle_started = threading.Event()
            self.io_guard = threading.Lock()

        def fileno(self) -> int:
            return self.reader_fd

        def handle(self) -> None:
            if not self.io_guard.acquire(blocking=False):
                raise RuntimeError(
                    "only one thread is allowed to call LCM.handle() or "
                    "LCM.handle_timeout() at a time"
                )
            try:
                self.handle_started.set()
                time.sleep(0.05)
            finally:
                self.io_guard.release()

        def publish(self, channel: str, payload) -> None:
            if not self.io_guard.acquire(blocking=False):
                raise RuntimeError("concurrent LCM I/O")
            try:
                super().publish(channel, payload)
            finally:
                self.io_guard.release()

        def close(self) -> None:
            os.close(self.reader_fd)
            os.close(self.writer_fd)

    fake_lcm = ConcurrencyCheckingLcm()
    bridge = G1GatewayArmActionBridge(
        dry_run=False,
        lcm_client=fake_lcm,
        arm_message_type=_FakeGatewayArmActionMsg,
        body_state_message_type=SimpleNamespace,
        imu_message_type=SimpleNamespace,
    )
    q = np.arange(29, dtype=np.float64) / 10.0
    bridge._handle_body_state(
        "body_control_data",
        SimpleNamespace(q=q, qd=np.zeros(29, dtype=np.float64), timestamp_us=123),
    )
    target = np.linspace(-0.4, 0.4, 7)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            bridge.publish_joints(target)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=publish)
    first.start()
    assert fake_lcm.handle_started.wait(timeout=1.0)
    second = threading.Thread(target=publish)
    second.start()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    fake_lcm.close()
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(fake_lcm.published) == 2


def test_lowcmd_connect_releases_motion_mode_before_publishing() -> None:
    from capx.integrations.g1.sdk import G1ArmSdkBridge

    events: list[tuple[str, object]] = []

    class FakeMotionSwitcherClient:
        def __init__(self) -> None:
            self.checks = 0

        def SetTimeout(self, timeout: float) -> None:
            events.append(("timeout", timeout))

        def Init(self) -> None:
            events.append(("motion_init", None))

        def CheckMode(self):
            self.checks += 1
            if self.checks == 1:
                return 0, {"name": "ai"}
            return 0, {"name": ""}

        def ReleaseMode(self):
            events.append(("release", None))
            return 0, None

    class FakePublisher:
        def __init__(self, topic, msg_type):
            events.append(("publisher", topic))

        def Init(self):
            events.append(("publisher_init", None))

    class FakeSubscriber:
        def __init__(self, topic, msg_type):
            events.append(("subscriber", topic))

        def Init(self, handler, queue_size):
            events.append(("subscriber_init", queue_size))

    bridge = G1ArmSdkBridge(dry_run=False, low_cmd_factory=_fake_low_cmd)
    bridge._load_sdk = lambda: {  # type: ignore[method-assign]
        "ChannelFactoryInitialize": lambda domain, interface: events.append(("factory", interface)),
        "ChannelPublisher": FakePublisher,
        "ChannelSubscriber": FakeSubscriber,
        "LowCmd_": object,
        "LowState_": object,
        "low_cmd_factory": _fake_low_cmd,
        "CRC": lambda: SimpleNamespace(Crc=lambda cmd: 123),
        "MotionSwitcherClient": FakeMotionSwitcherClient,
    }

    bridge.connect()

    assert ("factory", "enx6c1ff7c1192d") in events
    assert ("release", None) in events
    assert ("publisher", "rt/lowcmd") in events


def test_g1_real_low_level_propagates_dry_run_to_custom_bridge() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.dry_run = False

        def connect(self) -> None:
            raise AssertionError("dry-run low-level env must not connect custom bridge")

        def publish_joints(self, joints: np.ndarray) -> bool:
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return np.zeros(7, dtype=np.float64)

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=True,
        privileged=False,
        enable_render=False,
        viser_debug=False,
    )

    assert env.dry_run is True
    assert bridge.dry_run is True


def test_g1_real_low_level_dry_run_publishes_7_joint_right_arm_target() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []

        def connect(self) -> None:
            raise AssertionError("dry-run low-level env must not connect to DDS")

        def publish_joints(self, joints: np.ndarray) -> None:
            self.published.append(np.asarray(joints, dtype=np.float64).copy())

        def get_arm_joint_positions(self) -> np.ndarray:
            if self.published:
                return self.published[-1].copy()
            return np.zeros(7, dtype=np.float64)

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=True,
        privileged=False,
        enable_render=False,
        viser_debug=False,
    )
    target = np.linspace(-0.4, 0.4, 7)

    env.move_to_joints_blocking(target)
    env._set_gripper(0.0)

    assert np.allclose(bridge.published[-1], target)
    assert np.allclose(env.get_observation()["robot_joint_pos"], target)
    assert env.get_observation()["gripper_type"] == "rubber_hand"
    assert "left" not in env.latest_action
    assert "right" not in env.latest_action


def test_g1_real_low_level_runs_configured_pregrasp_side_pose_once() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []

        def connect(self) -> None:
            raise AssertionError("dry-run low-level env must not connect to DDS")

        def publish_joints(self, joints: np.ndarray) -> bool:
            self.published.append(np.asarray(joints, dtype=np.float64).copy())
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            if self.published:
                return self.published[-1].copy()
            return np.zeros(7, dtype=np.float64)

        def close(self) -> None:
            pass

    safe_joints = np.array([0.0, -1.25, 0.0, 1.1, 0.0, 0.0, 0.0], dtype=np.float64)
    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=True,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        pregrasp_side_joints=safe_joints.tolist(),
        pregrasp_side_before_first_pose=True,
    )

    assert env.move_to_pregrasp_side_pose_once() is True
    assert env.move_to_pregrasp_side_pose_once() is False
    assert len(bridge.published) == 1
    assert np.allclose(bridge.published[0], safe_joints)


def test_g1_real_low_level_records_video_frames_from_network_observations() -> None:
    from types import SimpleNamespace

    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeArmBridge:
        def connect(self) -> None:
            raise AssertionError("dry-run low-level env must not connect arm DDS")

        def publish_joints(self, joints: np.ndarray) -> bool:
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return np.zeros(7, dtype=np.float64)

        def close(self) -> None:
            pass

    env = G1RealLowLevel(
        sdk_bridge=FakeArmBridge(),
        dry_run=True,
        privileged=False,
        enable_render=False,
        viser_debug=False,
    )
    env.enable_video_capture(True, clear=True)
    assert env.get_video_frames() == []

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[:, :, 1] = 255
    env.low_level_server = SimpleNamespace(
        latest_observation={
            "robot0_robotview": {
                "images": {"rgb": rgb},
            }
        },
        latest_action=None,
    )

    env.get_observation()

    frames = env.get_video_frames()
    assert len(frames) == 1
    assert frames[0].shape == (480, 640, 3)


def test_g1_real_low_level_dry_run_publishes_dex3_open_close_and_index_pinch() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel
    from capx.integrations.g1.sdk import dex3_grasp_joints

    class FakeArmBridge:
        def connect(self) -> None:
            raise AssertionError("dry-run low-level env must not connect arm DDS")

        def publish_joints(self, joints: np.ndarray) -> bool:
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return np.zeros(7, dtype=np.float64)

        def close(self) -> None:
            pass

    class FakeHandBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []

        def connect(self) -> None:
            raise AssertionError("dry-run low-level env must not connect hand DDS")

        def publish_hand_joints(self, joints: np.ndarray) -> bool:
            self.published.append(np.asarray(joints, dtype=np.float64).copy())
            return True

        def get_hand_joint_positions(self) -> np.ndarray:
            if self.published:
                return self.published[-1].copy()
            return np.zeros(7, dtype=np.float64)

        def close(self) -> None:
            pass

    hand_bridge = FakeHandBridge()
    env = G1RealLowLevel(
        sdk_bridge=FakeArmBridge(),
        hand_bridge=hand_bridge,
        dry_run=True,
        privileged=False,
        enable_render=False,
        viser_debug=False,
    )

    env._set_gripper(1.0)
    env._set_gripper(0.0)
    env.close_dex3_index_pinch()

    assert np.allclose(hand_bridge.published[0], dex3_grasp_joints(trigger=0.0, squeeze=0.0))
    assert np.allclose(hand_bridge.published[1], dex3_grasp_joints(trigger=1.0, squeeze=1.0))
    assert np.allclose(hand_bridge.published[2], dex3_grasp_joints(trigger=1.0, squeeze=0.0))
    assert np.allclose(env.get_observation()["gripper_joint_pos"], hand_bridge.published[-1])


def test_g1_real_low_level_real_mode_republishes_until_state_reaches_target() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.connected = False
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            self.connected = True

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            self.published.append(np.asarray(joints, dtype=np.float64).copy())
            if len(self.published) >= 3:
                self.state = self.published[-1].copy()
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=0.0,
    )
    target = np.linspace(-0.2, 0.2, 7)

    env.move_to_joints_blocking(target, tolerance=0.01, max_steps=10)

    assert bridge.connected
    assert len(bridge.published) >= 3
    assert np.allclose(env.get_observation()["robot_joint_pos"], target)


def test_g1_real_low_level_real_mode_holds_last_arm_target_between_actions() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.connected = False
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            self.connected = True

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            target = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(target)
            self.state = target.copy()
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=0.01,
        joint_interpolation_steps=0,
    )
    target = np.linspace(-0.2, 0.2, 7)

    env.move_to_joints_blocking(target, tolerance=0.01, max_steps=10)
    published_after_move = len(bridge.published)
    time.sleep(0.05)
    env.close()

    assert len(bridge.published) > published_after_move
    assert np.allclose(bridge.published[-1], target)


def test_g1_real_low_level_real_mode_immediately_republishes_hold_target_after_arm_move() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            pass

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            target = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(target)
            self.state = target.copy()
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=10.0,
        joint_interpolation_steps=0,
    )
    target = np.linspace(-0.2, 0.2, 7)

    env.move_to_joints_blocking(target, tolerance=0.01, max_steps=10)
    env.close()

    assert len(bridge.published) >= 2
    assert np.allclose(bridge.published[-2], target)
    assert np.allclose(bridge.published[-1], target)


def test_g1_real_low_level_hand_motion_refreshes_last_arm_hold_target_immediately() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel
    from capx.integrations.g1.sdk import dex3_grasp_joints

    class FakeArmBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            pass

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            target = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(target)
            self.state = target.copy()
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    class FakeHandBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            pass

        def wait_for_hand_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_hand_joints(self, joints: np.ndarray) -> bool:
            target = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(target)
            self.state = target.copy()
            return True

        def get_hand_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    arm_bridge = FakeArmBridge()
    env = G1RealLowLevel(
        sdk_bridge=arm_bridge,
        hand_bridge=FakeHandBridge(),
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=10.0,
        joint_interpolation_steps=0,
    )
    arm_target = np.linspace(-0.2, 0.2, 7)
    env.move_to_joints_blocking(arm_target, tolerance=0.01, max_steps=10)
    arm_bridge.published.clear()

    env.move_hand_to_joints_blocking(dex3_grasp_joints(trigger=0.0, squeeze=0.0))
    env.close()

    assert len(arm_bridge.published) >= 2
    assert all(np.allclose(target, arm_target) for target in arm_bridge.published)


def test_g1_real_low_level_real_mode_keeps_hold_active_and_tracks_motion_waypoints() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            pass

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            target = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(target)
            self.state = target.copy()
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=0.0,
        arm_hold_publish_period=10.0,
        joint_interpolation_steps=3,
    )
    paused_calls: list[bool] = []
    hold_targets: list[tuple[np.ndarray, bool]] = []
    original_pause = env._set_arm_hold_paused
    original_set_hold = env._set_arm_hold_target

    def spy_pause(paused: bool) -> None:
        paused_calls.append(bool(paused))
        original_pause(paused)

    def spy_set_hold(target: np.ndarray, *, publish_now: bool = True) -> None:
        hold_targets.append((np.asarray(target, dtype=np.float64).copy(), bool(publish_now)))
        original_set_hold(target, publish_now=publish_now)

    env._set_arm_hold_paused = spy_pause
    env._set_arm_hold_target = spy_set_hold
    target = np.linspace(-0.3, 0.3, 7)

    env.move_to_joints_blocking(target, tolerance=0.01, max_steps=10)
    env.close()

    expected_waypoints = np.linspace(np.zeros(7, dtype=np.float64), target, 4)[1:]
    assert True not in paused_calls
    assert len(hold_targets) >= 4
    assert np.allclose(np.asarray([item[0] for item in hold_targets[:3]]), expected_waypoints)
    assert all(item[1] is False for item in hold_targets[:3])
    assert np.allclose(hold_targets[-1][0], target)
    assert hold_targets[-1][1] is True


def test_g1_real_low_level_streams_joint_trajectory_without_blocking_between_waypoints() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            pass

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            target = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(target)
            self.state = target.copy()
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=0.0,
        arm_hold_publish_period=10.0,
        joint_interpolation_steps=50,
    )
    paused_calls: list[bool] = []
    hold_targets: list[tuple[np.ndarray, bool]] = []
    original_pause = env._set_arm_hold_paused
    original_set_hold = env._set_arm_hold_target

    def spy_pause(paused: bool) -> None:
        paused_calls.append(bool(paused))
        original_pause(paused)

    def spy_set_hold(target: np.ndarray, *, publish_now: bool = True) -> None:
        hold_targets.append((np.asarray(target, dtype=np.float64).copy(), bool(publish_now)))
        original_set_hold(target, publish_now=publish_now)

    env._set_arm_hold_paused = spy_pause
    env._set_arm_hold_target = spy_set_hold
    trajectory = np.stack(
        [
            np.linspace(0.0, 0.1, 7),
            np.linspace(0.1, 0.2, 7),
            np.linspace(0.2, 0.3, 7),
        ],
        axis=0,
    )

    env.execute_joint_trajectory_streaming(trajectory, dt=0.0)
    env.close()

    assert True not in paused_calls
    assert np.allclose(np.asarray(bridge.published[:3]), trajectory)
    assert np.allclose(env.get_observation()["robot_joint_pos"], trajectory[-1])
    assert len(hold_targets) >= 4
    assert all(item[1] is False for item in hold_targets[:3])
    assert np.allclose([item[0] for item in hold_targets[:3]], trajectory)
    assert np.allclose(hold_targets[-1][0], trajectory[-1])
    assert hold_targets[-1][1] is True


def test_g1_real_low_level_streaming_upsamples_large_joint_steps() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            pass

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            target = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(target)
            self.state = target.copy()
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=0.0,
        arm_hold_publish_period=10.0,
        joint_interpolation_steps=50,
    )
    trajectory = np.asarray([[0.35] * 7], dtype=np.float64)

    env.execute_joint_trajectory_streaming(trajectory, dt=0.0, max_joint_step=0.1)
    env.close()

    published = np.asarray(bridge.published, dtype=np.float64)
    assert published.shape[0] > trajectory.shape[0]
    deltas = np.diff(np.vstack([np.zeros(7, dtype=np.float64), published]), axis=0)
    assert np.max(np.abs(deltas)) <= 0.100001
    assert np.allclose(published[-1], trajectory[-1])


def test_g1_real_low_level_real_mode_raises_when_lowcmd_publish_fails() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def connect(self) -> None:
            pass

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            return False

        def get_arm_joint_positions(self) -> np.ndarray:
            return np.zeros(7, dtype=np.float64)

        def close(self) -> None:
            pass

    env = G1RealLowLevel(
        sdk_bridge=FakeBridge(),
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=0.0,
        joint_interpolation_steps=0,
    )

    with pytest.raises(RuntimeError, match="rt/lowcmd publish failed"):
        env.move_to_joints_blocking(np.linspace(-0.2, 0.2, 7))


def test_g1_real_low_level_real_mode_interpolates_joint_target_before_waiting() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    class FakeBridge:
        def __init__(self) -> None:
            self.published: list[np.ndarray] = []
            self.state = np.zeros(7, dtype=np.float64)

        def connect(self) -> None:
            pass

        def wait_for_low_state(self, timeout_s: float = 3.0) -> bool:
            return True

        def publish_joints(self, joints: np.ndarray) -> bool:
            self.state = np.asarray(joints, dtype=np.float64).copy()
            self.published.append(self.state.copy())
            return True

        def get_arm_joint_positions(self) -> np.ndarray:
            return self.state.copy()

        def close(self) -> None:
            pass

    bridge = FakeBridge()
    env = G1RealLowLevel(
        sdk_bridge=bridge,
        dry_run=False,
        privileged=False,
        enable_render=False,
        viser_debug=False,
        action_publish_period=0.0,
        joint_interpolation_steps=5,
    )
    target = np.linspace(-0.5, 0.5, 7)

    env.move_to_joints_blocking(target, tolerance=0.01, max_steps=10)

    expected = np.linspace(np.zeros(7, dtype=np.float64), target, 6)[1:]
    assert len(bridge.published) == 6
    assert np.allclose(np.asarray(bridge.published[:5]), expected)
    assert np.allclose(bridge.published[-1], target)
    assert np.allclose(env.get_observation()["robot_joint_pos"], target)


def test_g1_real_low_level_defaults_to_requested_network_interface() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    env = G1RealLowLevel(
        dry_run=True,
        privileged=False,
        enable_render=False,
        viser_debug=False,
    )

    assert env.sdk_bridge.network_interface == "enx6c1ff7c1192d"


def test_g1_real_control_api_exposes_single_arm_joint_signature() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    signature = inspect.signature(G1RealControlApi.move_to_joints)

    assert list(signature.parameters) == ["self", "joints"]


def test_g1_real_control_api_goto_pose_runs_pregrasp_side_pose_before_ik() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    class FakeEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[str, np.ndarray | None]] = []

        def move_to_pregrasp_side_pose_once(self, *, force: bool = False) -> bool:
            self.calls.append(("pregrasp", None))
            return True

        def move_to_joints_blocking(self, joints: np.ndarray) -> None:
            self.calls.append(("joints", np.asarray(joints, dtype=np.float64).copy()))

    env = FakeEnv()
    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = env
    api._TCP_OFFSET = np.zeros(3, dtype=np.float64)
    api.cfg = None
    api._webui_enabled = False
    api.ik_solve_fn = lambda target_pose_wxyz_xyz, prev_cfg=None: np.ones(7, dtype=np.float64)

    api.goto_pose(np.array([0.4, -0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))

    assert env.calls[0] == ("pregrasp", None)
    assert env.calls[1][0] == "joints"


def test_g1_real_control_api_first_ik_uses_current_arm_joints_as_seed() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    current = np.array([0.0, -1.25, 0.0, 1.1, 0.0, 0.0, 0.0], dtype=np.float64)

    class FakeEnv:
        def get_current_arm_joints(self):
            return current.copy()

    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = FakeEnv()
    api.cfg = None
    seen_prev_cfg: list[np.ndarray | None] = []

    def fake_ik(target_pose_wxyz_xyz, prev_cfg=None):
        seen_prev_cfg.append(None if prev_cfg is None else np.asarray(prev_cfg, dtype=np.float64).copy())
        return np.ones(7, dtype=np.float64)

    api.ik_solve_fn = fake_ik

    api._solve_ik(np.array([0.4, 0.1, 0.3], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))

    assert len(seen_prev_cfg) == 1
    assert np.allclose(seen_prev_cfg[0], current)


def test_g1_palm_pose_from_pinch_center_pose_applies_closed_hand_offset() -> None:
    from capx.integrations.g1.grasp import palm_pose_from_pinch_center_pose

    pinch_pos = np.array([0.5, -0.2, 0.3], dtype=np.float64)
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    palm_pos, palm_quat = palm_pose_from_pinch_center_pose(
        pinch_pos,
        quat_wxyz,
        pinch_center_offset=np.array([0.08, 0.01, -0.02], dtype=np.float64),
    )

    assert np.allclose(palm_pos, [0.42, -0.21, 0.32])
    assert np.allclose(palm_quat, quat_wxyz)


def test_g1_real_control_api_moves_pinch_center_by_targeting_palm_link() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    class FakeEnv:
        def move_to_pregrasp_side_pose_once(self, *, force: bool = False) -> bool:
            return False

        def move_to_joints_blocking(self, joints: np.ndarray) -> None:
            pass

    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = FakeEnv()
    api._TCP_OFFSET = np.zeros(3, dtype=np.float64)
    api.cfg = None
    api._webui_enabled = False
    api._closed_pinch_center_offset = lambda: np.array([0.08, 0.01, -0.02], dtype=np.float64)
    ik_targets: list[np.ndarray] = []

    def fake_ik(target_pose_wxyz_xyz, prev_cfg=None):
        ik_targets.append(np.asarray(target_pose_wxyz_xyz, dtype=np.float64).copy())
        return np.ones(7, dtype=np.float64)

    api.ik_solve_fn = fake_ik

    api.move_pinch_center_to_pose(
        np.array([0.5, -0.2, 0.3], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert np.allclose(ik_targets[-1][-3:], [0.42, -0.21, 0.32])


def test_g1_real_control_api_sample_grasp_center_pose_reverses_legacy_local_z_offset(monkeypatch) -> None:
    from capx.integrations.franka.control import FrankaControlApi
    from capx.integrations.g1.control import G1RealControlApi

    def fake_sample_grasp_pose(self, object_name: str):
        assert object_name == "plastic water bottle"
        return np.array([0.0, 0.0, 0.12], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(FrankaControlApi, "sample_grasp_pose", fake_sample_grasp_pose)

    api = G1RealControlApi.__new__(G1RealControlApi)
    api._webui_enabled = False

    pos, quat = api.sample_grasp_center_pose("plastic water bottle")

    assert np.allclose(pos, [0.0, 0.0, 0.0])
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_g1_real_control_api_exposes_trigger_squeeze_hand_control() -> None:
    from capx.integrations.g1.control import G1RealControlApi
    from capx.integrations.g1.sdk import dex3_grasp_joints

    class FakeEnv:
        def __init__(self) -> None:
            self.hand_targets: list[np.ndarray] = []

        def move_hand_to_joints_blocking(self, joints: np.ndarray) -> None:
            self.hand_targets.append(np.asarray(joints, dtype=np.float64).copy())

    env = FakeEnv()
    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = env
    api._webui_enabled = False

    api.set_gripper_trigger_squeeze(trigger=0.25, squeeze=0.75)

    assert np.allclose(env.hand_targets[-1], dex3_grasp_joints(trigger=0.25, squeeze=0.75))
    assert "grasp_at_pinch_center" in api.functions()
    assert "set_gripper_trigger_squeeze" in api.functions()


def test_g1_front_pregrasp_default_retreats_20cm_in_world_minus_x() -> None:
    from capx.integrations.g1.grasp import front_pregrasp_pinch_center_pose

    pre_pos, pre_quat = front_pregrasp_pinch_center_pose(
        np.array([0.5, 0.1, 0.2], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert np.allclose(pre_pos, [0.3, 0.1, 0.2])
    assert np.allclose(pre_quat, [1.0, 0.0, 0.0, 0.0])



def test_g1_front_pregrasp_defaults_to_vertical_front_palm_orientation() -> None:
    from capx.integrations.g1.grasp import front_pregrasp_pinch_center_pose

    pre_pos, pre_quat = front_pregrasp_pinch_center_pose(
        np.array([0.5, 0.1, 0.2], dtype=np.float64),
        None,
    )

    assert np.allclose(pre_pos, [0.3, 0.1, 0.2])
    assert np.allclose(pre_quat, [1.0, 0.0, 0.0, 0.0])


def test_g1_front_policy_pinch_target_uses_contact_xy_visual_z_and_vertical_front_palm_orientation() -> None:
    from scipy.spatial.transform import Rotation as SciRotation

    from capx.integrations.g1.grasp import front_policy_pinch_target_pose

    target, quat = front_policy_pinch_target_pose(
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        visual_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, np.nan],
            ],
            dtype=np.float64,
        ),
        contact_points=np.array(
            [
                [0.7, 0.8, 0.1],
                [0.9, 1.0, 0.2],
            ],
            dtype=np.float64,
        ),
    )

    assert np.allclose(target, [0.8, 0.9, 0.2])
    rot = SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    assert np.allclose(rot.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    assert np.allclose(rot.apply([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(rot.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])


def test_g1_real_control_api_sample_grasp_center_pose_uses_front_policy_target(monkeypatch) -> None:
    from capx.integrations.franka.control import FrankaControlApi
    from capx.integrations.g1.control import G1RealControlApi

    class FakeEnv:
        def __init__(self) -> None:
            self.cube_points = None
            self.grasp_contact_pts = None
            self.grasp_scores = None

        def get_observation(self):
            return {
                "robot0_robotview": {
                    "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                }
            }

    def fake_sample_grasp_pose(self, object_name: str):
        assert object_name == "plastic water bottle"
        self._env.cube_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4],
            ],
            dtype=np.float64,
        )
        self._env.grasp_contact_pts = np.array(
            [
                [
                    [0.7, 0.8, 0.1],
                    [0.9, 1.0, 0.2],
                ]
            ],
            dtype=np.float64,
        )
        self._env.grasp_scores = np.array([1.0], dtype=np.float64)
        return np.array([0.1, 0.2, 0.42], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(FrankaControlApi, "sample_grasp_pose", fake_sample_grasp_pose)

    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = FakeEnv()
    api._webui_enabled = False

    pos, quat = api.sample_grasp_center_pose("plastic water bottle")

    assert np.allclose(pos, [0.8, 0.9, 0.2])
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_g1_real_control_api_grasp_at_pinch_center_forces_front_policy_vertical_orientation() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    api = G1RealControlApi.__new__(G1RealControlApi)
    api._webui_enabled = False
    hand_commands: list[tuple[float, float]] = []
    pregrasp_calls: list[tuple[np.ndarray, np.ndarray, float]] = []
    line_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    move_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_hand(trigger: float, squeeze: float) -> None:
        hand_commands.append((float(trigger), float(squeeze)))

    def fake_pregrasp(position, quaternion_wxyz=None, pregrasp_distance=0.20) -> None:
        pregrasp_calls.append(
            (
                np.asarray(position, dtype=np.float64).copy(),
                np.asarray(quaternion_wxyz, dtype=np.float64).copy(),
                float(pregrasp_distance),
            )
        )

    def fake_line(start_position, end_position, quaternion_wxyz=None, **kwargs) -> None:
        line_calls.append(
            (
                np.asarray(start_position, dtype=np.float64).copy(),
                np.asarray(end_position, dtype=np.float64).copy(),
                np.asarray(quaternion_wxyz, dtype=np.float64).copy(),
            )
        )

    def fake_move(position, quaternion_wxyz=None, z_approach=0.0) -> None:
        move_calls.append(
            (
                np.asarray(position, dtype=np.float64).copy(),
                np.asarray(quaternion_wxyz, dtype=np.float64).copy(),
            )
        )

    api.set_gripper_trigger_squeeze = fake_hand
    api.move_to_pinch_pregrasp = fake_pregrasp
    api.move_pinch_center_horizontal_line = fake_line
    api.move_pinch_center_to_pose = fake_move

    api.grasp_at_pinch_center(
        np.array([0.4, 0.2, 0.3], dtype=np.float64),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        lift_dz=0.15,
    )

    assert np.allclose(pregrasp_calls[0][1], [1.0, 0.0, 0.0, 0.0])
    assert pregrasp_calls[0][2] == pytest.approx(0.20)
    assert np.allclose(line_calls[0][0], [0.2, 0.2, 0.3])
    assert np.allclose(line_calls[0][1], [0.4, 0.2, 0.3])
    assert np.allclose(line_calls[0][2], [1.0, 0.0, 0.0, 0.0])
    assert all(np.allclose(quat, [1.0, 0.0, 0.0, 0.0]) for _, quat in move_calls)
    assert hand_commands == [(1.0, 1.0), (0.0, 0.0), (1.0, 1.0)]


def test_g1_real_control_api_grasp_at_pinch_center_uses_single_streamed_horizontal_approach() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    api = G1RealControlApi.__new__(G1RealControlApi)
    api._webui_enabled = False
    line_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    move_calls: list[np.ndarray] = []

    def fake_hand(trigger: float, squeeze: float) -> None:
        pass

    def fake_line(start_position, end_position, quaternion_wxyz=None, **kwargs) -> None:
        line_calls.append(
            (
                np.asarray(start_position, dtype=np.float64).copy(),
                np.asarray(end_position, dtype=np.float64).copy(),
                np.asarray(quaternion_wxyz, dtype=np.float64).copy(),
            )
        )

    def fake_move(position, quaternion_wxyz=None, z_approach=0.0) -> None:
        move_calls.append(np.asarray(position, dtype=np.float64).copy())

    api.set_gripper_trigger_squeeze = fake_hand
    api.move_pinch_center_horizontal_line = fake_line
    api.move_pinch_center_to_pose = fake_move

    target = np.array([0.4, 0.2, 0.3], dtype=np.float64)
    api.grasp_at_pinch_center(
        target,
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        pregrasp_distance=0.20,
        lift_dz=0.0,
    )

    assert len(move_calls) == 1
    assert np.allclose(move_calls[0], [0.2, 0.2, 0.3])
    assert len(line_calls) == 1
    assert np.allclose(line_calls[0][0], [0.2, 0.2, 0.3])
    assert np.allclose(line_calls[0][1], target)
    assert np.allclose(line_calls[0][2], [1.0, 0.0, 0.0, 0.0])


def test_g1_real_control_api_horizontal_pinch_line_keeps_z_and_streams_once() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    class FakeEnv:
        def __init__(self) -> None:
            self.trajectories: list[np.ndarray] = []
            self.kwargs: list[dict[str, object]] = []

        def execute_joint_trajectory_streaming(self, trajectory, **kwargs) -> None:
            self.trajectories.append(np.asarray(trajectory, dtype=np.float64).copy())
            self.kwargs.append(dict(kwargs))

    env = FakeEnv()
    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = env
    api._webui_enabled = False
    api._TCP_OFFSET = np.zeros(3, dtype=np.float64)
    api.cfg = None
    api._closed_pinch_center_offset = lambda: np.zeros(3, dtype=np.float64)
    ik_targets: list[np.ndarray] = []

    def fake_ik(target_pose_wxyz_xyz, prev_cfg=None):
        target = np.asarray(target_pose_wxyz_xyz, dtype=np.float64)
        ik_targets.append(target.copy())
        pos = target[-3:]
        return np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    api.ik_solve_fn = fake_ik

    api.move_pinch_center_horizontal_line(
        np.array([0.2, 0.1, 0.35], dtype=np.float64),
        np.array([0.4, 0.1, 0.35], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        num_waypoints=4,
    )

    ik_positions = [target[-3:] for target in ik_targets]
    ik_quats = [target[:4] for target in ik_targets]
    assert len(env.trajectories) == 1
    assert env.trajectories[0].shape == (4, 7)
    assert env.kwargs[0]["max_joint_step"] is None
    assert env.kwargs[0]["hold_final_seconds"] == pytest.approx(0.4)
    assert np.allclose([pos[1] for pos in ik_positions], [0.1, 0.1, 0.1, 0.1])
    assert np.allclose([pos[2] for pos in ik_positions], [0.35, 0.35, 0.35, 0.35])
    assert all(np.allclose(quat, [1.0, 0.0, 0.0, 0.0]) for quat in ik_quats)
    assert np.all(np.diff([pos[0] for pos in ik_positions]) > 0.0)
    assert np.allclose(env.trajectories[0][-1, :3], [0.4, 0.1, 0.35])


def test_g1_real_control_api_horizontal_pinch_line_defaults_to_dense_cartesian_ik_waypoints() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    class FakeEnv:
        def __init__(self) -> None:
            self.trajectory: np.ndarray | None = None

        def execute_joint_trajectory_streaming(self, trajectory, **kwargs) -> None:
            self.trajectory = np.asarray(trajectory, dtype=np.float64).copy()

    env = FakeEnv()
    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = env
    api._webui_enabled = False
    api._TCP_OFFSET = np.zeros(3, dtype=np.float64)
    api.cfg = None
    api._closed_pinch_center_offset = lambda: np.zeros(3, dtype=np.float64)

    def fake_ik(target_pose_wxyz_xyz, prev_cfg=None):
        pos = np.asarray(target_pose_wxyz_xyz, dtype=np.float64)[-3:]
        return np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    api.ik_solve_fn = fake_ik

    api.move_pinch_center_horizontal_line(
        np.array([0.2, 0.1, 0.35], dtype=np.float64),
        np.array([0.4, 0.1, 0.35], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert env.trajectory is not None
    assert env.trajectory.shape == (50, 7)
    assert np.allclose(env.trajectory[:, 1], 0.1)
    assert np.allclose(env.trajectory[:, 2], 0.35)


def test_g1_real_control_api_z_approach_offsets_in_world_z() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    class FakeEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[str, np.ndarray | None]] = []

        def move_to_pregrasp_side_pose_once(self, *, force: bool = False) -> bool:
            self.calls.append(("pregrasp", None))
            return True

        def move_to_joints_blocking(self, joints: np.ndarray) -> None:
            self.calls.append(("joints", np.asarray(joints, dtype=np.float64).copy()))

    env = FakeEnv()
    api = G1RealControlApi.__new__(G1RealControlApi)
    api._env = env
    api._TCP_OFFSET = np.zeros(3, dtype=np.float64)
    api.cfg = None
    api._webui_enabled = False
    ik_targets: list[np.ndarray] = []

    def fake_ik(target_pose_wxyz_xyz, prev_cfg=None):
        ik_targets.append(np.asarray(target_pose_wxyz_xyz, dtype=np.float64).copy())
        return np.ones(7, dtype=np.float64) * len(ik_targets)

    api.ik_solve_fn = fake_ik
    quat_90deg_y_wxyz = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    pos = np.array([0.4, -0.2, 0.3])

    api.goto_pose(pos, quat_90deg_y_wxyz, z_approach=0.1)

    assert np.allclose(ik_targets[0][-3:], [0.4, -0.2, 0.4])
    assert np.allclose(ik_targets[1][-3:], [0.4, -0.2, 0.3])


def test_g1_real_control_api_extracts_right_arm_from_dual_arm_ik_solution() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    dual_arm_cfg = np.arange(14, dtype=np.float64)

    joints = G1RealControlApi._extract_right_arm_joints(dual_arm_cfg)

    assert np.allclose(joints, np.arange(7, 14, dtype=np.float64))


def test_g1_real_control_api_extracts_right_arm_from_29dof_with_hand_ik_solution() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    whole_body_with_hand_cfg = np.arange(43, dtype=np.float64)

    joints = G1RealControlApi._extract_right_arm_joints(whole_body_with_hand_cfg)

    assert np.allclose(joints, np.arange(29, 36, dtype=np.float64))


def test_g1_real_api_and_env_are_registered() -> None:
    import capx.envs.simulators  # noqa: F401
    import capx.integrations  # noqa: F401
    from capx.envs.base import list_envs
    from capx.integrations.base_api import list_apis

    assert "g1_real_low_level" in list_envs()
    assert "G1RealControlApi" in list_apis()
    assert "G1CameraApi" in list_apis()


def test_g1_camera_api_decodes_protocol_a_and_builds_capx_observation() -> None:
    from capx.integrations.g1.vision import G1CameraApi, G1CameraConfig

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    depth_m = np.full((4, 5), 1.25, dtype=np.float32)
    payload = msgpack.packb(
        {
            "timestamps": {"ego_view": 10.5},
            "images": {"ego_view": _jpeg_bytes_from_rgb(rgb)},
            "depth_m": depth_m,
        },
        use_bin_type=True,
    )

    api = G1CameraApi(
        config=G1CameraConfig(
            rgb_key="ego_view",
            capx_camera_name="robot0_robotview",
            intrinsics=np.eye(3, dtype=np.float32),
            pose_mat=np.eye(4, dtype=np.float32),
        )
    )

    sample = api.decode_payload(payload)
    observation = api.sample_to_capx_observation(sample)

    assert sample.rgb_key == "ego_view"
    assert sample.rgb.shape == rgb.shape
    assert sample.depth_m is not None
    assert np.allclose(sample.depth_m, depth_m)
    assert set(observation) == {"camera_top", "timestamp"}
    assert observation["camera_top"]["images"]["rgb"].shape == rgb.shape
    assert observation["camera_top"]["images"]["depth"].shape == (4, 5, 1)
    assert observation["timestamp"] == 10.5


def test_dex3_left_hand_uses_mirrored_joint_signs_and_left_topics() -> None:
    from capx.integrations.g1.sdk import (
        G1Dex3HandBridge,
        dex3_grasp_joints,
        dex3_semantic_to_dds_joints,
    )

    target = dex3_grasp_joints(trigger=1.0, squeeze=0.0, hand_side="left")
    bridge = G1Dex3HandBridge(
        dry_run=True,
        hand_side="left",
        hand_cmd_factory=_fake_hand_cmd,
    )
    cmd = bridge.build_hand_cmd(target)

    assert target == pytest.approx([-0.5, 0.4, 0.7, -1.0, -1.0, 0.0, 0.0])
    assert bridge.publisher_topic == "rt/dex3/left/cmd"
    assert bridge.state_topic == "rt/dex3/left/state"
    assert [motor.q for motor in cmd.motor_cmd] == pytest.approx(
        dex3_semantic_to_dds_joints(target, hand_side="left")
    )


def test_lowcmd_left_arm_command_maps_7_joint_vector_and_preserves_right_arm() -> None:
    from capx.integrations.g1.sdk import G1ArmSdkBridge, G1_LEFT_ARM_MOTOR_INDICES

    bridge = G1ArmSdkBridge(dry_run=True, arm_side="left", low_cmd_factory=_fake_low_cmd)
    bridge._handle_low_state(_fake_low_state(mode_machine=5))
    target = np.linspace(-0.7, 0.7, 7)

    cmd = bridge.build_low_cmd(target)

    assert G1_LEFT_ARM_MOTOR_INDICES == tuple(range(15, 22))
    for i, motor_idx in enumerate(G1_LEFT_ARM_MOTOR_INDICES):
        assert cmd.motor_cmd[motor_idx].q == pytest.approx(target[i])
    assert cmd.motor_cmd[22].q == pytest.approx(2.2)


def test_gateway_arm_action_maps_left_arm_and_preserves_right_arm() -> None:
    from capx.integrations.g1.gateway import G1GatewayArmActionBridge

    bridge = G1GatewayArmActionBridge(
        dry_run=True,
        arm_side="left",
        arm_message_type=_FakeGatewayArmActionMsg,
    )
    q = np.arange(29, dtype=np.float64) / 10.0
    bridge._handle_body_state(
        "body_control_data",
        SimpleNamespace(q=q, qd=np.zeros(29, dtype=np.float64), timestamp_us=123),
    )
    target = np.linspace(-0.7, 0.7, 7)

    msg = bridge.build_arm_message(target)

    assert msg.act[:7] == pytest.approx(target)
    assert msg.act[7:14] == pytest.approx(q[22:29])
    assert np.allclose(bridge.get_arm_joint_positions(), q[15:22])


def test_left_control_api_matches_low_level_arm_side_and_exposes_right_api_surface() -> None:
    from capx.integrations.g1.control import G1LeftRealControlApi, G1RealControlApi

    class LeftEnv:
        arm_side = "left"

    left_api = G1LeftRealControlApi(LeftEnv(), use_sam3=False)
    generic_left_api = G1RealControlApi(LeftEnv(), use_sam3=False, arm_side="left")

    assert left_api.arm_side == "left"
    assert set(left_api.functions()) == set(generic_left_api.functions())
    with pytest.raises(ValueError, match="does not match"):
        G1RealControlApi(LeftEnv(), use_sam3=False)
