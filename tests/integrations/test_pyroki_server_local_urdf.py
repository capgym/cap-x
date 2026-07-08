from __future__ import annotations

from pathlib import Path


def test_init_pyroki_server_accepts_local_urdf_path() -> None:
    import capx.serving.launch_pyroki_server as server

    server._ROBOT = None
    server._ROBOT_COLL = None
    server._TARGET_LINK = None

    urdf_path = Path("env_configs/g1/g1_29dof_with_hand.urdf")

    server.init_pyroki_server(
        robot_urdf_name=str(urdf_path),
        target_link_name="right_hand_palm_link",
    )

    assert server._ROBOT is not None
    assert server._ROBOT_COLL is not None
    assert server._TARGET_LINK == "right_hand_palm_link"


def test_init_pyroki_server_can_reduce_g1_to_torso_root_right_arm_chain() -> None:
    import capx.serving.launch_pyroki_server as server

    server._ROBOT = None
    server._ROBOT_COLL = None
    server._TARGET_LINK = None

    urdf_path = Path("env_configs/g1/g1_29dof_with_hand.urdf")

    server.init_pyroki_server(
        robot_urdf_name=str(urdf_path),
        target_link_name="right_hand_palm_link",
        root_link_name="torso_link",
    )

    assert server._ROBOT is not None
    assert server._TARGET_LINK == "right_hand_palm_link"
    assert server._ROBOT.links.names[0] == "torso_link"
    assert server._ROBOT.joints.actuated_names == (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def test_init_pyroki_server_records_fk_tolerances(monkeypatch) -> None:
    import types

    import capx.serving.launch_pyroki_server as server

    class FakeRobot:
        links = types.SimpleNamespace(names=("torso_link", "right_hand_palm_link"))
        joints = types.SimpleNamespace(actuated_names=("right_shoulder_pitch_joint",))

    ctx = types.SimpleNamespace(robot=FakeRobot(), robot_coll=object())
    monkeypatch.setattr(server, "get_pyroki_context", lambda *args, **kwargs: ctx)

    server.init_pyroki_server(
        robot_urdf_name="fake.urdf",
        target_link_name="right_hand_palm_link",
        root_link_name="torso_link",
        fk_position_tolerance=0.1,
        fk_rotation_tolerance=0.75,
    )

    assert server._FK_POSITION_TOLERANCE == 0.1
    assert server._FK_ROTATION_TOLERANCE == 0.75


def test_pyroki_do_solve_ik_uses_prev_cfg_as_initial_cfg(monkeypatch) -> None:
    import numpy as np

    import capx.serving.launch_pyroki_server as server

    calls = {}

    def fake_solve_ik_vel_cost(**kwargs):
        calls.update(kwargs)
        return np.arange(7, dtype=np.float64)

    monkeypatch.setattr(server.pks, "solve_ik_vel_cost", fake_solve_ik_vel_cost)
    server._ROBOT = object()
    server._TARGET_LINK = "right_hand_palm_link"

    prev_cfg = np.linspace(0.0, 0.6, 7, dtype=np.float64)
    target_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.3], dtype=np.float64)

    joints = server._do_solve_ik(target_pose, prev_cfg)

    assert joints == list(np.arange(7, dtype=np.float64))
    assert np.allclose(calls["prev_cfg"], prev_cfg)
    assert np.allclose(calls["initial_cfg"], prev_cfg)


def test_pyroki_validate_ik_solution_returns_fk_residuals(monkeypatch) -> None:
    import capx.serving.launch_pyroki_server as server

    monkeypatch.setattr(server, "_ik_fk_residual", lambda joints, target: (0.01, 0.02))
    server._FK_POSITION_TOLERANCE = 0.1
    server._FK_ROTATION_TOLERANCE = 0.2

    assert server._validate_ik_solution([0.0], [0.0]) == (0.01, 0.02)

