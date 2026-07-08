from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _args(**overrides):
    values = {
        "interface": "test0",
        "domain_id": 0,
        "cmd_topic": "rt/dex3/right/cmd",
        "state_topic": "rt/lf/dex3/right/state",
        "cycles": 1,
        "hold_s": 0.0,
        "execute": False,
        "write_timeout": 2.0,
        "wait_for_state": True,
        "open_on_exit": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeBridge:
    instances: list["FakeBridge"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.connected = False
        self.waited = False
        self.published: list[np.ndarray] = []
        FakeBridge.instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def wait_for_hand_state(self, timeout_s: float = 3.0) -> bool:
        self.waited = True
        return True

    def publish_hand_joints(self, joints) -> bool:
        self.published.append(np.asarray(joints, dtype=np.float64).copy())
        return True

    def close(self) -> None:
        pass


def test_cycle_sequence_matches_open_index_pinch_open_three_finger_close_open() -> None:
    from capx.integrations.g1.sdk import dex3_grasp_joints
    from tools.g1_dex3_cycle import build_action_sequence

    sequence = build_action_sequence()

    assert [action.name for action in sequence] == [
        "open_gripper",
        "close_index_pinch",
        "open_gripper",
        "close_gripper",
        "open_gripper",
    ]
    assert np.allclose(sequence[0].joints, dex3_grasp_joints(trigger=0.0, squeeze=0.0))
    assert np.allclose(sequence[1].joints, dex3_grasp_joints(trigger=1.0, squeeze=0.0))
    assert np.allclose(sequence[3].joints, dex3_grasp_joints(trigger=1.0, squeeze=1.0))


def test_dry_run_does_not_connect_but_builds_all_commands() -> None:
    from capx.integrations.g1.sdk import dex3_grasp_joints
    from tools.g1_dex3_cycle import run

    FakeBridge.instances.clear()

    run(_args(cycles=2), bridge_cls=FakeBridge, sleep_fn=lambda _: None)

    bridge = FakeBridge.instances[-1]
    assert bridge.kwargs["dry_run"] is True
    assert bridge.kwargs["network_interface"] == "test0"
    assert bridge.connected is False
    assert bridge.waited is False
    assert len(bridge.published) == 11
    assert np.allclose(bridge.published[0], dex3_grasp_joints(trigger=0.0, squeeze=0.0))
    assert np.allclose(bridge.published[1], dex3_grasp_joints(trigger=1.0, squeeze=0.0))
    assert np.allclose(bridge.published[3], dex3_grasp_joints(trigger=1.0, squeeze=1.0))
    assert np.allclose(bridge.published[-1], dex3_grasp_joints(trigger=0.0, squeeze=0.0))


def test_execute_connects_waits_and_opens_on_exit() -> None:
    from capx.integrations.g1.sdk import dex3_grasp_joints
    from tools.g1_dex3_cycle import run

    FakeBridge.instances.clear()

    run(_args(execute=True, cycles=1), bridge_cls=FakeBridge, sleep_fn=lambda _: None)

    bridge = FakeBridge.instances[-1]
    assert bridge.kwargs["dry_run"] is False
    assert bridge.connected is True
    assert bridge.waited is True
    assert len(bridge.published) == 6
    assert np.allclose(bridge.published[-1], dex3_grasp_joints(trigger=0.0, squeeze=0.0))
