from __future__ import annotations

import numpy as np
import pytest

from tools.g1_bottle_carry_demo import BottleCarryDemoRunner


class FakeG1Api:
    def __init__(self) -> None:
        self.position = np.asarray([0.55, -0.18, 0.82], dtype=np.float64)
        self.quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.calls: list[tuple[str, object]] = []

    def functions(self):
        return {
            "sample_grasp_center_pose": self.sample_grasp_center_pose,
            "grasp_at_pinch_center": self.grasp_at_pinch_center,
            "move_pinch_center_to_pose": self.move_pinch_center_to_pose,
            "move_to_pregrasp_side_pose": self.move_to_pregrasp_side_pose,
            "open_gripper": self.open_gripper,
        }

    def sample_grasp_center_pose(self, object_name: str):
        self.calls.append(("sample", object_name))
        return self.position.copy(), self.quaternion.copy()

    def grasp_at_pinch_center(self, position, quaternion, **kwargs):
        self.calls.append(
            (
                "grasp",
                (
                    np.asarray(position).copy(),
                    np.asarray(quaternion).copy(),
                    dict(kwargs),
                ),
            )
        )

    def move_pinch_center_to_pose(self, position, quaternion):
        self.calls.append(
            (
                "move_pinch",
                (np.asarray(position).copy(), np.asarray(quaternion).copy()),
            )
        )

    def move_to_pregrasp_side_pose(self):
        self.calls.append(("side", None))

    def open_gripper(self):
        self.calls.append(("open", None))


def test_persistent_runner_keeps_grasp_pose_across_hold_and_place() -> None:
    api = FakeG1Api()
    prompts: list[str] = []
    runner = BottleCarryDemoRunner(
        api.functions(),
        input_fn=lambda prompt: prompts.append(prompt) or "",
    )

    result = runner.run()

    assert result["success"] is True
    assert result["phase"] == "placed"
    assert result["environment_persistent"] is True
    assert len(prompts) == 2
    assert [name for name, _payload in api.calls] == [
        "sample",
        "grasp",
        "side",
        "move_pinch",
        "move_pinch",
        "open",
        "move_pinch",
        "side",
    ]
    grasp_payload = api.calls[1][1]
    descend_payload = api.calls[4][1]
    np.testing.assert_allclose(grasp_payload[0], api.position)
    np.testing.assert_allclose(descend_payload[0], api.position)
    np.testing.assert_allclose(descend_payload[1], api.quaternion)
    assert sum(name == "sample" for name, _payload in api.calls) == 1


def test_place_requires_completed_grasp_phase() -> None:
    api = FakeG1Api()
    runner = BottleCarryDemoRunner(api.functions())

    with pytest.raises(RuntimeError, match="grasp_at_table_a"):
        runner.place_at_table_b()


def test_runner_rejects_incomplete_api_without_opening_environment() -> None:
    with pytest.raises(ValueError, match="missing required functions"):
        BottleCarryDemoRunner({"open_gripper": lambda: None})
