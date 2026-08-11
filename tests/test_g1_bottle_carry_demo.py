from __future__ import annotations

import inspect
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import tools.g1_bottle_carry_demo as bottle_demo
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
        hold_status_fn=lambda: {"success": True, "healthy": True},
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
    runner = BottleCarryDemoRunner(
        api.functions(),
        hold_status_fn=lambda: {"success": True, "healthy": True},
    )

    with pytest.raises(RuntimeError, match="grasp_at_table_a"):
        runner.place_at_table_b()


def test_runner_rejects_incomplete_api_without_opening_environment() -> None:
    with pytest.raises(ValueError, match="missing required functions"):
        BottleCarryDemoRunner(
            {"open_gripper": lambda: None},
            hold_status_fn=lambda: {"success": True, "healthy": True},
        )


@pytest.mark.parametrize("failure_point", ["grasp", "side"])
def test_partial_grasp_failure_does_not_authorize_placement(
    failure_point: str,
) -> None:
    api = FakeG1Api()

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{failure_point} failed")

    if failure_point == "grasp":
        api.grasp_at_pinch_center = fail
    else:
        api.move_to_pregrasp_side_pose = fail
    runner = BottleCarryDemoRunner(
        api.functions(),
        hold_status_fn=lambda: {"success": True, "healthy": True},
    )

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        runner.grasp_at_table_a()

    status = runner.status()
    assert status["success"] is False
    assert status["phase"] == "failed"
    assert status["grasp_position"] is None
    assert status["grasp_quaternion_wxyz"] is None
    with pytest.raises(RuntimeError):
        runner.place_at_table_b()
    assert not any(name == "move_pinch" for name, _payload in api.calls)


def test_hold_monitor_blocks_placement_after_publish_health_failure() -> None:
    api = FakeG1Api()
    health = {"success": True, "healthy": True, "last_error": None}
    prompt_count = 0
    runner: BottleCarryDemoRunner

    def input_fn(_prompt: str) -> str:
        nonlocal prompt_count
        prompt_count += 1
        if prompt_count == 2:
            health.update(
                healthy=False,
                last_error="arm_action hold publish failed",
            )
            deadline = time.monotonic() + 1.0
            while (
                runner.phase.value != "failed"
                and time.monotonic() < deadline
            ):
                time.sleep(0.002)
        return ""

    runner = BottleCarryDemoRunner(
        api.functions(),
        hold_status_fn=lambda: dict(health),
        hold_check_period_s=0.001,
        input_fn=input_fn,
    )

    with pytest.raises(RuntimeError, match="arm_hold_unhealthy"):
        runner.run()

    assert runner.status()["phase"] == "failed"
    assert not any(name == "move_pinch" for name, _payload in api.calls)


def test_runner_rejects_repeated_or_out_of_order_phase_transitions() -> None:
    api = FakeG1Api()
    runner = BottleCarryDemoRunner(
        api.functions(),
        hold_status_fn=lambda: {"success": True, "healthy": True},
    )

    runner.grasp_at_table_a()
    with pytest.raises(RuntimeError, match="initialized phase required"):
        runner.grasp_at_table_a()
    runner.place_at_table_b()
    with pytest.raises(RuntimeError, match="holding_for_navigation phase required"):
        runner.place_at_table_b()


def test_low_level_arm_hold_status_reports_dry_run_and_latched_errors() -> None:
    from capx.envs.simulators.g1_real import G1RealLowLevel

    alive_thread = SimpleNamespace(is_alive=lambda: True)
    low_level = SimpleNamespace(
        dry_run=False,
        _arm_hold_lock=threading.Lock(),
        _arm_hold_target=np.zeros(7, dtype=np.float64),
        _arm_hold_paused=False,
        _arm_hold_last_error=None,
        _arm_hold_thread=alive_thread,
        _hold_arm_after_move=True,
    )

    healthy = G1RealLowLevel.arm_hold_status(low_level)
    assert healthy["healthy"] is True
    low_level._arm_hold_last_error = "publish failed"
    failed = G1RealLowLevel.arm_hold_status(low_level)
    assert failed["healthy"] is False
    assert failed["last_error"] == "publish failed"

    low_level.dry_run = True
    dry_run = G1RealLowLevel.arm_hold_status(low_level)
    assert dry_run["healthy"] is True
    assert dry_run["dry_run"] is True


def test_persistent_environment_is_reset_exactly_once(monkeypatch) -> None:
    environment = SimpleNamespace(reset_calls=0)

    def reset() -> None:
        environment.reset_calls += 1

    environment.reset = reset
    config = {"env": {"cfg": {"low_level": {"dry_run": False}}}}
    monkeypatch.setattr(
        bottle_demo.DictLoader,
        "load",
        lambda _path: config,
    )
    monkeypatch.setattr(bottle_demo, "instantiate", lambda _factory: environment)

    loaded = bottle_demo._load_persistent_environment("demo.yaml", dry_run=True)

    assert loaded is environment
    assert environment.reset_calls == 1
    assert config["env"]["cfg"]["low_level"]["dry_run"] is True


def test_main_closes_low_level_environment_exactly_once(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("env: {}", encoding="utf-8")
    calls = {"close": 0, "run": 0}
    low_level = SimpleNamespace(
        close=lambda: calls.__setitem__("close", calls["close"] + 1)
    )
    environment = SimpleNamespace(low_level_env=low_level)
    api = FakeG1Api()

    class FakeRunner:
        REQUIRED_FUNCTIONS = BottleCarryDemoRunner.REQUIRED_FUNCTIONS

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> dict[str, object]:
            calls["run"] += 1
            return {"success": True}

    monkeypatch.setattr(
        bottle_demo,
        "_load_persistent_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(
        bottle_demo,
        "_find_g1_functions",
        lambda _environment: api.functions(),
    )
    monkeypatch.setattr(
        bottle_demo,
        "_find_arm_hold_status",
        lambda _environment: (lambda: {"healthy": True}),
    )
    monkeypatch.setattr(bottle_demo, "BottleCarryDemoRunner", FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "g1_bottle_carry_demo.py",
            "--dry-run",
            "--config-path",
            str(config_path),
        ],
    )

    assert bottle_demo.main() == 0
    assert calls == {"close": 1, "run": 1}


def test_runner_source_does_not_own_locomotion_or_lowcmd_transport() -> None:
    source = inspect.getsource(bottle_demo.BottleCarryDemoRunner)

    assert "pedal_command" not in source
    assert "rt/lowcmd" not in source
    assert "LCMPedalCommandTransport" not in source
