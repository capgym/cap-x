from verl_agent_reward import capx_franka_reward


class _FakeEnv:
    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def step(self, source: str):
        return {}, 0.2, False, False, {
            "sandbox_rc": 0,
            "task_completed": source == "succeed()",
        }


def test_reward_won_uses_task_completion(monkeypatch) -> None:
    fake = _FakeEnv()
    monkeypatch.setitem(capx_franka_reward.initialized_envs, "test", fake)

    failed = capx_franka_reward.compute_score("test", "noop()", {}, {"seed": 4})
    succeeded = capx_franka_reward.compute_score("test", "succeed()", {}, {"seed": 5})

    assert failed["score"] == 0.2
    assert failed["won"] is False
    assert succeeded["won"] is True
    assert len(succeeded["program_sha256"]) == 64


def test_reward_rejects_environment_bypass_before_execution(monkeypatch) -> None:
    fake = _FakeEnv()
    monkeypatch.setitem(capx_franka_reward.initialized_envs, "test", fake)

    result = capx_franka_reward.compute_score(
        "test", "env.robosuite_env.reset()", {}, {"seed": 9}
    )

    assert result["score"] == 0.0
    assert result["won"] is False
    assert "forbidden" in result["error"]


def test_structure_bonus_requires_documented_api_calls() -> None:
    assert capx_franka_reward._api_structure_bonus("x = 1") == 0.0
    assert capx_franka_reward._api_structure_bonus("open_gripper()") == 0.005
    assert capx_franka_reward._api_structure_bonus(
        "open_gripper()\nposition, quaternion = sample_grasp_pose('square nut handle')"
    ) == 0.01
