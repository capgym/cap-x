from capx.cli import prepare_verl_dataset


class _FakeEnv:
    oracle_code = "expert_move()"

    def _get_observation(self):
        return {
            "full_prompt": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "insert the nut"}],
                },
            ]
        }

    def close(self) -> None:
        pass


def _patch_env(monkeypatch) -> None:
    monkeypatch.setattr(prepare_verl_dataset, "get_config", lambda _: object())
    monkeypatch.setattr(prepare_verl_dataset, "get_exec_env", lambda _: lambda _: _FakeEnv())


def test_prompt_only_rows_exclude_oracle_program(monkeypatch) -> None:
    _patch_env(monkeypatch)

    rows = prepare_verl_dataset._generate_rows(
        [0, 1],
        split="train",
        data_source="franka_nut_assembly_code_env",
        seed_base=100,
        include_oracle_ground_truth=False,
    )

    assert [row["extra_info"]["seed"] for row in rows] == [100, 101]
    assert all(row["reward_model"]["ground_truth"] == {"program": None} for row in rows)
    assert rows[0]["prompt"][-1]["content"] == "insert the nut"


def test_oracle_ground_truth_requires_explicit_opt_in(monkeypatch) -> None:
    _patch_env(monkeypatch)

    rows = prepare_verl_dataset._generate_rows(
        [0],
        split="train",
        data_source="test",
        seed_base=0,
        include_oracle_ground_truth=True,
    )

    assert rows[0]["reward_model"]["ground_truth"] == {"program": "expert_move()"}
