from argparse import Namespace
import json

import pytest

from scripts.validate_nut_assembly_results import validate


def _result_dir(tmp_path, *, success_rate: float = 0.81, oracle: bool = False):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    summary = {
        "total_trials": 2,
        "task_completed": round(success_rate * 2),
        "task_completion_rate": success_rate,
        "use_oracle_code": oracle,
        "git_dirty": False,
        "git_commit": "abc1234",
    }
    (result_dir / "summary.json").write_text(json.dumps(summary))
    for trial in range(1, 3):
        trial_dir = result_dir / f"trial_{trial:02d}"
        trial_dir.mkdir()
        (trial_dir / "code.py").write_text("open_gripper()")
    return result_dir


def _args(result_dir, **overrides):
    values = {
        "result_dir": result_dir,
        "expected_trials": 2,
        "minimum_success_rate": 0.8,
        "require_video": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_accepts_strictly_greater_than_threshold(tmp_path) -> None:
    receipt = validate(_args(_result_dir(tmp_path)))

    assert receipt["generated_programs_checked"] == 2
    assert receipt["task_completion_rate"] == 0.81


def test_rejects_equal_threshold(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be greater"):
        validate(_args(_result_dir(tmp_path, success_rate=0.8)))


def test_rejects_oracle_run(tmp_path) -> None:
    with pytest.raises(ValueError, match="oracle-code"):
        validate(_args(_result_dir(tmp_path, oracle=True)))
