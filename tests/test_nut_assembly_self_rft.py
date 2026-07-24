import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_nut_assembly_self_rft.py"
_SPEC = importlib.util.spec_from_file_location("train_nut_assembly_self_rft", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_validate_trial_dir = _MODULE._validate_trial_dir


def test_self_rft_accepts_guarded_non_oracle_success(tmp_path) -> None:
    result_dir = tmp_path / "result"
    trial_dir = result_dir / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    trial_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(json.dumps({"use_oracle_code": False}))
    (trial_dir / "code.py").write_text("open_gripper()\n")

    source = _validate_trial_dir(trial_dir)

    assert source["program"] == "open_gripper()\n"
    assert len(source["program_sha256"]) == 64


def test_self_rft_rejects_oracle_rollouts(tmp_path) -> None:
    result_dir = tmp_path / "result"
    trial_dir = result_dir / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    trial_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(json.dumps({"use_oracle_code": True}))
    (trial_dir / "code.py").write_text("open_gripper()\n")

    with pytest.raises(ValueError, match="oracle rollout"):
        _validate_trial_dir(trial_dir)


def test_self_rft_accepts_tasks_without_repository_oracle(tmp_path) -> None:
    result_dir = tmp_path / "result"
    trial_dir = result_dir / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    trial_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(json.dumps({"use_oracle_code": False}))
    (trial_dir / "code.py").write_text("get_active_nut_types()\n")

    source = _validate_trial_dir(trial_dir, oracle_code=None)

    assert source["program"] == "get_active_nut_types()\n"
