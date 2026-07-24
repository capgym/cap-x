from unittest.mock import patch

from capx.envs.simulators.robosuite_nut_assembly import (
    FrankaRobosuiteNutAssembly,
    FrankaRobosuiteNutAssemblyFull,
)
from capx.envs.tasks.franka.franka_nut_assembly import FULL_PROMPT


def test_full_nut_assembly_uses_two_sequence_horizon_by_default() -> None:
    with patch.object(FrankaRobosuiteNutAssembly, "__init__") as init:
        FrankaRobosuiteNutAssemblyFull()

    assert init.call_args.kwargs["task_variant"] == "full"
    assert init.call_args.kwargs["max_steps"] == 2000


def test_full_nut_assembly_preserves_explicit_horizon() -> None:
    with patch.object(FrankaRobosuiteNutAssembly, "__init__") as init:
        FrankaRobosuiteNutAssemblyFull(max_steps=2500)

    assert init.call_args.kwargs["task_variant"] == "full"
    assert init.call_args.kwargs["max_steps"] == 2500


def test_full_prompt_disambiguates_dynamic_object_queries_and_action_order() -> None:
    assert "extruded handle of <nut_type> nut" in FULL_PROMPT
    assert '"<nut_type> nut"' in FULL_PROMPT
    assert '"<nut_type> peg"' in FULL_PROMPT
    assert "Do not close the gripper for the first time at the peg" in FULL_PROMPT
    assert "do not use hard-coded object coordinates" in FULL_PROMPT
