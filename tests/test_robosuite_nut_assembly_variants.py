from unittest.mock import patch

from capx.envs.simulators.robosuite_nut_assembly import (
    FrankaRobosuiteNutAssembly,
    FrankaRobosuiteNutAssemblyFull,
)


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
