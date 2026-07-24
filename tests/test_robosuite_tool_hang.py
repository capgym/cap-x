import inspect

from capx.envs.simulators.robosuite_tool_hang import FrankaRobosuiteToolHang


def test_tool_hang_uses_full_two_stage_horizon_by_default() -> None:
    parameter = inspect.signature(FrankaRobosuiteToolHang.__init__).parameters[
        "max_steps"
    ]

    assert parameter.default == 10000
