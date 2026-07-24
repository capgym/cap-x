from capx.envs.tasks import get_config, get_exec_env


def test_additional_nut_assembly_tasks_are_registered_without_oracles() -> None:
    expected = {
        "franka_nut_assembly_round_code_env": "franka_robosuite_nut_assembly_round_low_level",
        "franka_nut_assembly_single_code_env": "franka_robosuite_nut_assembly_single_low_level",
        "franka_nut_assembly_full_code_env": "franka_robosuite_nut_assembly_full_low_level",
        "franka_tool_hang_code_env": "franka_robosuite_tool_hang_low_level",
        "two_arm_peg_in_hole_code_env": "two_arm_peg_in_hole_robosuite",
    }

    for data_source, low_level in expected.items():
        assert get_config(data_source).low_level == low_level
        assert get_exec_env(data_source).oracle_code is None
