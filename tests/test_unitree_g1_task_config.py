from pathlib import Path

from capx.envs.configs.loader import DictLoader


def test_unitree_g1_grasp_bottle_task_is_registered() -> None:
    import capx.envs.tasks as tasks

    cfg = tasks.get_config("unitree_g1_grasp_bottle_code_env")

    assert "unitree_g1_grasp_bottle_code_env" in tasks.list_exec_envs()
    assert cfg.low_level == "g1_real_low_level"
    assert cfg.apis == ["G1RealControlApi", "G1CameraApi"]
    assert cfg.privileged is False


def test_unitree_g1_grasp_bottle_task_prompt_and_oracle_use_front_policy_pinch_center_api() -> None:
    from capx.envs.tasks.unitree_g1.grasp_bottle import UnitreeG1GraspBottleCodeEnv

    prompt = UnitreeG1GraspBottleCodeEnv.prompt
    oracle = UnitreeG1GraspBottleCodeEnv.oracle_code

    assert prompt is not None
    assert "plastic water bottle" in prompt
    assert "right arm" in prompt
    assert "single-arm" in prompt
    assert "7-DoF" in prompt
    assert oracle is not None
    assert "sample_grasp_center_pose" in oracle
    assert "grasp_at_pinch_center" in oracle
    assert "move_to_pregrasp_side_pose" in oracle
    assert oracle.index("grasp_at_pinch_center") < oracle.index("move_to_pregrasp_side_pose")
    assert "sample_grasp_pose" not in oracle
    assert "goto_pose(grasp_pos, grasp_quat" not in oracle
    assert "Junhao-style front policy" in prompt
    assert "contact/visual center as pinch target" in prompt
    assert "fixed vertical front-palm rotation" in prompt
    assert "elevated transit" in prompt
    assert "0.06m" in prompt
    assert "local table plane" in prompt
    assert "set_gripper_trigger_squeeze(trigger, squeeze)" in prompt
    assert "close_index_pinch()" in prompt
    assert "close_middle_pinch()" in prompt
    assert "move_hand_joints(joints)" in prompt
    assert "move_to_pinch_pregrasp(pos, quat)" in prompt
    assert "move_pinch_center_horizontal_line" in prompt
    assert "short final horizontal approach" in prompt
    assert "After the grasp/lift completes" in prompt


def test_unitree_g1_grasp_bottle_yaml_points_to_task_and_right_hand() -> None:
    cfg = DictLoader.load("env_configs/g1/g1_grasp_bottle.yaml")

    env = cfg["env"]
    task_cfg = env["cfg"]
    low_level = task_cfg["low_level"]
    sam3_server = cfg["api_servers"][0]
    pyroki_server = cfg["api_servers"][2]

    assert Path("env_configs/g1/g1_grasp_bottle.yaml").exists()
    assert env["_target_"] == (
        "capx.envs.tasks.unitree_g1.grasp_bottle.UnitreeG1GraspBottleCodeEnv"
    )
    assert low_level["_target_"] == "capx.envs.simulators.g1_real.G1RealLowLevel"
    assert low_level["network_interface"] is None
    assert low_level["enable_dex3"] is True
    assert low_level["sam3_mask_popup"] is True
    assert low_level["grasp_debug_visualization"] is True
    assert low_level["grasp_debug_output_dir"] == "outputs/g1_grasp_bottle/grasp_debug"
    assert low_level["grasp_debug_approach_distance"] == 0.1
    assert low_level["table_estimation_enabled"] is True
    assert low_level["table_estimation_required"] is True
    assert low_level["table_high_clearance_m"] == 0.20
    assert low_level["target_high_clearance_m"] == 0.12
    assert low_level["target_table_clearance_m"] == 0.025
    assert low_level["short_approach_distance_m"] == 0.06
    assert low_level["dex3_hand_side"] == "right"
    assert low_level["dex3_hand_cmd_topic"] == "rt/dex3/right/cmd"
    assert low_level["dex3_hand_state_topic"] == "rt/lf/dex3/right/state"
    assert low_level["pregrasp_side_before_first_pose"] is True
    assert low_level["pregrasp_side_joints"] == [0.0, -1.25, 0.0, 1.1, 0.0, 0.0, 0.0]
    assert task_cfg["apis"] == ["G1RealControlApi", "G1CameraApi"]
    assert sam3_server["checkpoint_path"] == "capx/model_weights/sam3/sam3.pt"
    assert pyroki_server["robot"] == "env_configs/g1/g1_29dof_with_hand.urdf"
    assert pyroki_server["target_link"] == "right_hand_palm_link"
    assert pyroki_server["root_link"] == "torso_link"
    assert pyroki_server["fk_position_tolerance"] == 0.12
    assert pyroki_server["fk_rotation_tolerance"] == 0.2
    assert cfg["output_dir"] == "./outputs/g1_grasp_bottle"


def test_unitree_g1_upper_body_yaml_uses_29dof_hand_urdf_for_pyroki() -> None:
    cfg = DictLoader.load("env_configs/g1/g1_real_upper_body.yaml")

    low_level = cfg["env"]["cfg"]["low_level"]
    sam3_server = cfg["api_servers"][0]
    pyroki_server = cfg["api_servers"][2]

    assert low_level["enable_dex3"] is True
    assert low_level["dex3_hand_side"] == "right"
    assert sam3_server["checkpoint_path"] == "capx/model_weights/sam3/sam3.pt"
    assert pyroki_server["robot"] == "env_configs/g1/g1_29dof_with_hand.urdf"
    assert pyroki_server["target_link"] == "right_hand_palm_link"
    assert pyroki_server["root_link"] == "torso_link"
    assert pyroki_server["fk_position_tolerance"] == 0.12
    assert pyroki_server["fk_rotation_tolerance"] == 0.2



def test_g1_real_control_api_exposes_dex3_hand_functions_to_task_code() -> None:
    from capx.integrations.g1.control import G1RealControlApi

    class FakeEnv:
        pass

    functions = G1RealControlApi(FakeEnv(), use_sam3=False).functions()

    assert "open_gripper" in functions
    assert "close_gripper" in functions
    assert "close_index_pinch" in functions
    assert "close_middle_pinch" in functions
    assert "set_gripper_trigger_squeeze" in functions
    assert "move_hand_joints" in functions
    assert "move_to_pregrasp_side_pose" in functions
    assert "sample_grasp_center_pose" in functions
    assert "move_pinch_center_to_pose" in functions
    assert "move_to_pinch_pregrasp" in functions
    assert "move_pinch_center_horizontal_line" in functions
    assert "grasp_at_pinch_center" in functions


def test_unitree_g1_left_grasp_bottle_task_is_registered() -> None:
    import capx.envs.tasks as tasks

    cfg = tasks.get_config("unitree_g1_grasp_bottle_left_code_env")

    assert "unitree_g1_grasp_bottle_left_code_env" in tasks.list_exec_envs()
    assert cfg.low_level == "g1_real_low_level"
    assert cfg.apis == ["G1LeftRealControlApi", "G1CameraApi"]


def test_unitree_g1_left_grasp_bottle_yaml_routes_every_control_layer_left() -> None:
    cfg = DictLoader.load("env_configs/g1/g1_grasp_bottle_left.yaml")

    task_cfg = cfg["env"]["cfg"]
    low_level = task_cfg["low_level"]
    pyroki_server = cfg["api_servers"][2]

    assert cfg["env"]["_target_"] == (
        "capx.envs.tasks.unitree_g1.grasp_bottle_left.UnitreeG1LeftGraspBottleCodeEnv"
    )
    assert low_level["arm_side"] == "left"
    assert low_level["sdk_bridge"]["arm_side"] == "left"
    assert low_level["dex3_hand_side"] == "left"
    assert low_level["table_estimation_enabled"] is True
    assert low_level["table_estimation_required"] is True
    assert low_level["short_approach_distance_m"] == 0.06
    assert low_level["dex3_hand_cmd_topic"] == "rt/dex3/left/cmd"
    assert low_level["dex3_hand_state_topic"] == "rt/dex3/left/state"
    assert low_level["pregrasp_side_joints"] == [0.0, 1.25, 0.0, 1.1, 0.0, 0.0, 0.0]
    assert task_cfg["apis"] == ["G1LeftRealControlApi", "G1CameraApi"]
    assert pyroki_server["target_link"] == "left_hand_palm_link"
