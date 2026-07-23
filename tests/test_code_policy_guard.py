from capx.security import validate_generated_program
from capx.envs.tasks.franka.franka_nut_assembly import FrankaNutAssemblyCodeEnv


def test_allows_documented_robot_api_program() -> None:
    source = """
import numpy as np
from scipy.spatial.transform import Rotation as R

position, quaternion = sample_grasp_pose("square nut handle")
goto_pose(position + np.array([0.0, 0.0, 0.02]), quaternion)
close_gripper()
"""

    assert validate_generated_program(source).allowed


def test_rejects_direct_environment_access() -> None:
    result = validate_generated_program("env.robosuite_env.reset()")

    assert not result.allowed
    assert "forbidden" in result.reason


def test_rejects_oracle_and_introspection_access() -> None:
    for source in (
        "print(ORACLE_CODE)",
        "program = ground_truth['program']",
        "import inspect",
        "open('/tmp/program.py').read()",
        "getattr(APIS, 'oracle_code')",
    ):
        result = validate_generated_program(source)
        assert not result.allowed, source


def test_syntax_errors_are_left_to_the_executor() -> None:
    assert validate_generated_program("def broken(").allowed


def test_rejects_shadowing_documented_robot_api() -> None:
    result = validate_generated_program("def goto_pose(*args):\n    pass")

    assert not result.allowed
    assert result.reason == "cannot redefine documented API: goto_pose"


def test_rejects_undocumented_imports_and_exception_swallowing() -> None:
    imported = validate_generated_program("import franka_emika_api")
    swallowed = validate_generated_program("try:\n    goto_home_joint_position()\nexcept Exception:\n    pass")

    assert not imported.allowed
    assert imported.reason == "forbidden import: franka_emika_api"
    assert not swallowed.allowed
    assert "try/except" in swallowed.reason


def test_nut_assembly_environment_rejects_bypass_before_execution() -> None:
    environment = object.__new__(FrankaNutAssemblyCodeEnv)
    environment._task_prompt = "insert nut"
    environment._get_observation = lambda: {"state": "initial"}
    environment.compute_reward = lambda: 0.0
    environment.low_level_env = type(
        "LowLevel", (), {"task_completed": lambda self: False}
    )()

    observation, reward, terminated, truncated, info = environment.step(
        "env.robosuite_env.reset()"
    )

    assert observation == {"state": "initial"}
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["sandbox_rc"] == 1
    assert "forbidden" in info["stderr"]
