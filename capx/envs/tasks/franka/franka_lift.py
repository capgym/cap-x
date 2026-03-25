from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with API described below.
Goal: pick up the red cube and lift it.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""
ORACLE_CODE = """
import numpy as np

# Get a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red cube")

# Open the gripper before approaching
open_gripper()

# Approach the grasp pose from above (0.1 m offset in Z)
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the exact grasp pose
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grasp the cube
close_gripper()

# Lift the cube slightly to ensure a safe grasp
lift_offset = np.array([0.0, 0.0, 0.1])  # 10 cm lift
lift_pos = grasp_pos + lift_offset
goto_pose(lift_pos, grasp_quat)
"""


# ---------------------------- High-level Env -----------------------------
class FrankaLiftCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka lift using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "FrankaLiftCodeEnv",
]
