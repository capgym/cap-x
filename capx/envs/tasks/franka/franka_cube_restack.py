from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with API described below.
Goal: Gently place the red cube on top of the green cube and then open the gripper. Nothing should be dropped from a height.
Use the extent of the cubes to calculate the exact height for placement.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, scipy, torch, etc. you need to import them explicitly.
"""
ORACLE_CODE = """
import numpy as np

# Start with gripper open
open_gripper()

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
_, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)


# Sample a grasp pose for the green cube and pick it up
pick_pos, pick_quat = sample_grasp_pose("green cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()
# Lift the green cube after grasping
post_pick_pos = pick_pos.copy()
post_pick_pos[0] -= 0.15
post_pick_pos[2] += 0.1
goto_pose(post_pick_pos, pick_quat)
open_gripper()

# Sample a grasp pose for the red cube and pick it up
pick_pos, pick_quat = sample_grasp_pose("red cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()
# Lift the red cube after grasping
post_pick_pos = pick_pos.copy()
post_pick_pos[0] -= 0.15
post_pick_pos[2] += 0.3
goto_pose(post_pick_pos, pick_quat)

# Compute placement pose on top of the green cube
green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)

place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2] + red_ext[2]
# Use down orientation for placement
place_quat = np.array([0.0, 0.0, 1.0, 0.0])

# Approach and place the red cube on the green cube
goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

# Retract after placing
post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, place_quat)
"""


# ---------------------------- High-level Env -----------------------------
class FrankaRestackCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka cube restack using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "FrankaRestackCodeEnv",
]
