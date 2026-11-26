"""CaP-X code execution task: TRON2 tabletop red cube lifting (visual, SAM3 + Contact-GraspNet)."""

from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a LimX TRON2 dual-arm robot with the API described below.
Goal: pick up the red cube on the table with the left arm and lift it at least 10 cm off the table.

Coordinate frame (base_Link):
- +X is forward (toward the table), +Y is left, +Z is up.
- All positions are [x, y, z] in meters relative to base_Link.
- Use `sample_grasp_pose("red cube")` to get a reachable (grasp_pos, grasp_quat) for the gripper.
- Always open the gripper before approaching, use `approach_offset=0.11` in `goto_pose_left` to approach cleanly without knocking the cube over, close the gripper, then lift vertically (+Z).
- Check the return from `goto_pose_left`. If the approach fails, select and plan a new grasp before closing the gripper.
- For the lift, keep the grasp X/Y and orientation and pass a position with a positive Z offset. A 0.12 to 0.16 m tool lift allows some margin for the 0.10 m cube-height goal.
- Leave the cube held above the table at the end of execution.

You may write python code comments for reasoning, but ONLY write executable Python code and do not wrap it in code fences.
The API functions above are already imported. If you need numpy, import it explicitly.
"""

ORACLE_CODE = """
import numpy as np

grasp_pos, grasp_quat = sample_grasp_pose("red cube")
open_gripper_left()
goto_pose_left(grasp_pos, grasp_quat, approach_offset=0.11)
close_gripper_left()

lift_pos = grasp_pos + np.array([0.0, 0.0, 0.16])
goto_pose_left(lift_pos, grasp_quat)
"""


class Tron2CubeLiftVisualCodeEnv(CodeExecutionEnvBase):
    """Non-privileged visual CaP-X task for TRON2 cube lifting."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["Tron2CubeLiftVisualCodeEnv"]
