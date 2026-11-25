"""CaP-X code execution task: TRON2 tabletop red cube lifting."""

from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a LimX TRON2 dual-arm robot with the API described below.
Goal: pick up the red cube on the table with the left arm and lift it at least 10 cm off the table.

Coordinate frame (base_Link):
- +X is forward (toward the table), +Y is left, +Z is up.
- All positions are [x, y, z] in meters relative to base_Link.
- The default gripper orientation points horizontally forward (+X), with jaws opening along +/-Y.
- To avoid knocking the cube over, approach from behind (-X offset ~10 cm), slide forward, close the gripper, then lift vertically (+Z).

You may write python code comments for reasoning, but ONLY write executable Python code and do not wrap it in code fences.
The API functions above are already imported. If you need numpy, import it explicitly.
"""

ORACLE_CODE = """
import numpy as np

cube_pos = get_red_cube_pos()
open_gripper_left()

grasp_pos = cube_pos.copy()
grasp_pos[2] += 0.010  # 1 cm above cube center for clean tabletop clearance

pre_grasp = grasp_pos.copy()
pre_grasp[0] -= 0.11   # 11 cm behind cube along -X

goto_pose_left(pre_grasp)
goto_pose_left(grasp_pos)
close_gripper_left()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.16    # lift 16 cm vertically
goto_pose_left(lift_pos)
"""


class Tron2CubeLiftCodeEnv(CodeExecutionEnvBase):
    """High-level CaP-X code execution environment for TRON2 cube lifting."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["Tron2CubeLiftCodeEnv"]