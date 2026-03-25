from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a two-arm Franka Emika robot system with API described below.
Goal: Arm 0 should pick up the hammer, lift it, and hand it over to Arm 1. Arm 1 should then grasp the hammer handle (not hammer head). The z-value of hammer must be between 0.15 and 0.20 during the handover to count as a success.

Coordinate system:
- All pose functions accept positions in robot0's base frame (same coordinate system as returned/used by get_object_pose/goto_pose_arm*).
- The table surface is not necessarily at z=0.
- The coordinate axis follows these conventions:
  - Z-axis: up (positive) and down (negative)
  - X-axis: right (positive) and left (negative)
  - Y-axis: forward (positive) and backward (negative)

Environment details:
- Arm 0 (left) and Arm 1 (right) are positioned on opposite sides of the table.
- The hammer handle length is randomized between 0.15m and 0.25m.
- The hammer initially lies flat on the table, aligned along the Y-axis, with the handle toward +Y and the hammer head toward -Y

Critical information:
- Handover must occur near the midpoint of the initial gripper positions. Reason about the best handover hammer orientation that allows collision-free transfer to Arm 1.
- The table does not span the entire region between the two arms. If the hammer falls in the central gap it will drop to the floor, making the task UNRECOVERABLE
- AVOID COLLISIONS. You must decompose movements into rotation and translation components, moving stepwise. Reason about the optimal sequence of waypoints to safely avoid collisions.
- Arm links have volume, and intermediate motions are not collision-checked. Keep a minimum 8cm buffer between grippers

Reference quaternions:
- Arm0 gripper facing down opening along X-axis: [0, 0.707, 0.707, 0].
- Arm0 gripper facing down opening along Y-axis: [0, 1, 0, 0].
- Arm1 gripper facing down opening along Y-axis: [0, 0, 1, 0].

Arm gripper approximate initial starting positions (robot0 frame). These values are rough and may vary by a few centimeters each trial:
- Arm0: x = 0.44. y = 0.0
- Arm1: x = 1.18. y = 0.0

You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""


UNPRIVILEGED_ORACLE_CODE = """
import numpy as np

# get poses
handle_pos, handle_quat, _ = get_object_pose('hammer') # unprivileged
handle_pos[2] -= 0.025  # handle_pos is slightly high sometimes

# pickup quat
gripper_down_quat = np.array([0, 1, 0, 0]) # handover quat
gripper_pick_quat = np.array([0, 0.707, 0.707, 0]) # down grip quat

# handover pos
arm0_pos = np.array([0.44, 0.0, 0.0]) # approx init positions
arm1_pos = np.array([1.18, 0.0, 0.0])
handover_pos = (arm0_pos + arm1_pos) / 2
handover_pos[2] = 0.10

# --- Sequence ---
# Arm0: pick up hammer at actual handle pose
open_gripper_arm0()
goto_pose_arm0(handle_pos, gripper_pick_quat, z_approach=0.15)

close_gripper_arm0()
goto_pose_arm0(handle_pos + np.array([0, 0, 0.1]), gripper_pick_quat)
goto_pose_arm0(handle_pos + np.array([0, 0, 0.2]), gripper_pick_quat)

# Arm0: move to handover (shifted toward arm1)
goto_pose_arm0(handover_pos, gripper_down_quat)

# Arm1 approach
arm1_quat = np.array([0, 0, 1, 0])
open_gripper_arm1()
goto_pose_arm1(handover_pos + np.array([0.1, 0, -0.01]), arm1_quat, z_approach=0.12) # account for hammer length
close_gripper_arm1()

# Arm0: release and retract
open_gripper_arm0()
goto_pose_arm0(handover_pos + np.array([-0.1, 0, 0.06]), gripper_down_quat)
"""


PRIVILEGED_ORACLE_CODE = """
import numpy as np

# get poses
handle_pos, handle_quat = get_hammer_pose() # privileged
handle_pos[2] -= 0.025  # handle_pos is slightly high sometimes
handle_pos[1] -= 0.035  # handle pos is in the middle

# pickup quat
gripper_down_quat = np.array([0, 1, 0, 0]) # handover quat
gripper_pick_quat = np.array([0, 0.707, 0.707, 0]) # down grip quat

# handover pos
arm0_pos = np.array([0.44, 0.0, 0.0]) # approx init positions
arm1_pos = np.array([1.18, 0.0, 0.0])
handover_pos = (arm0_pos + arm1_pos) / 2
handover_pos[2] = 0.10

# --- Sequence ---
# Arm0: pick up hammer at actual handle pose
open_gripper_arm0()
goto_pose_arm0(handle_pos, gripper_pick_quat, z_approach=0.15)

close_gripper_arm0()
goto_pose_arm0(handle_pos + np.array([0, 0, 0.1]), gripper_pick_quat)
goto_pose_arm0(handle_pos + np.array([0, 0, 0.2]), gripper_pick_quat)

# Arm0: move to handover (shifted toward arm1)
goto_pose_arm0(handover_pos, gripper_down_quat)

# Arm1 approach
arm1_quat = np.array([0, 0, 1, 0])
open_gripper_arm1()
goto_pose_arm1(handover_pos + np.array([0.1, 0, -0.01]), arm1_quat, z_approach=0.12) # account for hammer length
close_gripper_arm1()

# Arm0: release and retract
open_gripper_arm0()
goto_pose_arm0(handover_pos + np.array([-0.1, 0, 0.06]), gripper_down_quat)
"""

# ---------------------------- High-level Env -----------------------------
class TwoArmHandoverCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for two-arm handover task."""

    prompt = PROMPT
    oracle_code = PRIVILEGED_ORACLE_CODE


__all__ = [
    "TwoArmHandoverCodeEnv",
]
