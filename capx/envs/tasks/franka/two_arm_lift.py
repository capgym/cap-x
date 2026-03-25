from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a two-arm Franka Emika robot system on opposite sides of the table with API described below.
Goal: The two arms should coordinate to lift a pot. Arm 0 should grasp handle 0, and Arm 1 should grasp handle 1. Then both arms should lift the pot up at the same height.

Guidance:
- The pot is in the center of the table and has two handles on opposite sides.
- You may want to slightly lift both arms first to avoid occluding the pot and handles.
- Use `get_handle0_pos()` to get the bounding box center position of handle 0 (returns a single 3D array).
- Use `get_handle1_pos()` to get the bounding box center position of handle 1 (returns a single 3D array).
- Avoid using top-down grasps, i.e. avoid using quaternion wxyz [0, 0, 1, 0] for the approach.
- Sideways grasps are preferred with the y-axis of the gripper aligned with the world z-axis.
- Coordinate system: All pose functions accept/return positions in robot0's base frame.

You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) above are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""

ORACLE_CODE = """
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# --- Get poses ---
# Handle both privileged (2 values) and non-privileged (3 values) APIs

use_privileged_api = False
if callable(globals().get("get_handle0_pose")):
    use_privileged_api = True


if not use_privileged_api:
    # Non-privileged API: returns only bbox_center
    
    # Get current gripper poses for orientation
    gripper0_pos, gripper0_quat = get_arm0_gripper_pose()
    gripper1_pos, gripper1_quat = get_arm1_gripper_pose()
    
    # Define gripper orientations for approach
    gripper0_wxyz = np.array([0.7071, 0.7071, 0, 0])
    gripper1_wxyz = np.array([0.7071, -0.7071, 0, 0])
    
    # Move grippers up to avoid occlusion
    goto_pose_arm0(gripper0_pos, gripper0_quat, z_approach=0.1)
    goto_pose_arm1(gripper1_pos, gripper1_quat, z_approach=0.1)
    
    handle0_bbox_center = get_handle0_pos()
    handle1_bbox_center = get_handle1_pos()
    
    # Adjust bbox center positions to account for handle offsets
    handle0_bbox_center[1] += 0.15
    handle1_bbox_center[1] -= 0.15
    
    # Open grippers
    open_gripper_arm0()
    open_gripper_arm1()
    
    # Approach handles using bbox center positions with offset
    goto_pose_arm0(handle0_bbox_center, gripper0_wxyz, z_approach=-0.02)
    goto_pose_arm1(handle1_bbox_center, gripper1_wxyz, z_approach=-0.02)
    
    # Adjust bbox center back to original position
    handle0_bbox_center[1] -= 0.15
    handle1_bbox_center[1] += 0.15
    
    # Move to final grasp positions
    goto_pose_arm0(handle0_bbox_center, gripper0_wxyz, z_approach=-0.02)
    goto_pose_arm1(handle1_bbox_center, gripper1_wxyz, z_approach=-0.02)
    
    # Close grippers
    close_gripper_arm0()
    close_gripper_arm1()
    
    # Lift both arms together
    lift_height = 0.20
    handle0_bbox_center[2] += lift_height
    handle1_bbox_center[2] += lift_height
    goto_pose_both(handle0_bbox_center, gripper0_wxyz, handle1_bbox_center, gripper1_wxyz)
else:
    # Privileged API: returns (position, quaternion_wxyz)
    handle0_pos, handle0_wxyz = get_handle0_pose()
    handle1_pos, handle1_wxyz = get_handle1_pose()
    
    def get_target_gripper_pose(handle_wxyz):
        w, x, y, z = handle_wxyz
        obj_quat_scipy = [x, y, z, w]
        r_obj = R.from_quat(obj_quat_scipy)
        r_flip = R.from_euler('x', 180, degrees=True)
        
        r_gripper = r_obj * r_flip
        
        g_x, g_y, g_z, g_w = r_gripper.as_quat()
        return np.array([g_w, g_x, g_y, g_z])

    handle0_wxyz_target = get_target_gripper_pose(handle0_wxyz)
    handle1_wxyz_target = get_target_gripper_pose(handle1_wxyz)
    
    # --- Sequence ---
    
    # 1. Open grippers
    open_gripper_arm0()
    open_gripper_arm1()
    
    goto_pose_arm0(handle0_pos, handle0_wxyz_target, z_approach=0.2)
    goto_pose_arm1(handle1_pos, handle1_wxyz_target, z_approach=0.2)
    
    handle0_pos[2] -= 0.0
    handle1_pos[2] -= 0.0
    goto_pose_both(handle0_pos, handle0_wxyz_target, handle1_pos, handle1_wxyz_target)
    
    close_gripper_arm0()
    close_gripper_arm1()
    
    # 4. Lift both arms together
    lift_height = 0.20
    target_pos0 = handle0_pos.copy()
    target_pos0[2] += lift_height
    target_pos1 = handle1_pos.copy()
    target_pos1[2] += lift_height
    
    # Move arms up simultaneously
    goto_pose_both(target_pos0, handle0_wxyz_target, target_pos1, handle1_wxyz_target)
"""


# ---------------------------- High-level Env -----------------------------
class TwoArmLiftCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for two-arm lift task."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "TwoArmLiftCodeEnv",
]
