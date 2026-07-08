from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling the right arm of a real Unitree G1 robot as a single-arm robot with the API described below.
Goal: grasp a plastic water bottle and lift it off the table.

Key rules:
- Use the right Dex3 hand as the gripper. Available hand APIs: set_gripper_trigger_squeeze(trigger, squeeze), open_gripper(), close_gripper(), close_index_pinch(), close_middle_pinch(), and move_hand_joints(joints).
- Dex3 trigger/squeeze semantics: trigger closes thumb+index for a 2D pinch; squeeze closes thumb+middle; trigger=1 and squeeze=1 closes thumb/index/middle for a bottle grasp.
- Treat the arm as a 7-DoF single-arm robot. Do not distinguish left and right arms in code.
- Prefer sample_grasp_center_pose("plastic water bottle") and grasp_at_pinch_center(pos, quat). These APIs use the Junhao-style front policy: contact/visual center as pinch target, fixed vertical front-palm rotation, Dex3 closed-pinch offset converted to the palm IK target, and a rate-limited streamed horizontal pregrasp-to-grasp approach that keeps Z stable. Do not send this target directly to goto_pose().
- For low-level manual control, use move_to_pinch_pregrasp(pos, quat), set_gripper_trigger_squeeze(0, 0), pre_pos = pos - np.array([0.20, 0.0, 0.0]), move_pinch_center_horizontal_line(pre_pos, pos, quat), set_gripper_trigger_squeeze(1, 1), then lift with move_pinch_center_to_pose(lift_pos, quat).
- For table-top grasps, the pinch pregrasp retreats 0.20m along world -X. The pregrasp-to-grasp segment should be one rate-limited streamed joint trajectory generated from horizontal Cartesian IK waypoints, not multiple blocking move_to_joints/goto_pose calls.
- Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
bottle_name = "plastic water bottle"

# Get the Junhao-style front-policy pinch target for the plastic water bottle.
grasp_pos, grasp_quat = sample_grasp_center_pose(bottle_name)

# Execute the standard G1 Dex3 bottle grasp: front pregrasp, open hand, put the
# closed pinch center on the visual grasp center, close thumb/index/middle, lift.
grasp_at_pinch_center(grasp_pos, grasp_quat, trigger=1.0, squeeze=1.0, lift_dz=0.12)
"""


class UnitreeG1GraspBottleCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for grasping a plastic bottle with Unitree G1 right arm."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "UnitreeG1GraspBottleCodeEnv",
]
