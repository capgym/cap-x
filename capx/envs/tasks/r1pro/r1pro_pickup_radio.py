from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a R1Pro robot with API described below.
Goal: pick up the red radio on the table. You can use sample_grasp_pose to find a grasp pose for the object. There maybe multiple grasp poses candidates for the object.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""
ORACLE_CODE = """
import numpy as np

# debug_pyroik()

# get robot current position
robot_pos, robot_quat, robot_yaw = get_robot_position()
print(f"Robot position at start: {robot_pos}, {robot_quat}")

# find the radio object
find_radio = find_object_base_rotate("red radio")
print(f"Radio found: {find_radio}")

# get radio pose
radio_pos, radio_qua, _, P_radio, radio_obb = get_object_pose("red radio")
print(f"Radio points: {P_radio}")
print(f"Radio pose: {radio_pos}, {radio_qua}")

# # get the table pose
table_pos, table_qua, _, P_table, table_obb = get_object_pose("table")
print(f"Table pose: {table_pos}, {table_qua}")
print(f"Table points: {P_table}")
# goto_radio()

#move to the radio
# P_table = np.asarray(P_table.points)
# P_radio = np.asarray(P_radio.points)
goal = get_navigation_pose(P_table, P_radio)
success = navigate_to_pose(goal)
print(f"Navigation to radio", goal)
print(f"Navigation to radio success: {success}")

save_current_observation("pregrasp_observation")

find_radio = find_object_torso_rotate("red radio")
print(f"Radio found: {find_radio}")

if not find_radio:
    find_radio = find_object_base_rotate("red radio")
    if not find_radio:
        find_radio = find_object_torso_rotate("red radio")

radio_pos, radio_qua, _, P_radio, radio_obb = get_object_pose("red radio")
# print(f"Radio points: {P_radio}")
# print(f"Radio pose: {radio_pos}, {radio_qua}")

# grasp the radio
pregrasp_poses, grasp_poses = sample_grasp_pose("red radio")
for i, (pregrasp_pose, grasp_pose) in enumerate(zip(pregrasp_poses, grasp_poses)):
    grasp_object(pregrasp_pose, grasp_pose, "red radio")
    if check_object_in_hand():
        break
    if i ==2:
        break

# save_current_observation("grasp_observation")
"""


# ---------------------------- High-level Env -----------------------------
class R1ProRadioCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for R1Pro radio pickup using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "R1ProRadioCodeEnv",
]
