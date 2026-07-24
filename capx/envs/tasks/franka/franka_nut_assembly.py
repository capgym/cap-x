from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.security import validate_generated_program

# PROMPT = """
# You are controlling a Franka Emika robot with API described below.
# Goal: grasp and insert the `square nut` with its handle onto the small `square peg`.
# Note that you would grasp the nut by its handle, not the center of the nut.
# There's a rigid transform that you need to compute, and you would need to reapply the rigid transform when inserting the nut onto the peg.
# You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
# The functions (APIs) below are already imported to the environment.
# If you want to use numpy, or scipy for spatial transformations, you need to import it explicitly.
# """

# allowing code fences
PROMPT = """
You are controlling a Franka Emika robot with API described below.
Goal: grasp and insert the `brown square nut` onto the `brown square block`.
Note that you would grasp the nut by its handle. You can try language query `extruded handle of the brown square nut` to get a good grasp at the handle.
The brown square nut and the extruded handle of the brown square nut are part of the same rigid body.
The grasp pose query for 'extruded handle of the brown square nut' returns an end-effector pose expressed in world frame, located on the handle region.
Before moving the robot, save the handle pose relative to the nut center with relative_pose. After grasping the handle, compose the square block pose with that saved relative pose to obtain the insertion handle pose.
Approach the handle from above, close the gripper, return home for a stable IK seed, approach the insertion handle pose from above, lower 2 cm in world Z, and release.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment.
If you want to use numpy, or scipy for spatial transformations, you need to import it explicitly.
Call the provided API functions directly. Do not redefine them or import external robot-control modules.
Only numpy and scipy imports are allowed. Do not use try/except. Your response must be one compact, self-contained executable program.
"""

ROUND_PROMPT = """
You are controlling a Franka Emika robot with API described below.
Goal: grasp and insert the round nut onto the round peg.
Grasp the nut by its extruded handle. Before moving the robot, save the handle pose relative to the nut center with relative_pose. After grasping the handle, compose the round peg pose with that saved relative pose to obtain the insertion handle pose.
Approach the handle from above, close the gripper, return home for a stable IK seed, approach the insertion handle pose from above, lower 2 cm in world Z, and release.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment.
If you want to use numpy, or scipy for spatial transformations, you need to import it explicitly.
Call the provided API functions directly. Do not redefine them or import external robot-control modules.
Only numpy and scipy imports are allowed. Do not use try/except. Your response must be one compact, self-contained executable program.
"""

SINGLE_PROMPT = """
You are controlling a Franka Emika robot with API described below.
Goal: insert the one active nut onto its matching peg. The active nut is randomly square or round on every reset.
First call get_active_nut_types() and use the returned type in the object queries. Grasp the active nut by its extruded handle. Before moving the robot, save the handle pose relative to the nut center with relative_pose. After grasping, compose the matching peg pose with that saved relative pose to obtain the insertion handle pose.
Approach the handle from above, close the gripper, return home for a stable IK seed, approach the insertion handle pose from above, lower 2 cm in world Z, and release.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment.
If you want to use numpy, or scipy for spatial transformations, you need to import it explicitly.
Call the provided API functions directly. Do not redefine them or import external robot-control modules.
Only numpy and scipy imports are allowed. Do not use try/except. Your response must be one compact, self-contained executable program.
"""

FULL_PROMPT = """
You are controlling a Franka Emika robot with API described below.
Goal: insert both the square nut and the round nut onto their matching pegs.
For each nut type returned by get_active_nut_types(), grasp that nut by its extruded handle. Before moving it, save the handle pose relative to the nut center with relative_pose. Compose the matching peg pose with the saved relative pose to obtain the insertion handle pose. Approach from above, close the gripper, return home for a stable IK seed, approach the insertion pose from above, lower 2 cm in world Z, release, and return home before handling the next nut.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment.
If you want to use numpy, or scipy for spatial transformations, you need to import it explicitly.
Call the provided API functions directly. Do not redefine them or import external robot-control modules.
Only numpy and scipy imports are allowed. Do not use try/except. Your response must be one compact, self-contained executable program.
"""
ORACLE_CODE = """
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pos, quat):
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy expects xyzw
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = pos
    return mat

def matrix_to_pose(mat):
    pos = mat[:3, 3]
    rot = R.from_matrix(mat[:3, :3])
    quat_xyzw = rot.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz

def flip_xy_axis(mat: np.ndarray) -> np.ndarray:
    # mat is a 3x3 rotation matrix
    flip_mat = np.array(
        [[-1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]]
    )
    return mat @ flip_mat

# 1. Sample grasp pose for nut (handle)
handle_pos, handle_quat = sample_grasp_pose("extruded handle of the brown square nut")

# 2. Get poses of nut (center) and peg
nut_pos, _ = get_object_pose("white hollow center of the brown square nut")
peg_pos, peg_quat = get_object_pose("square block")

# 4. Compute transform from handle to nut center
v_world = nut_pos - handle_pos
handle_orientation = R.from_quat([handle_quat[1], handle_quat[2], handle_quat[3], handle_quat[0]]).as_matrix()
print("direction of handle: ", v_world @ handle_orientation[:, 0])
if v_world @ handle_orientation[:, 0] < 0:
    print("flipping the xy axis")
    # flip the xy axis 
    handle_orientation = flip_xy_axis(handle_orientation)
    handle_xyzw = R.from_matrix(handle_orientation).as_quat()
    handle_quat = np.array([handle_xyzw[3], handle_xyzw[0], handle_xyzw[1], handle_xyzw[2]])
else:
    print("x axis direction is correct")

# 3. Execute motion: open gripper, go to grasp pose, close, then insert
goto_pose(handle_pos, handle_quat, z_approach=0.05)  # approach from above
close_gripper()

# 3.5. Return to home joint position so that it can have a better ik solve
goto_home_joint_position()

T_handle = pose_to_matrix(handle_pos, handle_quat)
T_nut = pose_to_matrix(nut_pos, handle_quat)
T_handle_to_center = np.linalg.inv(T_nut) @ T_handle

# 5. Compute desired handle pose for insertion onto peg
T_peg = pose_to_matrix(peg_pos, peg_quat)
T_desired_handle = T_peg @ T_handle_to_center
desired_handle_pos, desired_handle_quat = matrix_to_pose(T_desired_handle)

# 6. Final insertion: move slightly along z to ensure contact
goto_pose(desired_handle_pos, desired_handle_quat, z_approach=0.05)  # approach to pe
final_pos = desired_handle_pos + np.array([0, 0, -0.02])  # lower 2 cm
goto_pose(final_pos, desired_handle_quat, z_approach=0.0)  # final insertion

# 7. Release nut if needed
open_gripper()
"""


# ---------------------------- High-level Env -----------------------------
class FrankaNutAssemblyCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka nut assembly using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE

    def step(self, action: str):
        guard = validate_generated_program(action)
        if guard.allowed:
            return super().step(action)

        observation = self._get_observation()
        reward = self.compute_reward()
        task_completed = self.low_level_env.task_completed()
        info = {
            "sandbox_rc": 1,
            "stdout": "",
            "stderr": guard.reason,
            "task_prompt": self._task_prompt,
            "task_completed": task_completed,
        }
        return observation, reward, False, False, info


class FrankaNutAssemblyRoundCodeEnv(FrankaNutAssemblyCodeEnv):
    prompt = ROUND_PROMPT
    oracle_code = None


class FrankaNutAssemblySingleCodeEnv(FrankaNutAssemblyCodeEnv):
    prompt = SINGLE_PROMPT
    oracle_code = None


class FrankaNutAssemblyFullCodeEnv(FrankaNutAssemblyCodeEnv):
    prompt = FULL_PROMPT
    oracle_code = None


__all__ = [
    "FrankaNutAssemblyCodeEnv",
    "FrankaNutAssemblyFullCodeEnv",
    "FrankaNutAssemblyRoundCodeEnv",
    "FrankaNutAssemblySingleCodeEnv",
]
