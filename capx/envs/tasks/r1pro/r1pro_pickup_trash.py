from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a R1Pro robot with API described below.
Goal: pick up the blue can of soda on the floor. When you navigate to the object, you should keep in mind that the robot has a limited view and you may lose sight of the object if you get too close. You should look around by rotating your torso to look for the object when you are close to it but lost sight of it. If you are too far away from the object, you will not be able to reach and grasp the object. The best practice is to move towards the object and stop when the object is in the view and reachable. You can use sample_grasp_pose to find a grasp pose for the object. You should try back up and place the object in the center of view to get good grasp poses. until you find a successful grasp. There is a time limit of 600s to finish the task. So avoid getting stuck and try to finish the task in the given time limit.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
If you want to use numpy, scipy for spatial transformations, opencv, pytorch, or any other libraries, you need to import them explicitly.
Note that API may fail. Make sure the code is fault tolerant.
You should consider retrying, try and except, and retrying other combinations of APIs or write your own code to recreate the same capability.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""
ORACLE_CODE = """
# Code block 0
import numpy as np

def solve_task():    
    # Constants
    SODA_NAME = "blue can of soda"
    TRASH_CAN_NAME = "trash can"
    
    print(f"Looking for {SODA_NAME}...")

    # Search for soda
    found_soda = find_object_base_rotate(SODA_NAME)
    if not found_soda:
        # Try torso search
        found_soda = find_object_torso_rotate(SODA_NAME)
    
    if not found_soda:
        print("No more soda cans found. Task complete.")
        return

    # Get Soda Pose info
    # We need the point cloud (P_object) to navigate close to it if necessary.
    soda_pos, soda_quat, _, soda_points, _ = get_object_pose(SODA_NAME)
    
    if soda_pos is None:
        print("Detected soda but failed to estimate pose. Retrying search...")
        return
    
    # 3. Navigate to the soda
    
    dx = soda_pos[0] - get_robot_position()[0][0]
    dy = soda_pos[1] - get_robot_position()[0][1]
    yaw_to_soda = np.arctan2(dy, dx)
    
    # Target position: stop 0.7m short of the object along the vector
    dist = np.sqrt(dx**2 + dy**2)
    target_dist = max(0.0, dist - 0.7)
    nav_x = get_robot_position()[0][0] + (dx/dist) * target_dist
    nav_y = get_robot_position()[0][1] + (dy/dist) * target_dist
    
    nav_pose = np.array([nav_x, nav_y, yaw_to_soda])
    
    print(f"Navigating to soda at {nav_pose}...")
    navigate_to_pose(nav_pose)
    

    found_soda = find_object_torso_rotate(SODA_NAME)

    # 4. Grasp the Soda
    # We use arm 0 (left) for manipulation
    arm_id = 0 
    open_gripper(arm=arm_id)
    
    # Sample grasps
    # Returns lists of [simple, graspnet, topdown, 90deg]
    grasp_data = sample_grasp_pose(SODA_NAME)
    
    if grasp_data[0] is None:
        print("Could not generate grasps for this soda. Creating an obstacle/ignoring and trying next.")
        # In a real scenario, we might add it to an ignore list, but here we just loop.
        # Moving the robot slightly might help change the view.
        current_pos = get_robot_position()
        # back up a bit
        navigate_to_pose(np.array([current_pos[0][0]-0.2, current_pos[0][1], current_pos[2]]))

    pregrasp_poses, grasp_poses = grasp_data
    
    success_grasp = False
    
    # Iterate through available grasp strategies
    for i in range(len(pregrasp_poses)):
        pre_p = pregrasp_poses[i] # (pos, quat_xyzw)
        gr_p = grasp_poses[i]     # (pos, quat_xyzw)
        
        # Since the API signatures for grasp_pose are (pos, quat_xyzw) tuples usually:
        # sample_grasp_pose doc says it returns lists of poses.
        # grasp_object doc takes (pregrasp_pose, grasp_pose, name, arm).
        
        # The sample_grasp_pose returns tuples of (pos, quat) usually.
        # Let's verify format. The doc says "Returns: pregrasp_poses: List...".
        # Assuming elements are compatible with grasp_object inputs.
        
        print(f"Attempting grasp strategy {i}...")
        
        try:
            # Execute grasp
            grasp_object(pre_p, gr_p, SODA_NAME, arm=arm_id)
            if check_object_in_hand(arm=arm_id):
                success_grasp = True
                break
            else:
                print("Grasp reported success but check_object_in_hand failed.")
                open_gripper(arm=arm_id) # Release failed grasp
        except Exception as e:
            print(f"Grasp execution failed: {e}")
            continue

solve_task()
"""


# ---------------------------- High-level Env -----------------------------
class R1ProTrashCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for R1Pro trash pickup using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "R1ProTrashCodeEnv",
]
