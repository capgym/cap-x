from capx.envs.tasks.base import CodeExecutionEnvBase
EMPTY_CODE = ""

DEBUG_CODE = """
import numpy as np
import PIL.Image as Image

ee_pos, ee_quat = get_ee_pose()
ee_pos = np.asarray(ee_pos, dtype=np.float64).copy()
ee_quat = np.asarray(ee_quat, dtype=np.float64).copy()
# move ee to the left so it can see the basket
ee_pos[1] += 0.12
goto_pose(ee_pos, ee_quat)

object_name = "basket"
mask_pc_dict = get_object_3d_points_and_masks_from_language(object_name, use_multiview=True)
np.save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/object_pc_points_world.npy", mask_pc_dict["points_3d"])
filtered_points, _ = filter_noise(mask_pc_dict["points_3d"])
np.save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/object_pc_points_world_filtered.npy", filtered_points)


obs = get_observation()
wrist_sam3_mask = max(segment_sam3_text_prompt(obs["robot0_eye_in_hand"]["images"]["rgb"], object_name), key=lambda x: x["score"])
agentview_sam3_mask = max(segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], object_name), key=lambda x: x["score"])

def _overlay_mask_red(rgb_array, mask_array, alpha=0.4):
    overlay = np.array(rgb_array, dtype=np.float64).copy()
    mask_bool = np.asarray(mask_array, dtype=bool)
    if mask_bool.ndim == 3:
        mask_bool = mask_bool.any(axis=-1)
    overlay[mask_bool, 0] = overlay[mask_bool, 0] * (1 - alpha) + 255 * alpha
    overlay[mask_bool, 1] = overlay[mask_bool, 1] * (1 - alpha) + 0 * alpha
    overlay[mask_bool, 2] = overlay[mask_bool, 2] * (1 - alpha) + 0 * alpha
    return Image.fromarray(overlay.astype(np.uint8))

ext_rgb = obs["agentview"]["images"]["rgb"]
wrist_rgb = obs["robot0_eye_in_hand"]["images"]["rgb"]

Image.fromarray(ext_rgb).save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/external_cam.png")
Image.fromarray(wrist_rgb).save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/wrist_cam.png")
_overlay_mask_red(ext_rgb, mask_pc_dict["agentview_mask"]).save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/external_cam_obj_mask.png")
_overlay_mask_red(wrist_rgb, mask_pc_dict["wrist_mask"]).save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/wrist_cam_obj_mask.png")
_overlay_mask_red(wrist_rgb, wrist_sam3_mask["mask"]).save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/wrist_cam_sam3_mask.png")
_overlay_mask_red(ext_rgb, agentview_sam3_mask["mask"]).save(f"/home/karimelrafi/hyrl_release/HyRL/debug_grasping_mar12/external_cam_sam3_mask.png")
"""

PICK_X_PLACE_IN_BASKET_CODE = """
# object_name = "blue alphabet soup can"
object_name = "milk carton"


import numpy as np
import viser.transforms as vtf
import os

save_debug_data = False
use_multiview_grasps = True
use_multiview_img = True

states = ['find_object', 'grasp_object', 'lift_object', 'place_object']
current_state = 'find_object'
while True:
    if current_state == 'find_object':
        open_gripper()
        goto_home_joint_position()
        mask_pc_dict = get_object_3d_points_and_masks_from_language(object_name)
        obj_mask = mask_pc_dict["agentview_mask"]
        if mask_pc_dict is None:
            print("Could not get 3D points and masks for" + object_name + ".")
            object_name = query_VLM("The segmentation perception model failed to segment the " + object_name + " object. When the object name was '" + object_name + "', what is the correct object name?")
            continue
        object_3d_points = mask_pc_dict["points_3d"]
        object_3d_points, _ = filter_noise(object_3d_points)
        object_oriented_bounding_box = get_oriented_bounding_box_from_3d_points(object_3d_points)
        grasp_pos, grasp_quat = get_top_down_grasp_from_obb(object_oriented_bounding_box)
        grasp_poses = [(grasp_pos, grasp_quat)]
        # # old GraspNet version
        # positions, quaternions, scores = get_grasp_poses(object_name, top_k=15, use_multiview=use_multiview_grasps)
        # grasp_poses = [(positions[i], quaternions[i]) for i in range(len(positions))]

        current_state = 'grasp_object'
    elif current_state == 'grasp_object':
        open_gripper()
        success, joint_traj, goalset_idx = plan_grasp_trajectory(
            object_name,
            object_mask=obj_mask,
            grasp_poses=grasp_poses,
            use_world_collision=True,
            robot_distance_threshold=0.15,
            robot_collision_sphere_buffer=-0.02,
            collision_activation_distance=0.02,
        )
        if not success or joint_traj is None:
            # raise RuntimeError("plan_grasp_trajectory failed; no collision-free path found.")
            print("plan_grasp_trajectory failed; no collision-free path found.")
            current_state = 'find_object'
            continue
        execute_joint_trajectory(joint_traj, subsample=2)
        close_gripper()

        response = query_VLM_yes_no("Is the object grasped?", use_multiview=use_multiview_img)
        if response:
            current_state = 'lift_object'
        else:
            current_state = 'find_object'
    elif current_state == 'lift_object':
        # LIFT_DZ = 0.12
        # # have gripper perpendicular to the camera so can is more visible
        # ee_pos, _ = get_ee_pose()
        # ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        # ee_pos[2] += LIFT_DZ
        # goto_pose(ee_pos, ee_quat)
        LIFT_DZ = 0.12
        ee_pos, ee_quat = get_ee_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        ee_quat = np.asarray(ee_quat, dtype=np.float64).copy()
        ee_pos[2] += LIFT_DZ
        goto_pose(ee_pos, ee_quat)

        # have gripper perpendicular to the camera so can is more visible
        ee_pos, _ = get_ee_pose()
        ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        goto_pose(ee_pos, ee_quat)

        response = query_VLM_yes_no("Is the " + object_name + " lifted?", use_multiview=use_multiview_img)
        if response:
            obj_name_dict = get_object_3d_points_and_masks_from_language(object_name, use_multiview=False)
            obj_obb = get_oriented_bounding_box_from_3d_points(obj_name_dict["points_3d"])
            obj_pos_lift = obj_obb["center"]
            obj_quat_lift = vtf.SO3.from_matrix(obj_obb["R"]).wxyz
            obj_mask_lift = obj_name_dict["agentview_mask"]
            current_state = 'place_object'
        else:
            current_state = 'find_object'
        # break
    elif current_state == 'place_object':
        basket_mask_pc_dict = get_object_3d_points_and_masks_from_language("basket")
        basket_mask = basket_mask_pc_dict["agentview_mask"]
        basket_3d_points = basket_mask_pc_dict["points_3d"]
        basket_3d_points, _ = filter_noise(basket_3d_points)
        basket_oriented_bounding_box = get_oriented_bounding_box_from_3d_points(basket_3d_points)
        basket_pos = basket_oriented_bounding_box['center']
        basket_pos = np.asarray(basket_pos, dtype=np.float64).copy()
        basket_pos[2] += 0.3
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])

        success, joint_traj = plan_with_grasped_object(
            (basket_pos, target_quat),
            object_name,
            object_pose=(obj_pos_lift, obj_quat_lift),
            object_mask=obj_mask_lift,
        )
        if success and joint_traj is not None:
            execute_joint_trajectory(joint_traj, subsample=2)
            open_gripper()
        else:
            # raise RuntimeError("Could not plan to basket.")
            print("Could not plan to basket.")
            current_state = 'find_object'
            continue

        # # shift robot end effector slightly back so it can better see inside of the basket 
        # # NOTE: idt this really helps much :(
        # shifted_basket_pos = basket_pos + np.array([-0.05, 0.0, 0.0])
        # goto_pose(shifted_basket_pos, target_quat)
        query_str = "Is the " + object_name + " in the basket?"
        response = query_VLM_yes_no(query_str, use_multiview=use_multiview_img)
        # obs = get_observation()
        # wrist_rgb = obs["robot0_eye_in_hand"]["images"]["rgb"]
        # import PIL.Image as Image
        # Image.fromarray(wrist_rgb).save(f"/home/karim/hyrl_alt2/HyRL/debug_grasping_feb21/wrist_rgb.png")
        if response:
            break
        else:
            current_state = 'find_object'
"""

PICK_MILK_PLACE_IN_BASKET_CODE = """
object_name = "milk carton"

open_gripper()
goto_home_joint_position()
mask_pc_dict = get_object_3d_points_and_masks_from_language(object_name)
obj_mask = mask_pc_dict["agentview_mask"]
if mask_pc_dict is None:
    print("Could not get 3D points and masks for" + object_name + ".")
    object_name = query_VLM("The segmentation perception model failed to segment the " + object_name + " object. When the object name was '" + object_name + "', what is the correct object name?")
    # continue
object_3d_points = mask_pc_dict["points_3d"]
object_3d_points, _ = filter_noise(object_3d_points)
object_oriented_bounding_box = get_oriented_bounding_box_from_3d_points(object_3d_points)
grasp_pos, grasp_quat = get_top_down_grasp_from_obb(object_oriented_bounding_box)
grasp_poses = [(grasp_pos, grasp_quat)]

# goto_pose(grasp_pos, grasp_quat)

success, joint_traj, goalset_idx = plan_grasp_trajectory(
    object_name,
    object_mask=obj_mask,
    grasp_poses=grasp_poses,
    use_world_collision=True,
    robot_distance_threshold=0.15,
    robot_collision_sphere_buffer=-0.02,
    collision_activation_distance=0.02,
)

execute_joint_trajectory(joint_traj, subsample=2)
close_gripper()
"""

lgs_task3_open_drawer_code = """
# Code block 1
import numpy as np

# 1. Inspect the scene to find the handle again (since it moved)
obs = get_observation()
agentview = obs["agentview"]
rgb = agentview["images"]["rgb"]
depth = agentview["images"]["depth"]
intrinsics = agentview["intrinsics"]
extrinsics = agentview["pose_mat"]

# 2. Open the drawer fully
handle_text_prompt = "handle of the top drawer"
handle_masks = segment_sam3_text_prompt(rgb, handle_text_prompt)

if handle_masks:
    best_handle = max(handle_masks, key=lambda x: x["score"])
    handle_mask = best_handle["mask"]
    
    # Plan grasp for the handle
    grasp_poses_cam, grasp_scores = plan_grasp(depth, intrinsics, handle_mask)
    
    if len(grasp_scores) > 0:
        best_idx = np.argmax(grasp_scores)
        best_grasp_cam = grasp_poses_cam[best_idx]
        best_grasp_world = extrinsics @ best_grasp_cam
        
        grasp_pos, grasp_quat = decompose_transform(best_grasp_world)
        
        # Approach vector (gripper Z)
        approach_vec = best_grasp_world[:3, 2]
        
        # Pre-grasp (approach)
        pre_grasp_pos = grasp_pos - approach_vec * 0.10
        
        open_gripper()
        move_to_joints(solve_ik(pre_grasp_pos, grasp_quat))
        move_to_joints(solve_ik(grasp_pos, grasp_quat))
        close_gripper()
        
        # Pull the drawer significantly more this time.
        # Based on previous logic, pulling towards +Y seemed correct for "right face".
        # We increase the distance to ensure it opens fully.
        pull_distance = 0.35 
        pull_target_pos = grasp_pos + np.array([0, pull_distance, 0])
        
        # Execute pull
        move_to_joints(solve_ik(pull_target_pos, grasp_quat))
        
        open_gripper()
        
        # Retreat
        retreat_pos = pull_target_pos + np.array([0, -0.1, 0.1])
        move_to_joints(solve_ik(retreat_pos, grasp_quat))
        
        # 3. Pick the Bowl
        # Refresh observation
        obs = get_observation()
        rgb = obs["agentview"]["images"]["rgb"]
        depth = obs["agentview"]["images"]["depth"]
        intrinsics = obs["agentview"]["intrinsics"]
        
        bowl_text_prompt = "silver bowl"
        bowl_masks = segment_sam3_text_prompt(rgb, bowl_text_prompt)
        
        if bowl_masks:
            best_bowl = max(bowl_masks, key=lambda x: x["score"])
            bowl_mask = best_bowl["mask"]
            
            grasp_poses_cam, grasp_scores = plan_grasp(depth, intrinsics, bowl_mask)
            best_bowl_grasp_world, _ = select_top_down_grasp(grasp_poses_cam, grasp_scores, extrinsics)
            
            if best_bowl_grasp_world is not None:
                bowl_pos, bowl_quat = decompose_transform(best_bowl_grasp_world)
                
                # Pre-grasp
                pre_bowl_pos = bowl_pos + np.array([0, 0, 0.15])
                move_to_joints(solve_ik(pre_bowl_pos, bowl_quat))
                move_to_joints(solve_ik(bowl_pos, bowl_quat))
                close_gripper()
                
                # Lift
                lift_pos = bowl_pos + np.array([0, 0, 0.25])
                move_to_joints(solve_ik(lift_pos, bowl_quat))
                
                # 4. Place in Drawer
                # Calculate drop position based on the handle's final pulled position.
                # If we pulled to `pull_target_pos` (handle center), the drawer interior is 'behind' it.
                # Assuming pull was along +Y, "behind" is -Y.
                # We drop it inside the drawer volume.
                drop_pos = pull_target_pos + np.array([0, -0.15, 0.05])
                
                # Move to pre-drop (high)
                pre_drop_pos = drop_pos + np.array([0, 0, 0.15])
                move_to_joints(solve_ik(pre_drop_pos, bowl_quat))
                
                # Move to drop
                move_to_joints(solve_ik(drop_pos, bowl_quat))
                open_gripper()
                
                # Retreat
                move_to_joints(solve_ik(pre_drop_pos, bowl_quat))
            else:
                print("No top-down grasp for bowl.")
        else:
            print("Bowl not found.")
    else:
        print("No grasp for handle found.")
else:
    print("Handle not found.")

# Code block 2
import numpy as np

# 1. Inspect the scene
# Since the drawer is already open (according to the observation), we need to find the bowl and place it into the drawer.
obs = get_observation()
agentview = obs["agentview"]
rgb = agentview["images"]["rgb"]
depth = agentview["images"]["depth"]
intrinsics = agentview["intrinsics"]
extrinsics = agentview["pose_mat"]

# 2. Locate the bowl
bowl_text_prompt = "silver bowl"
bowl_masks = segment_sam3_text_prompt(rgb, bowl_text_prompt)

if not bowl_masks:
    print("Could not find the bowl.")
else:
    best_bowl = max(bowl_masks, key=lambda x: x["score"])
    bowl_mask = best_bowl["mask"]
    
    # Plan grasp for the bowl
    # We want a top-down grasp for the bowl
    grasp_poses_cam, grasp_scores = plan_grasp(depth, intrinsics, bowl_mask)
    
    best_bowl_grasp_world, best_score = select_top_down_grasp(grasp_poses_cam, grasp_scores, extrinsics)
    
    if best_bowl_grasp_world is not None:
        bowl_pos, bowl_quat = decompose_transform(best_bowl_grasp_world)
        
        # Pre-grasp (approach from above)
        pre_grasp_pos = bowl_pos + np.array([0, 0, 0.15])
        
        open_gripper()
        
        # Move to pre-grasp
        joints_pre = solve_ik(pre_grasp_pos, bowl_quat)
        move_to_joints(joints_pre)
        
        # Move to grasp
        joints_grasp = solve_ik(bowl_pos, bowl_quat)
        move_to_joints(joints_grasp)
        
        close_gripper()
        
        # Lift the bowl
        lift_height = 0.20
        lift_pos = bowl_pos + np.array([0, 0, lift_height])
        joints_lift = solve_ik(lift_pos, bowl_quat)
        move_to_joints(joints_lift)
        
        # 3. Place in the drawer
        # We need to find the open drawer to place the bowl inside.
        # Since the drawer is open, we can try to segment the "inside of the open drawer" or the handle again to get a reference.
        # Let's find the handle of the top drawer again to use as a reference point.
        # The handle should now be pulled out (towards +Y in world frame, or +X depending on setup, but typically +Y based on previous context).
        
        # Re-observe to find handle new position
        obs = get_observation()
        rgb_new = obs["agentview"]["images"]["rgb"]
        depth_new = obs["agentview"]["images"]["depth"]
        intrinsics_new = obs["agentview"]["intrinsics"]
        
        handle_text_prompt = "handle of the top drawer"
        handle_masks = segment_sam3_text_prompt(rgb_new, handle_text_prompt)
        
        drop_target_pos = None
        
        if handle_masks:
            best_handle = max(handle_masks, key=lambda x: x["score"])
            handle_mask = best_handle["mask"]
            
            # Get 3D point of the handle
            # We can pick a point from the mask and deproject it
            ys, xs = np.where(handle_mask)
            if len(ys) > 0:
                # Use the center of the mask
                y_c, x_c = int(np.mean(ys)), int(np.mean(xs))
                z_c = depth_new[y_c, x_c]
                if z_c > 0:
                    handle_pos_world = pixel_to_world_point(x_c, y_c, z_c, intrinsics_new, extrinsics)
                    
                    # Heuristic: The drawer interior is 'behind' the handle.
                    # Previous reasoning suggested pulling along Y. If handle is at Y_handle, drawer box is at Y < Y_handle.
                    # Let's place it roughly 10-15cm behind the handle in Y, and keep X similar.
                    # NOTE: "In image space the right is world frame positive Y". If the drawer is on the right side of the image, 
                    # and we pull it open, we pull towards positive Y? No, usually drawers open forward relative to the cabinet front.
                    # If the camera looks from front, and right is +Y, then cabinet front face normal is likely +X (backwards into camera is +X?? No wait).
                    # "backward into the camera is world frame positive X". So camera looks towards -X.
                    # This means the robot and table are in -X direction relative to camera? Or camera is at +X looking -X?
                    # Let's rely on relative position. 
                    # If the handle is at `handle_pos_world`, and the drawer is open, the "inside" is usually along the axis of opening.
                    # Assuming standard drawer opening, we place it slightly "in" from the handle.
                    # Let's try an offset. If handle is at (x, y, z), we place at (x, y - 0.15, z + small_offset)?
                    # If the handle is on the "right face", it implies the drawer moves along Y axis.
                    # If we pulled it open, it moved +Y (to the right in image). So inside is -Y relative to handle.
                    
                    drop_target_pos = handle_pos_world + np.array([0.0, -0.15, 0.05])
        
        if drop_target_pos is None:
            # Fallback if handle detection fails or depth is bad:
            # Use a hardcoded guess relative to the lift position or just blindly rely on previous successful pull coordinate?
            # Let's assume the bowl lift position is safe and move relative to the handle if found, 
            # otherwise we might be stuck. But since the drawer is open, handle should be visible.
            print("Could not robustly detect handle 3D position. Trying a fallback heuristic based on bowl position.")
            # This is risky, but better than crashing.
            # Assuming bowl was picked from table, maybe (0, 0, 0.2) relative to table center?
            # Let's just print error and stop if we can't find target.
            print("Cannot place bowl without target.")
        else:
            # Execute Place
            # Move to high approach position above drawer
            pre_drop_pos = drop_target_pos + np.array([0, 0, 0.15])
            
            joints_pre_drop = solve_ik(pre_drop_pos, bowl_quat)
            move_to_joints(joints_pre_drop)
            
            # Lower into drawer
            joints_drop = solve_ik(drop_target_pos, bowl_quat)
            move_to_joints(joints_drop)
            
            open_gripper()
            
            # Retreat
            joints_retreat = solve_ik(pre_drop_pos, bowl_quat)
            move_to_joints(joints_retreat)
            
    else:
        print("Could not find a valid grasp for the bowl.")
"""

class FrankaLiberoCodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for any LIBERO task.

    Unlike the task-specific subclasses (e.g. FrankaLiberoPickAlphabetSoupCodeEnv),
    this class does not hardcode a prompt or oracle code.  Both are expected to
    come from the YAML config via ``CodeExecEnvConfig.prompt``.

    The LIBERO suite_name and task_id are specified on the low-level env
    (``FrankaLiberoEnv``) in the YAML config, so no per-task Python file is needed.
    """

    # oracle_code = PICK_X_PLACE_IN_BASKET_CODE
    # oracle_code = PICK_MILK_PLACE_IN_BASKET_CODE
    # oracle_code = DEBUG_CODE
    # oracle_code = EMPTY_CODE


__all__ = ["FrankaLiberoCodeEnv"]
