"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Stack environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

import os
from typing import Any, Tuple, List
import numpy as np
from collections import deque

from capx.envs.base import BaseEnv
# from base_env import BaseEnv

import sys
import yaml
try:
    import omnigibson as og
except Exception as e:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "Behaviour not available; add submodule."
    ) from e
from omnigibson.macros import gm
# here = os.path.dirname(os.path.abspath(__file__))
# vendor_root = os.path.normpath(os.path.join(here, "..", "third_party", "BEHAVIOR-1K"))
# if os.path.isdir(vendor_root) and vendor_root not in sys.path:
#     sys.path.append(vendor_root)
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.utils.transform_utils import euler2quat, quat_multiply, quat2mat
from omnigibson.object_states.toggle import ToggledOn
import torch
import mediapy as media
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
import json
import math
from omnigibson.action_primitives.curobo import (
    holonomic_base_command_in_world_frame,
)
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
)
from scipy.spatial.transform import Rotation as R
from omnigibson.sensors.vision_sensor import VisionSensor
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import time
import random
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
)
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.HEADLESS = True
# OMNIGIBSON_HEADLESS=1  python...
def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def quat2yaw(q, degrees=False):
    if isinstance(q, torch.Tensor):
        q = q.tolist()
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n > 0.0:
        x, y, z, w = x/n, y/n, z/n, w/n
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw) if degrees else yaw

def get_grasp_poses_for_object_sticky(target_obj, object_obb):
    """
    Obtain a grasp pose for an object from top down, to be used with sticky grasping.
    The grasp pose should be in the world frame.

    Args:
        target_obj (StatefulObject): Object to get a grasp pose for

    Returns:
        list: List of grasp poses.
    """

    # aabb_min_world, aabb_max_world = target_obj.aabb
    # bbox_center_world = (aabb_min_world + aabb_max_world) / 2
    # bbox_extent_world = aabb_max_world - aabb_min_world
    bbox_center_world = torch.tensor(object_obb.center)
    bbox_extent_world = torch.tensor(object_obb.extent)

    grasp_center_pos = bbox_center_world + torch.tensor([0, 0, bbox_extent_world[2] / 2])
    towards_object_in_world_frame = bbox_center_world - grasp_center_pos
    towards_object_in_world_frame /= torch.norm(towards_object_in_world_frame)

    # Identity quaternion for top-down grasping (x-forward, y-right, z-down)
    grasp_quat = euler2quat(torch.tensor([0, 0, 0], dtype=torch.float32))

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_poses = [grasp_pose]

    return grasp_poses


class R1ProBehaviourLowLevel(BaseEnv):
    """ 
    Behavior-1K low-level environment.
    """

    def __init__(
        self,
        controller_cfg: str = "r1pro_primitives.yaml",
        privileged: bool = False,
        save_video: bool = False,
        activity_name: str | None = None,
        *args, **kwargs
    ) -> None:
        super().__init__()
        config_filename =  os.path.join(og.example_config_path, controller_cfg)
        self.controller_cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        if activity_name is not None:
            self.controller_cfg['task']['activity_name'] = activity_name
            # Load all rooms so any task can find its required objects
            self.controller_cfg['scene']['load_room_types'] = None
        self.task_name = self.controller_cfg['task']['activity_name']
        
        self._step_count = 0
        self._sim_step_count = 0
        
        self.cur_observation = None
        self.observation_buffer = deque(maxlen=10)
        
        
        # Load config file
        task_name = self.controller_cfg["task"]['activity_name']
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])
        
        self.env = og.Environment(configs=self.controller_cfg)
        
        self.robot = self.env.robots[0]
        self.task_relevant_obj = self.env.task.object_scope #keys and values of task relevant objects
        self.object_state_key = self.env.task.low_dim_obs_keys # object state values are in state['task']['low_dim']
        print("Initialized Behavior1KLowLevel environment for task: ", self.task_name)
        
        self.metrics = self.load_metrics()
        
        
        self.controller = StarterSemanticActionPrimitives(self.env, self.env.robots[0], enable_head_tracking=False)
        # torso_joint = self.controller.robot.joints["torso_joint2"]
        self.torso_idx_map = {
            "torso_joint1": 6,
            "torso_joint2": 7,
            "torso_joint3": 8,
            "torso_joint4": 9,
        }
        # self.torso_joint_limits = [ torso_joint.lower_limit, torso_joint.upper_limit]
        
        self.obs_buffer = {
            'robot': {
                      'ego': [],
                      'left_wrist': [],
                      'right_wrist': [],
                      },
            'external': [],
        }
        self.save_video = save_video
        self._frame_buffer: list[np.ndarray] = []
        self._record_frames = False
        self.step_num = 0
    
    def load_metrics(self) -> List[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]
    
    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Interal use only] Whether to load the hidden test instance.
        """
        self.env.reset()
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
            
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()
        for _ in range(5):
            og.sim.render()
        self.env.reset()
    
        
    def reset(
        self,
        seed: int = None,
        options: dict[str, Any] = None,
        *args, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_count = 0
        self._sim_step_count = 0
        self.obs_buffer = {
            'robot': {
                      'ego': [],
                      'left_wrist': [],
                      'right_wrist': [],
                      },
            'external': [],
        }
        
        # zero_action = np.zeros_like(self.env.robots[0].action_space.sample())
        # obs, reward, terminated, truncated, info = self.env.step(zero_action)
        
        for arm_name in self.robot.gripper_control_idx.keys():
            grpiper_control_idx = self.robot.gripper_control_idx[arm_name]
            self.robot.set_joint_positions(torch.ones_like(grpiper_control_idx), indices=grpiper_control_idx, normalized=True)
        self.robot.keep_still()
        
        for _ in range(5):
            og.sim.step()
        
        # self.env.scene.update_initial_file()
        self.env.reset()
        
        info = {
            "task_prompt": f"Complete the task: {self.task_name}"
        }
        if self.task_name == "turning_on_radio":
            self.initial_radio_pose = self.get_object_state("radio")[0].clone()
        if 'trash' in self.task_name:
            self.initial_trash_pose = self.get_object_state("can__of__soda.n.01_3")[0].clone()
            
        obs = self.get_observation()
        
        if options is not None and 'trial' in options:
            self.load_task_instance(int(options['trial']))
        elif seed is not None:
            self.load_task_instance(int(seed))
        else:
            self.load_task_instance(0)
            
        for metric in self.metrics:
            metric.start_callback(self.env)
            
        self.step_num = 0
        
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._step_count += 1
        # This is a fallback; normally FrankaControlApi methods are used
        obs, reward, terminated, truncated, info = self.env.step(action)
        # self.observation_buffer.append(obs)
        self.cur_observation = obs
        success = info['done']['success']
        obs_keys = list(obs.keys())
        robot_key = [key for key in obs_keys if 'robot' in key][0]
        if self.save_video:
            self.obs_buffer['robot']['ego'].append(obs[f'{robot_key}'][f'{robot_key}:zed_link:Camera:0']['rgb'][:,:,:3])
            self.obs_buffer['robot']['left_wrist'].append(obs[f'{robot_key}'][f'{robot_key}:left_realsense_link:Camera:0']['rgb'][:,:,:3])
            self.obs_buffer['robot']['right_wrist'].append(obs[f'{robot_key}'][f'{robot_key}:right_realsense_link:Camera:0']['rgb'][:,:,:3])
            self.obs_buffer['external'].append(self.env.external_sensors['external_camera'].get_obs()[0]['rgb'][:,:,:3])
        
        self._record_frame()
        
        for metric in self.metrics:
            metric.step_callback(self.env)
        return obs, reward, terminated, truncated, info
    
    
    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        obs = self.get_observation()
        robot_key = [key for key in obs.keys() if 'robot' in key][0]
        frame = obs[f'{robot_key}'][f'{robot_key}:zed_link:Camera:0']['rgb'][:,:,:3]
        return frame.cpu().numpy()
    
    def pick_up_radio_reward(self) -> float:
        """
        Compute the reward for picking up the radio.
        """
        table_obj = self.env.scene.object_registry("name", "coffee_table_koagbh_0")
        table_pose, table_quat = table_obj.get_position_orientation()
        table_height = table_pose[-1]
        grasp_obj = self.env.scene.object_registry("name", "radio_89")
        radio_pos, radio_quat = grasp_obj.get_position_orientation()
        if self.controller._get_obj_in_hand() == grasp_obj and radio_pos[2] > self.initial_radio_pose[2] + 0.005:
            reward = 1
        else:
            reward = 0
        
        return reward
    
    def pick_up_trash_reward(self) -> float:
        """
        Compute the reward for picking up the trash.
        """
        trash_obj = self.env.scene.object_registry("name", "can_of_soda_113")
        trash_pose, trash_quat = trash_obj.get_position_orientation()
        if self.controller._get_obj_in_hand() == trash_obj and trash_pose[2] > self.initial_trash_pose[2] + 0.005:
            reward = 1
        else:
            reward = 0
        
        return reward
    
    def compute_reward(self) -> float:
        """
        Compute the reward for the current state.
        Args:
            None.
        Returns:
            Reward for the current state.
        """
        if self.task_name == "turning_on_radio":
            return self.pick_up_radio_reward()
        if 'trash' in self.task_name:
            return self.pick_up_trash_reward()

        for metric in self.metrics:
            metric.end_callback(self.env)
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.gather_results())
            
        if self.task_completed():
            return 1
        else:
            return metrics['q_score']['final']

    def get_object_state(self, object_name: str) -> dict[str, Any]:
        """
        Get the state of an object in the environment.
        Args:
            object_name: Name of the object to get the state of. Must be in self.task_relevant_obj.
        Returns:
            State of the object.
        """
        object_instance = None
        for key, value in self.task_relevant_obj.items():
            if object_name in key:
                object_instance = value
                break
        if object_instance is None:
            raise ValueError(f"Object {object_name} is not in the task relevant objects.")
        object_state = object_instance.get_position_orientation()
        return object_state
    
    def get_observation(self) -> dict[str, Any]:
        """
        Get the observation of the environment.
        Returns:
            Observation of the environment.
            obs:
                - task:
                    - low_dim: A list of object state with the keys in self.object_state_key.
                - robot_wjqjxx:
                    - robot_wjqjxx:left_realsense_link:Camera:0:
                        - rgb: RGB image of the camera.
                        - depth: Depth image of the camera.
                    - robot_wjqjxx:right_realsense_link:Camera:0:
                        - rgb: RGB image of the camera.
                        - depth: Depth image of the camera.
                    - robot_wjqjxx:zed_link:Camera:0:
                        - rgb: RGB image of the camera.
                        - depth: Depth image of the camera.
                    - proprio: 68

        """
        obs, info = self.env.get_obs()
        return obs
    
    
    def update_torso(self, torso_joint_name: str, angle: float, max_steps=50):
        torso_joint = self.controller.robot.joints[torso_joint_name]
        torso_idx = self.torso_idx_map[torso_joint_name]
        
        # current_torso_joint_goal = angle
        
        # current_torso_joint_goal = torch.clamp(current_torso_joint_goal, self.torso_joint_limits[0], self.torso_joint_limits[1])
        target_joint_positions = self.robot.get_joint_positions().clone()
        target_joint_positions[torso_idx] = angle
        success = self._move_to_joint_positions(target_joint_positions, max_steps)
        return success

    
    def _navigate_to_pose(self, pose_2d):
        """
        Navigate to a pose in the environment.
        Args:
            pose: Pose to navigate to, xy and yaw in world frame.
        Returns:
            success: Whether the pose was navigated to successfully.
        """
        try:
            ctrl_gen = self.controller._navigate_to_pose(pose_2d) #, ignore_all_obstacles=True, skip_obstacle_update=True)
            # execute_controller(ctrl_gen, self.env)
            for a in ctrl_gen:
                _ = self.step(a)
        except TimeoutError:
            raise
        except Exception as e:
            print("Navigate to pose failed:", e)
            return False
        return True

    
    def _navigate_to_obj(self, obj_name: str):
        #self.task_relevant_obj["radio_receiver.n.01_1"].unwrapped.name
        """
        Navigate to an object in the environment.
        Args:
            obj_name: Name of the object to navigate to. Must be in self.task_relevant_obj.
        Returns:
            None.
        """
        grasp_obj = self.env.scene.object_registry("name", obj_name)
        # ctrl_gen = self.controller.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, grasp_obj)
        ctrl_gen = self.controller._navigate_to_obj(grasp_obj)
        execute_controller(ctrl_gen, self.env)
        return self.get_observation()
    
    
    def _open_close_gripper(self, arm=0, max_steps=250, open=True):
        """
        Open the gripper in the environment.
        Args:
            None.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        target_joint_positions = self.controller._get_joint_position_with_fingers_at_limit("upper" if open else "lower")
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for _ in range(max_steps):
            self.step(action)
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
            
    def _execute_motion_plan(self, q_traj):
        """
        Execute a motion plan in the environment.
        Args:
            q_traj: Joint trajectory to execute.
        Returns:
            None.
        """
        for a in self.controller._execute_motion_plan(q_traj):
            self.step(a)
            
    def _settle_robot(self):
        """
        Settle the robot in the environment.
        Args:
            None.
        Returns:
            None.
        """
        for a in self.controller._settle_robot():
            self.step(a)
            
    def _execute_grasp(self, grasp_pose: np.ndarray, obj_name: str, arm=0):
        """
        Execute the grasp in the environment.
        Args:
            grasp_pose: Grasp pose to execute.
        Returns:
            None.
        """
        # Step a few times to update
        for a in self.controller._settle_robot():
            self.step(a)

        print("Checking grasp")
        obj_in_hand = self.controller._get_obj_in_hand()
        if obj_in_hand is None:
            print("Grasp completed, but no object detected in hand after executing grasp")
            grasped = False
            return grasped

        print("Done with grasp")

        # if self.controller._get_obj_in_hand() != obj:
        #     print("An unexpected object was detected in hand after executing grasp. Consider releasing it")
        #     return reached, grasped
        
        grasped = True
        return grasped

    def check_object_in_hand(self, arm=0):
        """
        Execute the grasp in the environment without reaching.
        Args:
            obj_name: Name of the object to grasp.
        Returns:
            success: Whether the grasp was successful.
        """
        self._settle_robot()
        self.controller.overwrite_arm(arm)
        obj_in_hand = self.controller._get_obj_in_hand()
        if obj_in_hand is None:
            print("Grasp completed, but no object detected in hand after executing grasp")
            return False

        print("Done with grasp")
        return True

    
    def _sample_grasp_pose(self, obj_name: str, object_obb) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment.
        """
        obj = self.env.scene.object_registry("name", obj_name)
        grasp_poses = get_grasp_poses_for_object_sticky(obj, object_obb)
        grasp_pos, grasp_quat = random.choice(grasp_poses)

        # Identity quaternion for top-down grasping (x-forward, y-right, z-down)
        approach_dir = quat2mat(grasp_quat) @ torch.tensor([0.0, 0.0, -1.0])

        avg_finger_offset = torch.mean(
            torch.tensor([length for length in self.controller.robot.eef_to_fingertip_lengths[self.controller.arm].values()])
        )
        pregrasp_offset = avg_finger_offset + 0.25

        pregrasp_pos = grasp_pos - approach_dir * pregrasp_offset

        # The sampled grasp pose is robot-agnostic
        # We need to multiply by the quaternion of the robot's eef frame of top-down grasping (x-forward, y-right, z-down)
        grasp_quat = quat_multiply(grasp_quat, torch.tensor([1.0, 0.0, 0.0, 0.0]))

        pregrasp_pose = (pregrasp_pos, grasp_quat)
        grasp_pose = (grasp_pos, grasp_quat)

        return pregrasp_pose, grasp_pose

    def _grasp_obj(self, obj_name: str, arm=0):
        obj = self.env.scene.object_registry("name", obj_name)
        pregrasp_pose, grasp_pose = self.controller._sample_grasp_pose(obj)

        self._open_close_gripper(arm=arm, max_steps=250, open=True)
        print("Moving hand to grasp pose")
        for a in self.controller._move_hand(pregrasp_pose):
            _ = self.step(a)

        if self.robot.grasping_mode == "sticky":
            print("Sticky grasping: close gripper")
            # Close the gripper

            print("Sticky grasping: approach")
            # Only translate in the z-axis of the goal frame (assuming z-axis points out of the gripper)
            # This is the same as requesting the end-effector to move along the approach_dir direction.
            # By default, it's NOT the z-axis of the world frame unless `project_pose_to_goal_frame=False` is set in curobo.
            # For sticky grasping, we also need to ignore the object during motion planning because the fingers are already closed.
            # ctrl = self.controller._move_hand(grasp_pose, motion_constraint=[1, 1, 1, 1, 1, 0], stop_on_ag=True, ignore_objects=[obj])
            ctrl = self.controller._move_hand(grasp_pose, stop_on_ag=True, ignore_objects=[obj])
            for a in ctrl:  _  = self.step(a)
            
        elif self.robot.grasping_mode == "assisted":
            print("Assisted grasping: approach")
            # Same as above in terms of moving along the approach_dir direction, but we don't ignore the object.
            # For this approach motion, we expect the fingers to move towards and eventually "wrap" around the object without collisions.
            for a in self.controller._move_hand(grasp_pose, motion_constraint=[1, 1, 1, 1, 1, 0]): _ = self.step(a)
            # for a in self.controller._move_hand(grasp_pose, ignore_objects=[obj]): _ = self.step(a)

            # Now we close the fingers to grasp the object with AG.
            print("Assisted grasping: close gripper")
            self._open_close_gripper(arm=arm, max_steps=250, open=False)

        # Step a few times to update
        for a in self.controller._settle_robot():
            self.step(a)

        print("Checking grasp")
        obj_in_hand = self.controller._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Grasp completed, but no object detected in hand after executing grasp",
                {"target object": obj.name},
            )

        print("Done with grasp")

        if self.controller._get_obj_in_hand() != obj:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "An unexpected object was detected in hand after executing grasp. Consider releasing it",
                {"expected object": obj.name, "actual object": self._get_obj_in_hand().name},
            )
        
        if self.controller._get_obj_in_hand() == obj:
            return True
        else:
            return False
    
    def _release_object(self, arm=0):
        """
        Release the object in the environment in the default arm.
        Args:
            None.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        # ctrl_gen = self.controller.apply_ref(StarterSemanticActionPrimitiveSet.RELEASE, None)
        ctrl_gen = self.controller._execute_release()
        execute_controller(ctrl_gen, env)
        return self.get_observation()
    
    def _move_hand_direct_ik(self, target_pose, arm=0):
        """
        Move the hand to a target pose in the environment using direct inverse kinematics.
        Args:
            target_pose: Target pose to move the hand to.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        try:
            ctrl_gen = self.controller._move_hand_direct_ik(target_pose)
            for a in ctrl_gen:
                _ = self.step(a)
            success = True
        except TimeoutError:
            raise
        except Exception as e:
            print("Move hand failed:", e)
            success = False
        return success

    
    def _move_hand(self, target_pose, arm=0, ignore_objects=None, motion_constraint=None, attached_obj_scale=None, ignore_all_obstacles=False, ik_only=False, ik_world_collision_check=True, skip_obstacle_update=False):
        """
        Move the hand to a target pose in the environment.
        Args:
            target_pose: Target pose to move the hand to.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        # ctrl_gen = self.controller.apply_ref(StarterSemanticActionPrimitiveSet.MOVE_HAND, target_pose)
        try:
            ctrl_gen = self.controller._move_hand(target_pose, ignore_objects=ignore_objects, motion_constraint=motion_constraint, attached_obj_scale=attached_obj_scale, ignore_all_obstacles=ignore_all_obstacles, ik_only=ik_only, ik_world_collision_check=ik_world_collision_check, skip_obstacle_update=skip_obstacle_update)
            for a in ctrl_gen:
                _ = self.step(a)
            success = True
        except TimeoutError:
            raise
        except Exception as e:
            print("Move hand failed:", e)
            success = False
        return success
    
    def _move_hand_upward(self, arm=0):
        """
        Move the hand upward in the environment.
        Args:
            None.
        Returns:
            None.
        """
        cur_joint_positions = self.get_joint_positions()
        target_joint_positions = cur_joint_positions.clone()
        if arm == 0:
            target_joint_positions[10] -= 0.4
        elif arm == 1:
            target_joint_positions[11] -= 0.4
            
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for _ in range(20):
            _ = self.step(action)
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
    
    def _move_to_joint_positions(self, target_joint_positions, max_steps=20, settle_steps=10):
        """
        Move the robot to a target joint positions in the environment.
        Args:
            target_joint_positions: Target joint positions to move the robot to.
        Returns:
            success: Whether the joint positions were reached successfully.
        """
        # cur_joint_positions = self.get_joint_positions()
        # delta = 0.2
        # N = int(abs(target_joint_positions - cur_joint_positions).max()/delta)
        # print("N:", N)
        # N = min(N, 10)
        # delta = abs(target_joint_positions - cur_joint_positions).max()/N
        # for idx in range(N):
        #     cur_joint_positions = self.get_joint_positions()
        #     cur_target = cur_joint_positions + (target_joint_positions - cur_joint_positions).clamp(-delta, delta)
        #     action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, cur_target))
        #     for j in range(10):
        #         _ = self.step(action)
        #         current_joint_positions = self.get_joint_positions()
        #         if torch.allclose(current_joint_positions, cur_target, atol=0.005):
        #             break
        
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for idx in range(max_steps):
            _ = self.step(action)
            if idx % int(settle_steps) == 0:
                self._settle_robot()
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
            
        # current_joint_positions = self.get_joint_positions()
        if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
            return True
        else:
            return False
    
    def _lift_arm(self, arm=0):
        """
        Lift the arm in the environment.
        Args:
            None.
        Returns:
            None.
        """
        cur_joint_positions = self.get_joint_positions()
        target_joint_positions = cur_joint_positions.clone()
        if arm == 0:
            target_joint_positions[10] -= 0.4
        elif arm == 1:
            target_joint_positions[11] -= 0.4
            
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for _ in range(20):
            _ = self.step(action)
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
    
    def get_joint_positions(self) -> np.ndarray:
        """
        Get the joint positions of the robot.
        Returns:
            env.robot.joints.keys()
            dict_keys(['base_footprint_x_joint', 'base_footprint_y_joint', 'base_footprint_z_joint', 'base_footprint_rx_joint', 'base_footprint_ry_joint', 'base_footprint_rz_joint', 'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2'])
        """
        return self.env.robots[0].get_joint_positions()
    
    
    def get_robot_eef_pose(self, arm=0) -> dict[str, Any]:
        """
        Get the end-effector pose of the robot.
        Args:
            arm: Arm to get the end-effector pose of.
        Returns:
            End-effector pose of the robot.
        """
        arm = self.robot.arm_names[arm]
        return self.robot.get_eef_pose(arm=arm)
    
    def get_robot_base_pose(self) -> dict[str, Any]:
        """
        Get the base pose of the robot. A list of 3 and 4
        Returns:
            Base pose of the robot.
        """
        return self.robot.get_position_orientation()

    def get_toggle_button_geom(self, obj) -> dict[str, Any]:
        """
        Get the toggle button of an object in the environment.
        Args:
            obj: Object to get the toggle button of.
        Returns:
            Toggle button object.
        """
        return obj.states[ToggledOn].visual_marker
    
    def task_completed(self) -> bool:
        """Check if task is completed."""
        # action = np.zeros_like(self.robot.action_space.sample())
        # obs, reward, terminated, truncated, info = self.env.step(action)
        # success = info['done']['success']
        if self.task_name == "turning_on_radio":
            success = self.pick_up_radio_reward() == 1
        elif 'trash' in self.task_name:
            success = self.pick_up_trash_reward() == 1
        else:
            action = np.zeros_like(self.robot.action_space.sample())
            obs, reward, terminated, truncated, info = self.env.step(action)
            success = info['done']['success']
        return success
    
    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if len(self._frame_buffer) == 0:
            self._record_frame()
        keys = list(self._frame_buffer[0].keys())
        frames = {key: [frame[key].copy() for frame in self._frame_buffer] for key in keys}
        if clear:
            self._frame_buffer.clear()
        return frames
    
    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        obs = self.get_observation()
        obs_keys = list(obs.keys())
        robot_key = [key for key in obs_keys if 'robot' in key][0]
        frame ={
            'rgb': self.env.external_sensors['external_camera'].get_obs()[0]['rgb'][:,:,:3].detach().cpu().numpy().copy(),
            "ego": obs[f'{robot_key}'][f'{robot_key}:zed_link:Camera:0']['rgb'][:,:,:3].detach().cpu().numpy().copy(),
            "left_wrist": obs[f'{robot_key}'][f'{robot_key}:left_realsense_link:Camera:0']['rgb'][:,:,:3].detach().cpu().numpy().copy(),
            "right_wrist": obs[f'{robot_key}'][f'{robot_key}:right_realsense_link:Camera:0']['rgb'][:,:,:3].detach().cpu().numpy().copy()
        }
        self._frame_buffer.append(frame)  # Flip vertically
    
__all__ = ["R1ProBehaviourLowLevel"]


def oracle_solution():
    env = R1ProBehaviourLowLevel(save_video=True)
    all_rewards = []
    for task_id in range(10):
        reward = 0
        og.log.info("Resetting environment")
        env.reset()
        env.load_task_instance(task_id)
        
        radio_state = env.get_object_state("radio")
        radio_pose = radio_state[0]
        initial_radio_pose = radio_state[0].clone()
        
        robot_pos, robot_quat = env.robot.get_position_orientation()
        cur_robot_yaw = quat2yaw(robot_quat)
        
        grasp_obj = env.env.scene.object_registry("name", "radio_89")
        pregrasp_pose, grasp_pose = env.controller._sample_grasp_pose(grasp_obj)
        
        table_obj = env.env.scene.object_registry("name", "coffee_table_koagbh_0")
        table_pose, table_quat = table_obj.get_position_orientation()
        table_a, table_b = table_obj.aabb
        table_width = table_b[0] - table_a[0]
        table_length = table_b[1] - table_a[1]
        short_side_center = (table_pose[0], table_pose[1] - table_length/2)
        long_side_center = (table_pose[0] + table_width/2, table_pose[1])
        
        dist_to_short_side = np.linalg.norm(np.array(short_side_center) - np.array(radio_pose[:2]))
        dist_to_long_side = np.linalg.norm(np.array(long_side_center) - np.array(radio_pose[:2]))
        buffer_dist = 0.1
        if dist_to_short_side < dist_to_long_side:
            # robot_target = (short_side_center[0], short_side_center[1] - buffer_dist, np.pi/2)
            robot_target = (radio_pose[0], short_side_center[1] - buffer_dist, np.pi/2)
        else:
            # robot_target = (long_side_center[0] + buffer_dist, long_side_center[1], np.pi)
            robot_target = (long_side_center[0] + buffer_dist, radio_pose[1], np.pi)
            
        
        navigation_success = env._navigate_to_pose(robot_target)
        robot_pos, robot_quat = env.robot.get_position_orientation()
        radio_pos, radio_quat = grasp_obj.get_position_orientation()
        if np.linalg.norm(np.array(robot_pos[:2]) - np.array(radio_pos[:2])) < 0.3:
            reward += 1/3
        
        try:
            grasp_success = env._grasp_obj('radio_89')
        except TimeoutError:
            raise
        except Exception as e:
            grasp_success = False
            print("Grasp failed:", e)
        if grasp_success:
            reward += 1/3
            
        cur_joint_positions = env.get_joint_positions()
        target_joint_positions = cur_joint_positions.clone()
        target_joint_positions[10] -= 0.4
        action = env.robot.q_to_action(holonomic_base_command_in_world_frame(env.robot, target_joint_positions))
        for _ in range(20):
            _ = env.step(action)
            current_joint_positions = env.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
        
        table_height = table_pose[-1]
        radio_pos, radio_quat = grasp_obj.get_position_orientation()
        if env.controller._get_obj_in_hand() == grasp_obj and radio_pos[2] > initial_radio_pose[2] + 0.01:
            reward += 1/3
            
        print("Reward of task", task_id, ":", reward)
        all_rewards.append(reward)
            
        media.write_video(f"navigate_external_{task_id}.mp4",np.array(env.obs_buffer['external']), fps=30)
    
    print("All rewards:", all_rewards)
    print("Average reward:", np.mean(all_rewards))
    
def extract_instances(rgb, inst_mask):
    instance_ids = np.unique(inst_mask)

    instance_rgbs = {}

    for inst_id in instance_ids:
        mask = (inst_mask == inst_id)[:, :, None]    # shape (H,W,1)
        masked_rgb = rgb * mask                      # apply mask
        instance_rgbs[int(inst_id)] = masked_rgb

    return instance_rgbs

def object_instance_id(instance_registry, object_name):
    for inst_id, inst_name in instance_registry.items():
        if object_name in inst_name:
            return inst_id
    return None

def backproject_depth(mask, depth, K, T_world_cam):
    # VisionSensor depth maps follow the OpenGL camera frame (camera looks down its -Z, +X right, +Y up; image v axis is downward
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    vs, us = np.nonzero(mask)          # row v, col u
    # zs = depth[vs, us]                 # depth values
    # xs = (us - cx) * zs / fx
    # ys = (vs - cy) * zs / fy
    d = depth[vs, us]
    xs = (us - cx) * d / fx
    ys = -(vs - cy) * d / fy  # image y-down -> camera Y-up
    zs = -d                   # camera looks along -Z

    pts_cam = np.stack([xs, ys, zs, np.ones_like(zs)], axis=1)  # N×4
    pts_world = (T_world_cam @ pts_cam.T).T[:, :3]              # N×3
    return pts_world


def quat_xyzw_to_R(q):
    x, y, z, w = q
    # (Optional) normalize to be safe
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = q / n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w

    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - zw),     2*(xz + yw)],
        [2*(xy + zw),     1 - 2*(xx + zz), 2*(yz - xw)],
        [2*(xz - yw),     2*(yz + xw),     1 - 2*(xx + yy)],
    ])
    return R

def pose_to_T_world_cam(position, quat_xyzw):
    """
    position: (3,) [tx, ty, tz] in world frame
    quat_xyzw: (4,) [x, y, z, w] orientation of camera
    Returns: 4x4 T_world_cam that maps p_cam -> p_world
    """
    t = np.asarray(position).reshape(3)
    R = quat_xyzw_to_R(np.asarray(quat_xyzw))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


def closest_point_on_segment(p, a, b):
    # p, a, b are 2D
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab
    

def get_navigation_pose(P_table, P_radio):
    radio_center = np.median(P_radio, axis=0)   # (3,)

    P_table_xy = P_table[:, :2]   # drop z
    #remove nan values
    hull = ConvexHull(P_table_xy)
    table_polygon = P_table_xy[hull.vertices]   # (M,2) vertices in CCW order
    table_center_xy = table_polygon.mean(axis=0)
    
    radio_xy = radio_center[:2]

    min_dist = np.inf
    best_edge_idx = None
    best_edge_point = None

    for i in range(len(table_polygon)):
        a = table_polygon[i]
        b = table_polygon[(i+1) % len(table_polygon)]
        cp = closest_point_on_segment(radio_xy, a, b)
        d = np.linalg.norm(cp - radio_xy)
        if d < min_dist:
            min_dist = d
            best_edge_idx = i
            best_edge_point = cp

    p_edge_xy = best_edge_point
    a = table_polygon[best_edge_idx]
    b = table_polygon[(best_edge_idx + 1) % len(table_polygon)]
    
    edge = b - a                    # 2D
    edge = edge / (np.linalg.norm(edge) + 1e-8)

    # Two possible normals:
    n1 = np.array([-edge[1], edge[0]])   # rotate +90°
    n2 = -n1                             # rotate -90°

    # Choose the one pointing away from table center
    to_center = table_center_xy - p_edge_xy
    if np.dot(n1, to_center) < 0:
        outward = n1
    else:
        outward = n2
        
    buffer_distance = 0.3
    base_xy = p_edge_xy + outward * buffer_distance
    
    dx, dy = radio_xy - base_xy
    yaw = np.arctan2(dy, dx)   # robot faces radio

    goal = (base_xy[0], base_xy[1], yaw)
    return goal


if __name__ == "__main__":
    env = R1ProBehaviourLowLevel(save_video=True)
    all_rewards = []
    all_times = []
    
    for task_id in range(10):
        start_time = time.time()
        env.reset(options={'trial': task_id})
        # env.load_task_instance(task_id)

        reward = 0
        radio_state = env.get_object_state("radio")
        radio_pose = radio_state[0]
        initial_radio_pose = radio_state[0].clone()
        
        obs = env.get_observation()
        all_keys = list(obs.keys())
        robot_key = [key for key in all_keys if 'robot' in key][0]
        ego_depth = obs[robot_key][f'{robot_key}:zed_link:Camera:0']['depth_linear']
        ego_instance = obs[robot_key][f'{robot_key}:zed_link:Camera:0']['seg_instance']
        
    
        # plt.imshow(ego_depth)
        # plt.savefig('ego_depth.png')
        # plt.imshow(ego_instance)
        # plt.savefig('ego_instance.png')
        
        robot_pos, robot_quat = env.robot.get_position_orientation()
        cur_yaw = quat2yaw(robot_quat)
        delta_yaw = 0.5
        num_idx = 2*np.pi // delta_yaw
        print("num_idx:", num_idx)
        for idx in range(int(num_idx)):
            new_yaw = cur_yaw + delta_yaw
            new_pose = (robot_pos[0], robot_pos[1], new_yaw)
            env._navigate_to_pose(new_pose)
            obs = env.get_observation()
            
            ego_rgb = obs[robot_key][f'{robot_key}:zed_link:Camera:0']['rgb']
            ego_depth = obs[robot_key][f'{robot_key}:zed_link:Camera:0']['depth_linear']
            ego_instance = obs[robot_key][f'{robot_key}:zed_link:Camera:0']['seg_instance']
            plt.imshow(ego_depth)
            plt.savefig(f'ego_depth_{idx}.png')
            plt.imshow(ego_instance)
            plt.savefig(f'ego_instance_{idx}.png')
            
            obot_pos, robot_quat = env.robot.get_position_orientation()
            cur_yaw = quat2yaw(robot_quat)
            
            instance_rgbs = extract_instances(ego_rgb, ego_instance)
            radio_instance_id = object_instance_id(VisionSensor.INSTANCE_REGISTRY, "radio")
            table_instance_id = object_instance_id(VisionSensor.INSTANCE_REGISTRY, "coffee_table")
            if radio_instance_id is not None and radio_instance_id in instance_rgbs:
                radio_rgb = instance_rgbs[radio_instance_id]
                plt.imshow(radio_rgb)
                plt.savefig(f'radio_rgb_{idx}.png')
                if table_instance_id is not None and table_instance_id in instance_rgbs:
                    table_rgb = instance_rgbs[table_instance_id]
                    plt.imshow(table_rgb)
                    plt.savefig(f'table_rgb_{idx}.png')
                break
        
        
        # plt.clf()
        radio_mask = (ego_instance == radio_instance_id)
        table_mask = (ego_instance == table_instance_id)
        # radio_depth = ego_depth * radio_mask
        # table_depth = ego_depth * table_mask
        # plt.imshow(radio_mask)
        # plt.savefig('radio_mask.png')
        # plt.imshow(table_mask)
        # plt.savefig('table_mask.png')
        # plt.clf()
        # plt.imshow(radio_depth)
        # plt.savefig('radio_depth.png')
        # plt.imshow(table_depth)
        # plt.savefig('table_depth.png')
            
        ego_camera = env.robot.sensors[f'{robot_key}:zed_link:Camera:0']
        ego_camera_pos, ego_camera_quat = ego_camera.get_position_orientation()
        side_intrinsic_matrix = env.robot.sensors[f'{robot_key}:left_realsense_link:Camera:0'].intrinsic_matrix
        intrinsic_matrix = ego_camera.intrinsic_matrix
        
        T_world_cam = pose_to_T_world_cam(np.array(ego_camera_pos), np.array(ego_camera_quat))
        
        P_table  = backproject_depth(np.array(table_mask), np.array(ego_depth), np.array(intrinsic_matrix), T_world_cam)   # (Nt, 3)
        P_radio  = backproject_depth(np.array(radio_mask), np.array(ego_depth), np.array(intrinsic_matrix), T_world_cam)   # (Nr, 3)
        
        goal = get_navigation_pose(P_table, P_radio)
        env._navigate_to_pose(goal)
        
        robot_pos, robot_quat = env.robot.get_position_orientation()
        grasp_obj = env.env.scene.object_registry("name", "radio_89")
        radio_pos, radio_quat = grasp_obj.get_position_orientation()
        if np.linalg.norm(np.array(robot_pos[:2]) - np.array(radio_pos[:2])) < 0.3:
            reward += 1/3
        
        try:
            grasp_success = env._grasp_obj('radio_89')
        except TimeoutError:
            raise
        except Exception as e:
            grasp_success = False
            print("Grasp failed:", e)
        if grasp_success:
            reward += 1/3
            
        cur_joint_positions = env.get_joint_positions()
        target_joint_positions = cur_joint_positions.clone()
        target_joint_positions[10] -= 0.4
        action = env.robot.q_to_action(holonomic_base_command_in_world_frame(env.robot, target_joint_positions))
        for _ in range(20):
            _ = env.step(action)
            current_joint_positions = env.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
        
        table_obj = env.env.scene.object_registry("name", "coffee_table_koagbh_0")
        table_pose, table_quat = table_obj.get_position_orientation()
        table_height = table_pose[-1]
        radio_pos, radio_quat = grasp_obj.get_position_orientation()
        if env.controller._get_obj_in_hand() == grasp_obj and radio_pos[2] > initial_radio_pose[2] + 0.005:
            reward += 1/3
        
        print("Reward of task", task_id, ":", reward)
        all_rewards.append(reward)
        end_time = time.time()
        all_times.append(end_time - start_time)
            
        media.write_video(f"navigate_external_{task_id}.mp4",np.array(env.obs_buffer['external']), fps=30)

    print("All times:", all_times)
    print("Average time:", np.mean(all_times))