"""Low-level Robosuite Two-Arm Lift environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's TwoArmLift environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import robosuite as suite
import viser

# Temporary viser debugging imports
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.utils.camera_utils import get_real_depth_map

from capx.envs.base import BaseEnv
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

os.environ.setdefault("MUJOCO_GL", "egl")


class RobosuiteTwoArmLiftEnv(BaseEnv):
    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl2.json",
        max_steps: int = 5000,
        seed: int | None = None,
        viser_debug: bool = False,  # TODO: move the viser visualization manager into a separate class, low level env agnostic
        privileged: bool = True,
        enable_render: bool = False,
    ) -> None:
        super().__init__()
        self.controller_cfg = controller_cfg
        self.max_steps = max_steps
        self.save_camera_name = "agentview"  # Scene-level camera to show both arms
        self.render_camera_names = ["agentview"]  # Scene-level camera for observations
        self.segmentation_level = "instance"

        self._render_width = 512
        self._render_height = 512

        # Initialize Robosuite environment
        # TwoArmLift requires 2 robots
        # Load controller config for both robots (same config for both)
        controller_config = load_composite_controller_config(controller=self.controller_cfg)

        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = suite.make(
                    "TwoArmLift",
                    robots=["Panda", "Panda"],
                    env_configuration="opposed",
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=[controller_config, controller_config],
                    horizon=max_steps,
                    use_object_obs=True,
                    reward_shaping=True,
                )
            else:
                self.robosuite_env = suite.make(
                    "TwoArmLift",
                    robots=["Panda", "Panda"],
                    env_configuration="opposed",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=[controller_config, controller_config],
                    horizon=max_steps,
                    use_object_obs=True,
                    use_camera_obs=True,
                    camera_depths=True,
                    reward_shaping=True,
                )
        else:
            self.robosuite_env = suite.make(
                "TwoArmLift",
                robots=["Panda", "Panda"],
                env_configuration="opposed",
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=[controller_config, controller_config],
                horizon=max_steps,
                use_object_obs=True,
                use_camera_obs=True,
                camera_depths=True,
                reward_shaping=True,
            )

        # Get camera ID and modify its position and orientation
        agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
        self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [1.5, 0.0, 2.5]
        self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.653, 0.271, 0.271, 0.653]

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._rng = np.random.default_rng(seed)

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._subsample_rate = 4

        # Robot link indices for transforms (robot0 and robot1)
        # Base links are fixed, so we cache their transforms
        self.gripper_metric_length = 0.04
        self.base_link_idx_0 = self.robosuite_env.sim.model.body_name2id("fixed_mount0_base")
        self.gripper_link_idx_0 = self.robosuite_env.sim.model.body_name2id("gripper0_right_eef")
        self.base_link_idx_1 = self.robosuite_env.sim.model.body_name2id("fixed_mount1_base")
        self.gripper_link_idx_1 = self.robosuite_env.sim.model.body_name2id("gripper1_right_eef")

        # Cache base transforms (these are constant, base doesn't move)
        self.base_link_wxyz_xyz_0 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_0],
            ]
        )
        self.base_link_wxyz_xyz_1 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_1],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_1],
            ]
        )

        # Gripper state (read from robosuite when needed, not stored)
        self._gripper_fraction_0 = 1.0  # Target gripper state for robot0
        self._gripper_fraction_1 = 1.0  # Target gripper state for robot1

        # Temporary viser debugging
        if viser_debug:
            self.viser_server = viser.ViserServer()

            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle_0 = None
            self.mjcf_ee_frame_handle_1 = None
            self.handle0_frame_handle = None
            self.handle1_frame_handle = None
            self.det_handle0_frame_handle = None
            self.det_handle1_frame_handle = None
            self.det_handle0_bbox_center_handle = None
            self.det_handle1_bbox_center_handle = None
            self.detected_poses = {}
            self.detected_bbox_centers = {}

            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()

        # Re-apply camera adjustment after reset (in case model was reloaded)
        agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
        self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [1.5, 0.0, 2.5]
        self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.653, 0.271, 0.271, 0.653]

        self._step_count = 0
        self._sim_step_count = 0

        obs = self.get_observation()
        info = {
            "task_prompt": "Two robotic arms (Panda) should coordinate to lift the pot. Arm 0 should grasp one handle, and Arm 1 should grasp the other handle."
        }
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._step_count += 1
        # This is a fallback; normally FrankaControlApi methods are used
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    # ----------------------- FrankaControlApi Interface -----------------------

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move robot0 to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            robot1_joints = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

            # Check convergence
            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            # Build Robosuite action for both robots
            # Each robot action = [7 joints, 1 gripper] = 8 dims
            robot0_action = np.concatenate([target, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([robot1_joints, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            self.robosuite_env.step(action)
            self._sim_step_count += 1

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

    def move_to_joints_blocking_arm1(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move robot1 to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)
            robot0_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)

            # Check convergence
            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            # Build Robosuite action for both robots
            robot0_action = np.concatenate([robot0_joints, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([target, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            self.robosuite_env.step(action)
            self._sim_step_count += 1

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

    def move_to_joints_blocking_both(
        self,
        joints0: np.ndarray,
        joints1: np.ndarray,
        *,
        tolerance: float = 0.02,
        max_steps: int = 100,
    ) -> None:
        """Move both robots to target joint positions simultaneously using Robosuite's controller.

        Args:
            joints0: (7,) target joint positions for robot0 in radians
            joints1: (7,) target joint positions for robot1 in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target0 = np.asarray(joints0, dtype=np.float64).reshape(7)
        target1 = np.asarray(joints1, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current0 = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            current1 = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

            # Check convergence for both
            error0 = np.linalg.norm(current0 - target0)
            error1 = np.linalg.norm(current1 - target1)

            if error0 < tolerance and error1 < tolerance:
                break

            # Build Robosuite action for both robots
            robot0_action = np.concatenate([target0, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([target1, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            self.robosuite_env.step(action)
            self._sim_step_count += 1

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

    def _set_gripper(self, fraction: float) -> None:
        """Set target gripper opening fraction for robot0.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction_0 = float(np.clip(fraction, 0.0, 1.0))

    def _set_gripper_arm1(self, fraction: float) -> None:
        """Set target gripper opening fraction for robot1.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction_1 = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """Execute one simulation step maintaining current joint positions and gripper states."""
        # Get current joint positions from robosuite
        robosuite_obs = self.robosuite_env._get_observations()
        robot0_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        robot1_joints = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

        # Build action for both robots (maintain current joints, apply gripper targets)
        robot0_action = np.concatenate([robot0_joints, [1.0 - self._gripper_fraction_0 * 2.0]])
        robot1_action = np.concatenate([robot1_joints, [1.0 - self._gripper_fraction_1 * 2.0]])
        action = np.concatenate([robot0_action, robot1_action])

        self.robosuite_env.step(action)
        self._sim_step_count += 1

        if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    def _pot_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get pot poses in robot base frame.

        Returns:
            Dict with "pot_pos", "handle0_pos", "handle1_pos" keys
        """
        # Get transforms
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_0],
            ]
        )

        # Get pot position and quaternion
        pot_pos_world = robosuite_obs.get("pot_pos", np.zeros(3))
        pot_quat_xyzw = robosuite_obs.get("pot_quat", np.array([0, 0, 0, 1]))
        # Convert quaternion from xyzw to wxyz
        pot_quat_wxyz = np.array(
            [pot_quat_xyzw[3], pot_quat_xyzw[0], pot_quat_xyzw[1], pot_quat_xyzw[2]]
        )

        # Get handle positions
        # Robosuite usually provides handle_xpos but for TwoArmLift, need to check keys.
        # Assuming handle0_xpos and handle1_xpos exist for TwoArmLift
        # If not present, we might need to infer them from pot pose.
        handle0_pos_world = robosuite_obs.get(
            "handle0_xpos", pot_pos_world + np.array([0.1, 0, 0])
        )  # Fallback/Guess
        handle1_pos_world = robosuite_obs.get(
            "handle1_xpos", pot_pos_world + np.array([-0.1, 0, 0])
        )  # Fallback/Guess

        # Transform pot to robot base frame
        pot_world = vtf.SE3(wxyz_xyz=np.concatenate([pot_quat_wxyz, pot_pos_world]))
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        pot_robot_base = base_transform @ pot_world

        # Transform handle positions (use same rotation as pot for now, or identity if points)
        handle0_world = vtf.SE3.from_translation(handle0_pos_world)
        handle0_robot_base = base_transform @ handle0_world

        handle1_world = vtf.SE3.from_translation(handle1_pos_world)
        handle1_robot_base = base_transform @ handle1_world

        return {
            "pot": [
                float(x)
                for x in np.concatenate(
                    [pot_robot_base.translation(), pot_robot_base.rotation().wxyz]
                )
            ],
            "handle0": [
                float(x)
                for x in np.concatenate(
                    [
                        handle0_robot_base.translation(),
                        pot_robot_base.rotation().wxyz,
                    ]  # Using pot rotation
                )
            ],
            "handle1": [
                float(x)
                for x in np.concatenate(
                    [
                        handle1_robot_base.translation(),
                        pot_robot_base.rotation().wxyz,
                    ]  # Using pot rotation
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute sparse lift reward."""
        # Use the robosuite environment's built-in reward function
        reward = float(self.robosuite_env.reward())
        return reward

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._pot_pose_dict(robosuite_obs)

        # Store pot poses explicitly
        pot_pose_array = np.stack(
            [
                np.asarray(pose_dict["pot"], dtype=np.float32),
                np.asarray(pose_dict["handle0"], dtype=np.float32),
                np.asarray(pose_dict["handle1"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["pot_poses"] = {
            "pot": pot_pose_array[0],
            "handle0": pot_pose_array[1],
            "handle1": pot_pose_array[2],
        }

        for camera_name in self.render_camera_names:
            if camera_name not in robosuite_obs:
                robosuite_obs[camera_name] = {}

            cam_world_wxyz_xyz = np.concatenate(
                [
                    vtf.SO3.from_matrix(
                        self.robosuite_env.sim.data.get_camera_xmat(camera_name)
                    ).wxyz,
                    self.robosuite_env.sim.data.get_camera_xpos(camera_name),
                ]
            )
            cam_robot_tf = (
                (
                    vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse()
                    @ vtf.SE3(wxyz_xyz=cam_world_wxyz_xyz)
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0),
                    translation=np.array([0, 0, 0]),
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi),
                    translation=np.array([0, 0, 0]),
                )
            )

            robosuite_obs[camera_name]["pose"] = np.concatenate(
                [
                    cam_robot_tf.translation(),
                    cam_robot_tf.rotation().wxyz,
                ]
            )
            robosuite_obs[camera_name]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.robosuite_env.sim.model.camera_name2id(camera_name)
            fovy = self.robosuite_env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)

            K = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            robosuite_obs[camera_name]["intrinsics"] = K

            robosuite_obs[camera_name]["images"] = {}
            if camera_name + "_image" in robosuite_obs:
                robosuite_obs[camera_name]["images"]["rgb"] = robosuite_obs[camera_name + "_image"][
                    ::-1
                ]
            if camera_name + "_depth" in robosuite_obs:
                # converts openGL z buffer to metric
                depth_metric = get_real_depth_map(
                    self.robosuite_env.sim, robosuite_obs[camera_name + "_depth"][::-1]
                )

                robosuite_obs[camera_name]["images"]["depth"] = depth_metric
            if camera_name + "_segmentation_" + self.segmentation_level in robosuite_obs:
                robosuite_obs[camera_name]["images"]["segmentation"] = robosuite_obs[
                    camera_name + "_segmentation_" + self.segmentation_level
                ][::-1]

        # Compute gripper pose for robot0
        gripper_link_wxyz_xyz_0 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx_0],
            ]
        )
        gripper_robot_base_0 = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse()
            @ vtf.SE3(wxyz_xyz=gripper_link_wxyz_xyz_0)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        # Compute gripper pose for robot1 in robot0 frame
        gripper_link_wxyz_xyz_1 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx_1],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx_1],
            ]
        )

        # First compute in robot1's base frame
        gripper_robot_base_1_local = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_1).inverse()
            @ vtf.SE3(wxyz_xyz=gripper_link_wxyz_xyz_1)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        # Transform from robot1's base frame to world frame
        gripper_world_1 = vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_1) @ gripper_robot_base_1_local

        # Transform from world frame to robot0's base frame
        gripper_robot_base_1 = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse() @ gripper_world_1
        )

        robosuite_obs["robot0_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base_0.translation(),
                gripper_robot_base_0.rotation().wxyz,
                [robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )

        # Now robot1_cartesian_pos is in robot0's base frame
        robosuite_obs["robot1_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base_1.translation(),
                gripper_robot_base_1.rotation().wxyz,
                [robosuite_obs["robot1_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )
        if len(self.render_camera_names) == 1:
            robosuite_obs["robot0_robotview"] = robosuite_obs[self.render_camera_names[0]]
        else:
            raise ValueError("Unhandle case, with multiple camera names")

        return robosuite_obs

    # ------------------------- Video Capture -------------------------

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def _record_frame(self) -> None:
        if not self._record_frames:
            return

        frame = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        self._frame_buffer.append(frame[::-1])  # Flip vertically

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        frame = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    # Temporary viser debugging
    def _update_viser_server(
        self,
    ) -> None:
        try:
            obs = self.get_observation()
            if self.viser_server is not None:
                self._viser_init_check()

                obs_cartesian_0 = obs["robot0_cartesian_pos"][:-1]
                obs_cartesian_1 = obs["robot1_cartesian_pos"][:-1]

                self.mjcf_ee_frame_handle_0.position = obs_cartesian_0[:3]
                self.mjcf_ee_frame_handle_0.wxyz = obs_cartesian_0[3:]

                self.mjcf_ee_frame_handle_1.position = obs_cartesian_1[:3]
                self.mjcf_ee_frame_handle_1.wxyz = obs_cartesian_1[3:]

                # Add handle visualization
                if "pot_poses" in obs:
                    handle0_pose = obs["pot_poses"]["handle0"]
                    if self.handle0_frame_handle is not None:
                        self.handle0_frame_handle.position = handle0_pose[:3]
                        self.handle0_frame_handle.wxyz = handle0_pose[3:]

                    handle1_pose = obs["pot_poses"]["handle1"]
                    if self.handle1_frame_handle is not None:
                        self.handle1_frame_handle.position = handle1_pose[:3]
                        self.handle1_frame_handle.wxyz = handle1_pose[3:]

                if hasattr(self, "detected_poses") and "handle0" in self.detected_poses:
                    pos, wxyz = self.detected_poses["handle0"]
                    if self.det_handle0_frame_handle is not None:
                        self.det_handle0_frame_handle.position = pos
                        self.det_handle0_frame_handle.wxyz = wxyz

                if hasattr(self, "detected_poses") and "handle1" in self.detected_poses:
                    pos, wxyz = self.detected_poses["handle1"]
                    if self.det_handle1_frame_handle is not None:
                        self.det_handle1_frame_handle.position = pos
                        self.det_handle1_frame_handle.wxyz = wxyz

                # Update bbox center visualizations
                if hasattr(self, "detected_bbox_centers"):
                    if "handle0" in self.detected_bbox_centers:
                        bbox_pos = self.detected_bbox_centers["handle0"]
                        if self.det_handle0_bbox_center_handle is not None:
                            self.det_handle0_bbox_center_handle.position = bbox_pos
                            # Use identity quaternion for bbox center (no rotation)
                            self.det_handle0_bbox_center_handle.wxyz = np.array(
                                [1.0, 0.0, 0.0, 0.0]
                            )

                    if "handle1" in self.detected_bbox_centers:
                        bbox_pos = self.detected_bbox_centers["handle1"]
                        if self.det_handle1_bbox_center_handle is not None:
                            self.det_handle1_bbox_center_handle.position = bbox_pos
                            # Use identity quaternion for bbox center (no rotation)
                            self.det_handle1_bbox_center_handle.wxyz = np.array(
                                [1.0, 0.0, 0.0, 0.0]
                            )

                rbg_imgs = obs_get_rgb(obs)
                for image_key in rbg_imgs:
                    if self.viser_img_handle is not None:
                        self.viser_img_handle.image = rbg_imgs[image_key]

                    if "pose" in obs[image_key]:
                        self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                        self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                        self.image_frustum_handle.image = rbg_imgs[image_key]
                    else:
                        self.image_frustum_handle.visible = False

                # Temporary hardcode to visualise some stuff for debugging
                if "depth" in obs["agentview"]["images"] and "agentview" in rbg_imgs:
                    depth_img = obs["agentview"]["images"]["depth"]
                    if depth_img.ndim == 3:
                        depth_img = depth_img[:, :, 0]

                    points, colors = depth_color_to_pointcloud(
                        depth_img,
                        rbg_imgs["agentview"],
                        obs["agentview"]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        "agentview/point_cloud",
                        points,
                        colors,
                        point_size=0.001,
                        point_shape="square",
                    )
        except Exception:
            # Silently catch viser update errors to prevent blocking execution
            # The visualization is non-critical, so we don't want it to stop the robot
            pass

    def update_viser_image(self, frame: np.ndarray) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        if self.viser_img_handle is not None:
            self.viser_img_handle.image = frame

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle_0 is None:
            self.mjcf_ee_frame_handle_0 = self.viser_server.scene.add_frame(
                "/panda0_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.mjcf_ee_frame_handle_1 is None:
            self.mjcf_ee_frame_handle_1 = self.viser_server.scene.add_frame(
                "/panda1_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.handle0_frame_handle is None:
            self.handle0_frame_handle = self.viser_server.scene.add_frame(
                "/ground_truth/handle0", axes_length=0.15, axes_radius=0.005
            )
        if self.handle1_frame_handle is None:
            self.handle1_frame_handle = self.viser_server.scene.add_frame(
                "/ground_truth/handle1", axes_length=0.15, axes_radius=0.005
            )

        if self.det_handle0_frame_handle is None:
            self.det_handle0_frame_handle = self.viser_server.scene.add_frame(
                "/detected/handle0", axes_length=0.15, axes_radius=0.005
            )
        if self.det_handle1_frame_handle is None:
            self.det_handle1_frame_handle = self.viser_server.scene.add_frame(
                "/detected/handle1", axes_length=0.15, axes_radius=0.005
            )

        if self.det_handle0_bbox_center_handle is None:
            self.det_handle0_bbox_center_handle = self.viser_server.scene.add_frame(
                "/detected/handle0_bbox_center", axes_length=0.1, axes_radius=0.003
            )
        if self.det_handle1_bbox_center_handle is None:
            self.det_handle1_bbox_center_handle = self.viser_server.scene.add_frame(
                "/detected/handle1_bbox_center", axes_length=0.1, axes_radius=0.003
            )

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="agentview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=self._render_width / self._render_height,
                scale=0.05,
            )


__all__ = ["RobosuiteTwoArmLiftEnv"]
