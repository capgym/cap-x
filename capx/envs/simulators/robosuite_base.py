"""Shared base class for single-arm Robosuite Franka environments.

Eliminates code duplication across robosuite_cubes.py, robosuite_cube_lift.py,
robosuite_cubes_restack.py, robosuite_spill_wipe.py, and robosuite_nut_assembly.py.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from robosuite.utils.camera_utils import get_real_depth_map

from capx.envs.base import BaseEnv
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

os.environ.setdefault("MUJOCO_GL", "egl")


class RobosuiteBaseEnv(BaseEnv):
    """Base class for single-arm Robosuite Franka environments.

    Subclasses must implement:
        - _create_robosuite_env() to create the specific robosuite environment
        - _post_env_init() for any task-specific post-initialization (optional)
        - reset() for task-specific reset logic
        - get_observation() for task-specific observation processing
        - compute_reward() for task-specific reward
        - task_completed() for task-specific completion check
    """

    # Subclasses can override these defaults
    _SUBSAMPLE_RATE: int = 5
    _ACTION_SLICE: int = -1  # action[:-1] for most envs, action[:-2] for spill_wipe

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        super().__init__()
        self.controller_cfg = controller_cfg
        self.max_steps = max_steps
        self.save_camera_name = "robot0_robotview"
        self.render_camera_names = [self.save_camera_name]
        self.segmentation_level = "instance"

        self._render_width = 512
        self._render_height = 512

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._rng = np.random.default_rng(seed)

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False
        self._wrist_camera_name = "robot0_eye_in_hand"
        self._subsample_rate = self._SUBSAMPLE_RATE

        # Control state
        self._current_joints = np.zeros(7, dtype=np.float64)
        self._gripper_fraction = 1.0  # 1.0 = open, 0.0 = closed

    def _init_robot_links(self) -> None:
        """Initialize robot link indices and base transforms. Call after robosuite_env is created."""
        self.gripper_metric_length = 0.04
        self.base_link_idx = self.robosuite_env.sim.model.body_name2id("fixed_mount0_base")
        self.gripper_link_idx = self.robosuite_env.sim.model.body_name2id("gripper0_right_eef")

        self.base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

    def _init_viser_debug(self, viser_debug: bool) -> None:
        """Initialize viser debug state. Call at end of subclass __init__."""
        if viser_debug:
            self.viser_server = viser.ViserServer()
            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584
            self.cube_points = None
            self.cube_color = None
            self.cube_center = None
            self.cube_rot = None

    # ----------------------- Shared Control Interface -----------------------

    def _set_gripper(self, fraction: float) -> None:
        """Set gripper opening fraction.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _build_action(self) -> np.ndarray:
        """Build a robosuite action array from current joints and gripper state."""
        action = np.concatenate(
            [self._current_joints, [self._gripper_fraction, self._gripper_fraction]]
        )
        # Map gripper: 1.0 (open) -> -1.0, 0.0 (closed) -> 1.0
        action[-2:] = 1.0 - action[-2:] * 2.0
        return action

    def _do_robosuite_step(self, action: np.ndarray) -> None:
        """Step robosuite with the given action, handling render skipping."""
        sliced = action[:self._ACTION_SLICE] if self._ACTION_SLICE != 0 else action
        need_render = (self._record_frames and self._sim_step_count % self._subsample_rate == 0) or hasattr(self, "viser_server")
        if need_render:
            self.robosuite_env.step(sliced)
        else:
            self.robosuite_env.step(sliced, skip_render_images=True)

    def _step_once(self) -> None:
        """Execute one simulation step with current control state."""
        action = self._build_action()
        self._do_robosuite_step(action)

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )
        if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()
        self._sim_step_count += 1

    def move_to_joints_non_blocking(self, joints: np.ndarray) -> None:
        """Move to target joint positions using Robosuite's controller (non-blocking)."""
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
        action[-2:] = 1.0 - action[-2:] * 2.0

        self._do_robosuite_step(action)

        if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

        self._sim_step_count += 1

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)

            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
            action[-2:] = 1.0 - action[-2:] * 2.0

            self._do_robosuite_step(action)

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._step_count += 1
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    # ----------------------- Camera / Observation Helpers -----------------------

    def _process_camera_observations(
        self, robosuite_obs: dict[str, Any], *, base_wxyz_xyz: np.ndarray | None = None
    ) -> None:
        """Process camera observations in-place on robosuite_obs.

        Adds pose, pose_mat, intrinsics, and images for each camera in render_camera_names.
        """
        if base_wxyz_xyz is None:
            base_wxyz_xyz = self.base_link_wxyz_xyz

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
                    vtf.SE3(wxyz_xyz=base_wxyz_xyz).inverse()
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
                depth_metric = get_real_depth_map(
                    self.robosuite_env.sim, robosuite_obs[camera_name + "_depth"][::-1]
                )
                robosuite_obs[camera_name]["images"]["depth"] = depth_metric
            if camera_name + "_segmentation_" + self.segmentation_level in robosuite_obs:
                robosuite_obs[camera_name]["images"]["segmentation"] = robosuite_obs[
                    camera_name + "_segmentation_" + self.segmentation_level
                ][::-1]

    def _compute_gripper_obs(self, robosuite_obs: dict[str, Any]) -> None:
        """Compute gripper pose and add robot_joint_pos / robot_cartesian_pos to obs."""
        gripper_robot_base = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=self.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        robosuite_obs["robot_joint_pos"] = np.concatenate(
            [
                robosuite_obs["robot0_joint_pos"],
                [robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )
        robosuite_obs["robot_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base.translation(),
                gripper_robot_base.rotation().wxyz,
                [robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )

    # ------------------------- Video Capture -------------------------

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

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

        if self._record_wrist_camera:
            wrist_frame = self.robosuite_env.sim.render(
                camera_name=self._wrist_camera_name,
                width=self._render_width,
                height=self._render_height,
                depth=False,
            )
            self._wrist_frame_buffer.append(wrist_frame[::-1])

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

    def render_wrist(self) -> np.ndarray:
        """Render the current frame from the wrist (eye-in-hand) camera."""
        frame = self.robosuite_env.sim.render(
            camera_name=self._wrist_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    # ------------------------- Viser Debugging -------------------------

    def _update_viser_server(self) -> None:
        """Default viser update. Subclasses with different viser logic should override."""
        obs = self.get_observation()
        if self.viser_server is not None:
            self._viser_init_check()

            obs_cartesian = obs["robot_cartesian_pos"][:-1]

            self.mjcf_ee_frame_handle.position = obs_cartesian[:3]
            self.mjcf_ee_frame_handle.wxyz = obs_cartesian[3:]

            rbg_imgs = obs_get_rgb(obs)
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                if "pose" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

                # Point cloud from depth (use whichever camera has depth)
                if "depth" in obs[image_key].get("images", {}):
                    points, colors = depth_color_to_pointcloud(
                        obs[image_key]["images"]["depth"][:, :, 0],
                        rbg_imgs[image_key],
                        obs[image_key]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        f"{image_key}/point_cloud",
                        points,
                        colors,
                        point_size=0.001,
                        point_shape="square",
                    )

            if self.cube_center is not None and self.cube_rot is not None:
                self.viser_server.scene.add_frame(
                    "robot0_robotview/cube_frame",
                    position=self.cube_center,
                    wxyz=vtf.SO3.from_matrix(self.cube_rot).wxyz,
                    axes_length=0.05,
                    axes_radius=0.005,
                )

            if self.cube_points is not None and self.cube_color is not None:
                self.viser_server.scene.add_point_cloud(
                    "robot0_robotview/cube_point_cloud",
                    self.cube_points,
                    self.cube_color,
                    point_size=0.001,
                    point_shape="square",
                )

            if hasattr(self, "grasp_sample") and self.grasp_sample is not None:
                grasp = self.grasp_sample[np.argmax(self.grasp_scores)]

                grasp_tf = vtf.SE3.from_matrix(grasp) @ vtf.SE3.from_translation(
                    np.array([0, 0, 0.1])
                )
                self.grasp_mesh_handle = self.viser_server.scene.add_frame(
                    "robot0_robotview/grasp",
                    position=grasp_tf.wxyz_xyz[-3:],
                    wxyz=grasp_tf.wxyz_xyz[:4],
                    axes_length=0.05,
                    axes_radius=0.0015,
                )

    def update_viser_image(self, frame: np.ndarray) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        if self.viser_img_handle is not None:
            self.viser_img_handle.image = frame

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/panda_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="robot0_robotview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=self._render_width / self._render_height,
                scale=0.05,
            )


__all__ = ["RobosuiteBaseEnv"]
