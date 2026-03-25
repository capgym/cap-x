"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's NutAssemblySquare environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import robosuite as suite
import viser
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.utils.camera_utils import get_real_depth_map
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

os.environ.setdefault("MUJOCO_GL", "egl")


class FrankaRobosuiteNutAssembly(RobosuiteBaseEnv):
    """Robosuite Franka NutAssembly environment with FrankaPickPlaceLowLevel-compatible interface."""

    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1000,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = True,
        enable_render: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )

        self.save_camera_name = "birdview"
        self.render_camera_names = [self.save_camera_name]

        # Initialize Robosuite environment
        self.privileged = privileged
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = suite.environments.manipulation.nut_assembly.NutAssemblySquare(
                    robots=["Panda"],
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    reward_shaping=True,
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                )
            else:
                self.robosuite_env = suite.environments.manipulation.nut_assembly.NutAssemblySquare(
                    robots=["Panda"],
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    camera_depths=True,
                    renderer="mujoco",
                    reward_shaping=True,
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                )
        else:
            self.robosuite_env = suite.environments.manipulation.nut_assembly.NutAssemblySquare(
                robots=["Panda"],
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                # camera_segmentations=self.segmentation_level,
                camera_depths=True,
                renderer="mujoco",
                reward_shaping=True,
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=load_composite_controller_config(controller=self.controller_cfg),
                horizon=max_steps,
            )

        # nut type
        self.nut_type = "square"
        self.nut_id = self.robosuite_env.nut_to_id[self.nut_type]
        self.nuts = self.robosuite_env.nuts

        self._init_robot_links()

        # Precompute fast joint qpos addresses for Panda (avoid heavy _get_observations in tight loops)
        joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self._panda_joint_qpos_addrs: list[int] = []
        for jn in joint_names:
            addr = self.robosuite_env.sim.model.get_joint_qpos_addr(jn)
            if isinstance(addr, tuple):
                addr = addr[0]
            self._panda_joint_qpos_addrs.append(int(addr))
        self.home_joint_position: np.ndarray | None = None

        # Temporary viser debugging
        self.viser_debug = viser_debug
        if viser_debug:
            self.viser_server = viser.ViserServer()

            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.mjcf_gripper_frame_handle = None
            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584
            self.urdf = load_robot_description("panda_description")
            self.urdf_vis = ViserUrdf(self.viser_server, urdf_or_path=self.urdf, load_meshes=True)
            self._viser_init_check()

            self.cube_points = None
            self.cube_color = None
            self.cube_center = None
            self.cube_rot = None
            self.grasp_frame_position = None
            self.grasp_frame_orientation = None
        else:
            self.viser_server = None

        self.reset()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        first_obs = self.robosuite_env.reset()
        self.home_joint_position = np.array(first_obs["robot0_joint_pos"], dtype=np.float64)
        self.robosuite_env.sim.data.qpos[:7] = np.array(
            [0, -1.585, 0, -2.645, 0, 1, 0.785]
        )  # should be out of the view from top camera
        self.robosuite_env.sim.forward()
        self._current_joints = self.robosuite_env.sim.data.qpos[:7].copy()

        self._step_count = 0
        self._sim_step_count = 0

        self._gripper_fraction = 1.0

        # NOTE: This seems very important: post reset there seems to be a bit of noise on the joint velocity
        # also the blocks are reset to be in air, so if we sample from the initial state, the blocks are not at the correct height
        for _ in range(100):
            self._step_once()

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        info = {
            "task_prompt": "Insert the square nut onto the small square peg. Quaternions are WXYZ."
        }

        return obs, info

    # Override _step_once to include the double _sim_step_count increment from original
    def _step_once(self) -> None:
        """Execute one simulation step with current control state."""
        action = self._build_action()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self.robosuite_env.step(action[:-1])
        else:
            self.robosuite_env.step(action[:-1], skip_render_images=True)

        self._sim_step_count += 1

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )
        if self.viser_debug and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

        self._sim_step_count += 1

    # Override move_to_joints_blocking to use fast qpos path
    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move to target joint positions using Robosuite's controller."""
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            # Fast path via direct qpos indices
            current = np.array(
                self.robosuite_env.sim.data.qpos[self._panda_joint_qpos_addrs], dtype=np.float64
            )

            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
            action[-2:] = 1.0 - action[-2:] * 2.0

            need_render = (self._record_frames and self._sim_step_count % self._subsample_rate == 0) or self.viser_debug
            if need_render:
                self.robosuite_env.step(action[:-1])
            else:
                self.robosuite_env.step(action[:-1], skip_render_images=True)

            self._sim_step_count += 1

            if self.viser_debug and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

    def _get_nut_pose(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get nut pose in robot base frame."""
        invert_grasp_pose = vtf.SE3.from_matrix(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        _sq_xyzw = np.asarray(robosuite_obs["SquareNut_quat"], dtype=np.float64)
        _sq_wxyz = np.array([_sq_xyzw[3], _sq_xyzw[0], _sq_xyzw[1], _sq_xyzw[2]], dtype=np.float64)
        square_nut_world = vtf.SE3(
            wxyz_xyz=np.concatenate([_sq_wxyz, robosuite_obs["SquareNut_pos"]])
        )
        nut_handle_to_center_offset = np.array([0.054, 0, 0])
        square_nut_handle_world = square_nut_world @ vtf.SE3.from_translation(
            nut_handle_to_center_offset
        )

        peg_world = vtf.SE3(
            wxyz_xyz=np.concatenate(
                [
                    self.robosuite_env.sim.data.xquat[self.robosuite_env.peg1_body_id],
                    self.robosuite_env.sim.data.xpos[self.robosuite_env.peg1_body_id],
                ]
            )
        )

        rotate_by_180 = vtf.SE3.from_rotation(rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi))

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        square_nut_robot_base = (
            base_transform @ square_nut_world @ rotate_by_180 @ invert_grasp_pose
        )
        square_nut_handle_robot_base = (
            base_transform @ square_nut_handle_world @ rotate_by_180 @ invert_grasp_pose
        )
        peg_height = 0.1
        peg_robot_base = (
            base_transform
            @ peg_world
            @ vtf.SE3.from_translation(translation=np.array([0, 0, peg_height]))
            @ invert_grasp_pose
        )

        return {
            "nut_handle_to_center_offset": nut_handle_to_center_offset,
            "square_nut": np.concatenate(
                [square_nut_robot_base.translation(), square_nut_robot_base.rotation().wxyz]
            ),
            "square_nut_handle": np.concatenate(
                [
                    square_nut_handle_robot_base.translation(),
                    square_nut_handle_robot_base.rotation().wxyz,
                ]
            ),
            "square_peg": np.concatenate(
                [peg_robot_base.translation(), peg_robot_base.rotation().wxyz]
            ),
        }

    def compute_reward(self) -> float:
        """Compute reward from the Robosuite environment."""
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaRobosuiteNutAssembly format."""
        robosuite_obs = self.robosuite_env._get_observations(force_update=True)
        pose_dict = self._get_nut_pose(robosuite_obs)

        robosuite_obs["nut_poses"] = pose_dict

        # Nut assembly uses birdview camera but stores results under "robot0_robotview" key
        for camera_name in self.render_camera_names:
            if camera_name not in robosuite_obs:
                robosuite_obs[camera_name] = {}
            if "robot0_robotview" not in robosuite_obs:
                robosuite_obs["robot0_robotview"] = {}

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
                    vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
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

            robosuite_obs["robot0_robotview"]["pose"] = (
                np.concatenate(
                    [
                        cam_robot_tf.translation(),
                        cam_robot_tf.rotation().wxyz,
                    ]
                )
            )
            robosuite_obs["robot0_robotview"]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.robosuite_env.sim.model.camera_name2id(camera_name)
            fovy = self.robosuite_env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)

            K = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            robosuite_obs["robot0_robotview"]["intrinsics"] = K

            robosuite_obs["robot0_robotview"]["images"] = {}
            if camera_name + "_image" in robosuite_obs:
                robosuite_obs["robot0_robotview"]["images"]["rgb"] = robosuite_obs[
                    camera_name + "_image"
                ][::-1]
            if camera_name + "_depth" in robosuite_obs:
                depth_metric = get_real_depth_map(
                    self.robosuite_env.sim, robosuite_obs[camera_name + "_depth"][::-1]
                )
                robosuite_obs["robot0_robotview"]["images"]["depth"] = depth_metric

        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs

    # Nut assembly has its own viser debugging logic
    def _update_viser_server(self) -> None:
        obs = self.get_observation()
        if self.viser_debug:
            self._viser_init_check()

            obs["robot_cartesian_pos"][:-1]

            action_joint_copy = obs["robot_joint_pos"].copy()
            action_joint_copy[-1] /= self.gripper_metric_length

            self.urdf_vis.update_cfg(action_joint_copy)

            rbg_imgs = obs_get_rgb(obs)
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                if "pose" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

                # Point cloud from depth
                if "depth" in obs[image_key].get("images", {}):
                    points_robot, colors = depth_color_to_pointcloud(
                        obs[image_key]["images"]["depth"][:, :, 0],
                        rbg_imgs[image_key],
                        obs[image_key]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        f"{image_key}/point_cloud",
                        points_robot,
                        colors,
                        point_size=0.001,
                        point_shape="square",
                    )

            if "square_nut" in obs["nut_poses"]:
                self.viser_server.scene.add_frame(
                    "square_nut_frame",
                    position=obs["nut_poses"]["square_nut"][:3],
                    wxyz=obs["nut_poses"]["square_nut"][3:],
                    axes_length=0.05,
                    axes_radius=0.005,
                )
                self.viser_server.scene.add_frame(
                    "square_nut_handle_frame",
                    position=obs["nut_poses"]["square_nut_handle"][:3],
                    wxyz=obs["nut_poses"]["square_nut_handle"][3:],
                    axes_length=0.05,
                    axes_radius=0.005,
                )
                self.viser_server.scene.add_frame(
                    "peg_frame",
                    position=obs["nut_poses"]["square_peg"][:3],
                    wxyz=obs["nut_poses"]["square_peg"][3:],
                    axes_length=0.05,
                    axes_radius=0.005,
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

            if self.grasp_frame_position is not None and self.grasp_frame_orientation is not None:
                self.viser_server.scene.add_frame(
                    "robot0_robotview/grasp",
                    position=self.grasp_frame_position,
                    wxyz=self.grasp_frame_orientation,
                    axes_length=0.05,
                    axes_radius=0.0015,
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

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/panda_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

            self.mjcf_gripper_frame_handle = self.viser_server.scene.add_frame(
                "/panda_gripper_target_mjcf", axes_length=0.15, axes_radius=0.005
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


class FrankaRobosuiteNutAssemblyVisual(FrankaRobosuiteNutAssembly):
    def __init__(self, *args, **kwargs):
        # make privileged to False
        kwargs["privileged"] = False
        super().__init__(*args, **kwargs)


__all__ = ["FrankaRobosuiteNutAssembly", "FrankaRobosuiteNutAssemblyVisual"]
