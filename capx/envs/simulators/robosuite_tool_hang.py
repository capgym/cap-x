"""Low-level RoboSuite ToolHang environment for guarded code policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


class FrankaRobosuiteToolHang(RobosuiteBaseEnv):
    """Single-Panda ToolHang wrapper with exact, documented object poses."""

    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 10000,
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
        self.save_camera_name = "agentview"
        self.render_camera_names = [self.save_camera_name] if enable_render else []
        controller = load_composite_controller_config(controller=self.controller_cfg)
        self.robosuite_env = suite.environments.manipulation.tool_hang.ToolHang(
            robots=["Panda"],
            use_camera_obs=enable_render,
            has_renderer=False,
            has_offscreen_renderer=enable_render,
            camera_names=self.render_camera_names,
            camera_depths=enable_render,
            renderer="mujoco",
            reward_shaping=False,
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=controller,
            horizon=max_steps,
            seed=seed,
        )
        self._init_robot_links()
        self.home_joint_position: np.ndarray | None = None
        self.viser_debug = viser_debug
        self.viser_server = None
        self.reset(seed=seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            task_rng = np.random.default_rng(seed)
            self._rng = task_rng
            self.robosuite_env.rng = task_rng
            for sampler in self.robosuite_env.placement_initializer.samplers.values():
                sampler.rng = task_rng

        first_obs = self.robosuite_env.reset()
        self.home_joint_position = np.asarray(first_obs["robot0_joint_pos"], dtype=np.float64)
        self.robosuite_env.sim.data.qpos[:7] = np.array(
            [0, -1.585, 0, -2.645, 0, 1, 0.785], dtype=np.float64
        )
        self.robosuite_env.sim.forward()
        self._current_joints = self.robosuite_env.sim.data.qpos[:7].copy()
        self._step_count = 0
        self._sim_step_count = 0
        self._gripper_fraction = 1.0
        for _ in range(100):
            self._step_once()
        return self.get_observation(), {
            "task_prompt": "Insert the hook frame into the stand, then hang the wrench on the frame. Quaternions are WXYZ."
        }

    def _pose_in_robot_base(
        self, position: np.ndarray, rotation_matrix: np.ndarray
    ) -> np.ndarray:
        world_pose = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(np.asarray(rotation_matrix).reshape(3, 3)),
            translation=np.asarray(position, dtype=np.float64),
        )
        grasp_frame = (
            vtf.SE3.from_rotation(rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi))
            @ vtf.SE3.from_rotation(rotation=vtf.SO3.from_rpy_radians(np.pi, 0.0, 0.0))
        )
        robot_pose = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ world_pose
            @ grasp_frame
        )
        return np.concatenate([robot_pose.translation(), robot_pose.rotation().wxyz])

    def _body_pose(self, name: str) -> np.ndarray:
        body_id = self.robosuite_env.sim.model.body_name2id(name)
        return self._pose_in_robot_base(
            self.robosuite_env.sim.data.body_xpos[body_id],
            self.robosuite_env.sim.data.body_xmat[body_id],
        )

    def _geom_pose(self, name: str) -> np.ndarray:
        geom_id = self.robosuite_env.sim.model.geom_name2id(name)
        return self._pose_in_robot_base(
            self.robosuite_env.sim.data.geom_xpos[geom_id],
            self.robosuite_env.sim.data.geom_xmat[geom_id],
        )

    def _site_pose(self, name: str) -> np.ndarray:
        site_id = self.robosuite_env.sim.model.site_name2id(name)
        return self._pose_in_robot_base(
            self.robosuite_env.sim.data.site_xpos[site_id],
            self.robosuite_env.sim.data.site_xmat[site_id],
        )

    def get_observation(self) -> dict[str, Any]:
        observation = self.robosuite_env._get_observations(force_update=True)
        stand_mount_id = self.robosuite_env.sim.model.site_name2id("stand_mount_site")
        stand_mount_matrix = self.robosuite_env.sim.data.site_xmat[stand_mount_id].reshape(3, 3)
        # RoboSuite's assembly predicate measures the frame tip against the
        # stand base geom (not the mount site's opening). Use that exact
        # position while retaining the upright mount orientation.
        stand_base_id = self.robosuite_env.obj_geom_id["stand_base"]
        frame_insert_position = self.robosuite_env.sim.data.geom_xpos[stand_base_id]
        frame_hang_id = self.robosuite_env.sim.model.site_name2id("frame_hang_site")
        frame_intersection_id = self.robosuite_env.sim.model.site_name2id(
            "frame_intersection_site"
        )
        frame_hang_position = self.robosuite_env.sim.data.site_xpos[frame_hang_id]
        frame_intersection_position = self.robosuite_env.sim.data.site_xpos[
            frame_intersection_id
        ]
        tool_hang_position = frame_hang_position + 0.3 * (
            frame_intersection_position - frame_hang_position
        )
        observation["tool_hang_poses"] = {
            "stand": self._body_pose("stand_root"),
            "stand_mount": self._site_pose("stand_mount_site"),
            "frame_insert_target": self._pose_in_robot_base(
                frame_insert_position, stand_mount_matrix
            ),
            "frame": self._body_pose("frame_root"),
            "frame_grip": self._geom_pose("frame_grip_frame"),
            "frame_tip": self._site_pose("frame_tip_site"),
            "frame_hang": self._site_pose("frame_hang_site"),
            "tool_hang_target": self._pose_in_robot_base(
                tool_hang_position,
                self.robosuite_env.sim.data.site_xmat[frame_hang_id],
            ),
            "tool": self._body_pose("tool_root"),
            "tool_grip": self._body_pose("tool_grip_main"),
            "tool_hole": self._site_pose("tool_hole1_center"),
        }
        if self.render_camera_names:
            self._process_camera_observations(observation)
        self._compute_gripper_obs(observation)
        return observation

    def compute_reward(self) -> float:
        return float(self.robosuite_env.reward())

    def task_completed(self) -> bool:
        return bool(self.robosuite_env._check_success())


__all__ = ["FrankaRobosuiteToolHang"]
