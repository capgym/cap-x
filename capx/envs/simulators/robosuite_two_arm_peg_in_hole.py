"""Low-level two-Panda RoboSuite peg-in-hole environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from capx.envs.simulators.robosuite_handover import RobosuiteHandoverEnv


class RobosuiteTwoArmPegInHoleEnv(RobosuiteHandoverEnv):
    """Two opposed Panda arms with a peg and plate rigidly attached to their EEFs."""

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1000,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = True,
        enable_render: bool = False,
    ) -> None:
        del viser_debug, privileged
        self.controller_cfg = controller_cfg
        self.max_steps = max_steps
        self.save_camera_name = "agentview"
        self.render_camera_names = [self.save_camera_name] if enable_render else []
        self._render_width = 512
        self._render_height = 512
        self._step_count = 0
        self._sim_step_count = 0
        self._rng = np.random.default_rng(seed)
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._subsample_rate = 5

        controller = load_composite_controller_config(controller=self.controller_cfg)
        self.robosuite_env = suite.environments.manipulation.two_arm_peg_in_hole.TwoArmPegInHole(
            robots=["Panda", "Panda"],
            env_configuration="opposed",
            gripper_types=None,
            controller_configs=[controller, controller],
            use_camera_obs=False,
            use_object_obs=True,
            has_renderer=False,
            has_offscreen_renderer=enable_render,
            camera_names=self.render_camera_names,
            renderer="mujoco",
            reward_shaping=False,
            horizon=max_steps,
            seed=seed,
        )
        model = self.robosuite_env.sim.model
        self.base_link_idx_0 = model.body_name2id("fixed_mount0_base")
        self.base_link_idx_1 = model.body_name2id("fixed_mount1_base")
        self.eef_link_idx_0 = model.body_name2id("gripper0_right_eef")
        self.eef_link_idx_1 = model.body_name2id("gripper1_right_eef")
        self.base_link_wxyz_xyz_0 = self._body_wxyz_xyz(self.base_link_idx_0)
        self.base_link_wxyz_xyz_1 = self._body_wxyz_xyz(self.base_link_idx_1)
        self.home_joint_position_0: np.ndarray | None = None
        self.home_joint_position_1: np.ndarray | None = None
        self.reset(seed=seed)

    def _body_wxyz_xyz(self, body_id: int) -> np.ndarray:
        return np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[body_id],
                self.robosuite_env.sim.data.xpos[body_id],
            ]
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            task_rng = np.random.default_rng(seed)
            self._rng = task_rng
            self.robosuite_env.rng = task_rng
        observation = self.robosuite_env.reset()
        self.home_joint_position_0 = np.asarray(
            observation["robot0_joint_pos"], dtype=np.float64
        )
        self.home_joint_position_1 = np.asarray(
            observation["robot1_joint_pos"], dtype=np.float64
        )
        self._step_count = 0
        self._sim_step_count = 0
        for _ in range(10):
            self._step_once()
        return self.get_observation(), {
            "task_prompt": "Move the peg held by arm 0 into the hole held by arm 1. Quaternions are WXYZ."
        }

    def _step_once(self) -> None:
        observation = self.robosuite_env._get_observations(force_update=True)
        action = np.concatenate(
            [observation["robot0_joint_pos"], observation["robot1_joint_pos"]]
        )
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self.robosuite_env.step(action)
            self._record_frame()
        else:
            self.robosuite_env.step(action, skip_render_images=True)
        self._sim_step_count += 1

    def _move_arm(
        self, arm: int, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 120
    ) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        for _ in range(max_steps):
            observation = self.robosuite_env._get_observations(force_update=True)
            current0 = np.asarray(observation["robot0_joint_pos"], dtype=np.float64)
            current1 = np.asarray(observation["robot1_joint_pos"], dtype=np.float64)
            current = current0 if arm == 0 else current1
            if np.linalg.norm(current - target) < tolerance:
                break
            action = np.concatenate(
                [target if arm == 0 else current0, target if arm == 1 else current1]
            )
            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self.robosuite_env.step(action)
                self._record_frame()
            else:
                self.robosuite_env.step(action, skip_render_images=True)
            self._sim_step_count += 1

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 120
    ) -> None:
        self._move_arm(0, joints, tolerance=tolerance, max_steps=max_steps)

    def move_to_joints_blocking_arm1(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 120
    ) -> None:
        self._move_arm(1, joints, tolerance=tolerance, max_steps=max_steps)

    def _world_pose_in_robot0(self, position: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        world = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(np.asarray(matrix).reshape(3, 3)),
            translation=np.asarray(position, dtype=np.float64),
        )
        pose = vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse() @ world
        return np.concatenate([pose.translation(), pose.rotation().wxyz])

    def _body_pose_in_robot0(self, body_id: int) -> np.ndarray:
        return self._world_pose_in_robot0(
            self.robosuite_env.sim.data.body_xpos[body_id],
            self.robosuite_env.sim.data.body_xmat[body_id],
        )

    def get_observation(self) -> dict[str, Any]:
        observation = self.robosuite_env._get_observations(force_update=True)
        hole_id = self.robosuite_env.hole_body_id
        hole_position = self.robosuite_env.sim.data.body_xpos[hole_id]
        hole_matrix = self.robosuite_env.sim.data.body_xmat[hole_id].reshape(3, 3)
        target_position = hole_position + hole_matrix @ np.array([0.1, 0.0, 0.0])
        align_peg_z_to_hole_x = vtf.SO3.from_rpy_radians(0.0, np.pi / 2.0, 0.0).as_matrix()
        target_matrix = hole_matrix @ align_peg_z_to_hole_x
        observation["peg_in_hole_poses"] = {
            "peg": self._body_pose_in_robot0(self.robosuite_env.peg_body_id),
            "hole": self._body_pose_in_robot0(hole_id),
            "hole_target": self._world_pose_in_robot0(target_position, target_matrix),
            "arm0_eef": self._body_pose_in_robot0(self.eef_link_idx_0),
            "arm1_eef": self._body_pose_in_robot0(self.eef_link_idx_1),
        }
        return observation

    def compute_reward(self) -> float:
        return float(self.robosuite_env.reward())

    def task_completed(self) -> bool:
        return bool(self.robosuite_env._check_success())


__all__ = ["RobosuiteTwoArmPegInHoleEnv"]
