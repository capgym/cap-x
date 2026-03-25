"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Stack environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

import collections.abc
from copy import copy
from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.models.objects.primitive.box import BoxObject
from robosuite.utils import RandomizationError
from robosuite.utils.placement_samplers import ObjectPositionSampler
from robosuite.utils.transform_utils import quat_multiply

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


class StackedObjectRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_range=(0, 0),
        y_range=(0, 0),
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
        rng=None,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
            rng=rng,
        )

    def _sample_x(self, object_horizontal_radius):
        minimum, maximum = self.x_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        minimum, maximum = self.y_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        if self.rotation is None:
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = self.rng.uniform(high=max(self.rotation), low=min(self.rotation))
        else:
            rot_angle = self.rotation

        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            raise ValueError(
                f"Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {self.rotation_axis}"
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert reference in placed_objects, (
                f"Invalid reference received. Current options are: {placed_objects.keys()}, requested: {reference}"
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert base_offset.shape[0] == 3, (
                f"Invalid reference received. Should be (x,y,z) 3-tuple, but got: {base_offset}"
            )

        def _location_valid(object_x, object_y, object_z, horizontal_radius, bottom_offset):
            if not self.ensure_valid_placement:
                return True
            for (x, y, z), _, other_obj in placed_objects.values():
                if (
                    np.linalg.norm((object_x - x, object_y - y))
                    <= other_obj.horizontal_radius + horizontal_radius
                ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                    return False
            return True

        mujoco_objects = list(self.mujoco_objects)

        # Special-case: place the second object directly on top of the first.
        if len(mujoco_objects) >= 2:
            first_obj = mujoco_objects[0]
            second_obj = mujoco_objects[1]

            assert first_obj.name not in placed_objects, (
                f"Object '{first_obj.name}' has already been sampled!"
            )
            assert second_obj.name not in placed_objects, (
                f"Object '{second_obj.name}' has already been sampled!"
            )

            first_hr = first_obj.horizontal_radius
            first_bo = first_obj.bottom_offset
            second_hr = second_obj.horizontal_radius
            second_bo = second_obj.bottom_offset

            success = False
            for _ in range(5000):
                object_x = self._sample_x(first_hr) + base_offset[0]
                object_y = self._sample_y(first_hr) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= first_bo[-1]

                if not _location_valid(object_x, object_y, object_z, first_hr, first_bo):
                    continue

                quat1 = self._sample_quat()
                if hasattr(first_obj, "init_quat"):
                    quat1 = quat_multiply(quat1, first_obj.init_quat)
                pos1 = (object_x, object_y, object_z)
                placed_objects[first_obj.name] = (pos1, quat1, first_obj)

                z_gap = max(float(self.z_offset), 0.01)
                object2_x = object_x - 0.01
                object2_y = object_y
                object2_z = object_z + first_obj.top_offset[-1] - second_bo[-1] + z_gap

                if not _location_valid(object2_x, object2_y, object2_z, second_hr, second_bo):
                    placed_objects.pop(first_obj.name, None)
                    continue

                quat2 = self._sample_quat()
                if hasattr(second_obj, "init_quat"):
                    quat2 = quat_multiply(quat2, second_obj.init_quat)
                pos2 = (object2_x, object2_y, object2_z)
                placed_objects[second_obj.name] = (pos2, quat2, second_obj)

                success = True
                break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            remaining_objects = mujoco_objects[2:]
        else:
            remaining_objects = mujoco_objects

        for obj in remaining_objects:
            assert obj.name not in placed_objects, f"Object '{obj.name}' has already been sampled!"

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for _ in range(5000):
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                if _location_valid(object_x, object_y, object_z, horizontal_radius, bottom_offset):
                    quat = self._sample_quat()
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class FrankaRobosuiteCubesRestackLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka Stack environment with FrankaPickPlaceLowLevel-compatible interface."""

    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
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

        self.cube_A_length = 0.02
        self.cube_B_length = 0.02

        # Initialize Robosuite environment
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = suite.environments.manipulation.stack.Stack(
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
                    cube_A_length=self.cube_A_length,
                    cube_B_length=self.cube_B_length,
                )
            else:
                self.robosuite_env = suite.environments.manipulation.stack.Stack(
                    robots=["Panda"],
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    camera_depths=True,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                    reward_shaping=True,
                    cube_A_length=self.cube_A_length,
                    cube_B_length=self.cube_B_length,
                )
        else:
            self.robosuite_env = suite.environments.manipulation.stack.Stack(
                robots=["Panda"],
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                # camera_segmentations=self.segmentation_level,
                camera_depths=True,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=load_composite_controller_config(controller=self.controller_cfg),
                horizon=max_steps,
                reward_shaping=True,
                cube_A_length=self.cube_A_length,
                cube_B_length=self.cube_B_length,
            )

        self.robosuite_env.cubeA = BoxObject(
            name="cubeA",
            size=[self.cube_A_length, self.cube_A_length, self.cube_A_length],
            rgba=[1, 0, 0, 1],
        )
        self.robosuite_env.cubeB = BoxObject(
            name="cubeB",
            size=[self.cube_B_length, self.cube_B_length, self.cube_B_length],
            rgba=[0, 1, 0, 1],
        )
        cubes = [self.robosuite_env.cubeA, self.robosuite_env.cubeB]

        self.robosuite_env.placement_initializer = StackedObjectRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cubes,
            x_range=[-0.12, 0.16],
            y_range=[-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.robosuite_env.table_offset,
            z_offset=0.01,
            rng=self.robosuite_env.rng,
        )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        # Adjust initial orientation
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        # We do this because for some reason modifying qpos does not update robot0_joint_pos
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )
        self.robosuite_env.cubeA = BoxObject(
            name="cubeA",
            size=[self.cube_A_length, self.cube_A_length, self.cube_A_length],
            rgba=[1, 0, 0, 1],
        )
        self.robosuite_env.cubeB = BoxObject(
            name="cubeB",
            size=[self.cube_B_length, self.cube_B_length, self.cube_B_length],
            rgba=[0, 1, 0, 1],
        )

        # Settle the environment
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        info = {
            "task_prompt": "Place the primary cube on top of the secondary cube. Quaternions are WXYZ."
        }
        return obs, info

    def _cube_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get cube poses in robot base frame."""
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        cubeA_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cubeA_quat"], robosuite_obs["cubeA_pos"]])
        )
        cubeB_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cubeB_quat"], robosuite_obs["cubeB_pos"]])
        )

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        cubeA_robot_base = base_transform @ cubeA_world
        cubeB_robot_base = base_transform @ cubeB_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [cubeA_robot_base.translation(), cubeA_robot_base.rotation().wxyz]
                )
            ],
            "secondary": [
                float(x)
                for x in np.concatenate(
                    [cubeB_robot_base.translation(), cubeB_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute dense stacking reward."""
        if self.robosuite_env.reward(action=None) > 0.4:
            # Secondary check to make sure both cubes are not up in the air
            robosuite_obs = self.robosuite_env._get_observations()
            pose_dict = self._cube_pose_dict(robosuite_obs)
            cube_pose_array = np.stack(
                [
                    np.asarray(pose_dict["primary"], dtype=np.float32),
                    np.asarray(pose_dict["secondary"], dtype=np.float32),
                ],
                axis=0,
            )
            if cube_pose_array[0, 2] > 0.04 and cube_pose_array[1, 2] > 0.04:
                return 0.0
            else:
                return self.robosuite_env.reward(action=None)
        else:
            return self.robosuite_env.reward(action=None)

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        if self.robosuite_env._check_success():
            # Secondary check to make sure both cubes are not up in the air
            robosuite_obs = self.robosuite_env._get_observations()
            pose_dict = self._cube_pose_dict(robosuite_obs)
            cube_pose_array = np.stack(
                [
                    np.asarray(pose_dict["primary"], dtype=np.float32),
                    np.asarray(pose_dict["secondary"], dtype=np.float32),
                ],
                axis=0,
            )
            if cube_pose_array[0, 2] > 0.04 and cube_pose_array[1, 2] > 0.04:
                return False
            else:
                return True
        else:
            return False

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._cube_pose_dict(robosuite_obs)
        cube_pose_array = np.stack(
            [
                np.asarray(pose_dict["primary"], dtype=np.float32),
                np.asarray(pose_dict["secondary"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["cube_poses"] = {
            "primary": cube_pose_array[0],
            "secondary": cube_pose_array[1],
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs


__all__ = ["FrankaRobosuiteCubesRestackLowLevel"]
