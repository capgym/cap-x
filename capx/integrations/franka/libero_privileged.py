from typing import Any, Callable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase


# ------------------------------- Control API ------------------------------
class FrankaLiberoPrivilegedApi(ApiBase):
    """Robot control helpers for Franka.
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._target_link_name = ctx.target_link_name
        self._pks = pks
        self.cfg = self._env.get_observation()["robot_joint_pos"]
        self.camera_name = "agentview"
        self.wrist_camera_name = "robot0_eye_in_hand"

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "get_all_object_poses": self.get_all_object_poses,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "goto_pose_interactive_cartesian": self.goto_pose_interactive_cartesian,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_eye_in_hand"]["images"]["depth"]: Current wrist camera depth image as a numpy array of shape (H, W), dtype float32.
                - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot_cartesian_pos"]: Current end-effector (panda_hand) pose in the robot/world frame as a numpy array of shape (8,), dtype float64. The first 3 elements are the robot's end-effector XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
                - ["robot_joint_pos"]: Current joint positions as a numpy array of shape (7,), dtype float64. The last element is the gripper position normalized, 0 (closed) to 1 (open).
        """
        obs = self._env.get_observation()
        obs[self.camera_name]["images"]["depth"] = obs[self.camera_name]["images"]["depth"].squeeze(-1)
        obs[self.wrist_camera_name]["images"]["depth"] = obs[self.wrist_camera_name]["images"]["depth"].squeeze(-1)
        return obs

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of, in underscore separated lowercase words.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        return self._env._get_object_pose(object_name)

    def get_all_object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get the poses of all objects in the scene (both movable and fixed).

        Returns:
            Dictionary mapping object name to a tuple of:
                position: (3,) XYZ in meters.
                quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        return self._env._get_all_object_poses()

    def sample_grasp_pose(self, object_name: str) -> None:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for, in underscore separated lowercase words.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        pos, _ = self._env._get_object_pose(object_name)
        return pos, np.array([0, 1, 0, 0])

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        # Align with legacy env: apply TCP offset in end-effector frame
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if (
            z_approach != 0.0
        ):  # If z_approach is not 0.0, approach the object from above by z_approach meters
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None:
            self.cfg = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
        else:
            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def goto_pose_interactive_cartesian(self, target_pose_predicate: Callable[[], Tuple[np.ndarray, np.ndarray]], replan_interval_s: float = 0.0, lin_vel_norm: float = 1.0, ang_vel_norm: float = 2.0, z_approach: float = 0.0, timeout_s: float = 20.0) -> None:
        """Go to pose using Inverse Kinematics.
        Target pose is updated regularly.
        Trajectory is mostly linear in Cartesian space.
        Args:
            target_pose_predicate: A function that returns the target position (3,) and WXYZ quaternion (4,) of the target pose.
            replan_interval_s: The interval at which to update the target pose. Default is 0.0, which means no replanning.
            lin_vel_norm: The desired EEF linear velocity norm in m/s. Default is 1.0.
            ang_vel_norm: The desired EEF angular velocity norm in rad/s. Default is 2.0.
            z_approach: The Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        position, quaternion_wxyz = target_pose_predicate()

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        # Align with legacy env: apply TCP offset in end-effector frame
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if (z_approach != 0.0):
            def target_pose_predicate_z_offset():
                pos, quat = target_pose_predicate()
                r = SciRotation.from_quat(quat, scalar_first=True)
                return pos + r.apply(np.array([0, 0, -z_approach])), quat

            self.goto_pose_interactive_cartesian(target_pose_predicate_z_offset, replan_interval_s, lin_vel_norm, ang_vel_norm, z_approach=0.0, timeout_s=timeout_s)

        robot_cartesian_pos = self._env.get_observation()["robot_cartesian_pos"]
        current_position = robot_cartesian_pos[:3]
        current_quaternion_wxyz = robot_cartesian_pos[3:7]
        current_target_position = current_position.copy()
        current_target_quaternion_wxyz = current_quaternion_wxyz.copy()
        last_plan_time_s = self._env.get_current_time_s()
        start_loop_time_s = self._env.get_current_time_s()

        loop_executed = False
        while not np.allclose(offset_pos, current_target_position) or not np.allclose(quaternion_wxyz, current_target_quaternion_wxyz):
            loop_executed = True
            current_time_s = self._env.get_current_time_s()
            delta_t = current_time_s - last_plan_time_s
            if timeout_s > 0 and current_time_s - start_loop_time_s > timeout_s:
                break
            if replan_interval_s > 0 and delta_t > replan_interval_s:
                offset_pos, quaternion_wxyz = target_pose_predicate()
                last_plan_time_s = current_time_s
                robot_cartesian_pos = self._env.get_observation()["robot_cartesian_pos"]
                current_position = robot_cartesian_pos[:3]
                current_quaternion_wxyz = robot_cartesian_pos[3:7]

            current_target_position, current_target_quaternion_wxyz = self.step_towards_pose(
                current_position, current_quaternion_wxyz, offset_pos, quaternion_wxyz, lin_vel_norm, ang_vel_norm, 1.0 / self._env._control_freq
            )
            current_position = current_target_position
            current_quaternion_wxyz = current_target_quaternion_wxyz

            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=current_target_position,
                target_wxyz=current_target_quaternion_wxyz,
                prev_cfg=self.cfg,
            )
            joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints, max_steps=1)
        if loop_executed:
            self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        self._env._set_gripper(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        self._env._set_gripper(0.0)
        for _ in range(60):
            self._env._step_once()

    def breakpoint_code_block(self) -> None:
        """Call this function to mark a significant checkpoint where you want to evaluate progress and potentially regenerate the remaining code.

        Args:
            None
        """
        return None

    def _normalize(self, q: np.ndarray) -> np.ndarray:
        return q / np.linalg.norm(q)

    def _slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """SLERP between two unit quaternions (WXYZ, shape (4,)), shortest path."""
        q0 = self._normalize(q0)
        q1 = self._normalize(q1)

        dot = np.dot(q0, q1)

        # Take shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        # If almost identical, nlerp
        if dot > 0.9995:
            q = (1.0 - t) * q0 + t * q1
            return self._normalize(q)

        theta = np.arccos(dot)           # geodesic angle on S^3
        sin_theta = np.sin(theta)

        w0 = np.sin((1.0 - t) * theta) / sin_theta
        w1 = np.sin(t * theta) / sin_theta

        return w0 * q0 + w1 * q1


    def step_towards_pose(
        self,
        pos_curr: np.ndarray,
        quat_curr: np.ndarray,
        pos_target: np.ndarray,
        quat_target: np.ndarray,
        target_linear_speed: float,
        target_angular_speed: float,
        delta_t: float,
    ):
        """
        Take one step from (pos_curr, quat_curr) towards (pos_target, quat_target),
        given linear and angular speed limits.

        - Positions: shape (3,)
        - Quaternions: WXYZ, shape (4,), assumed unit
        - target_linear_speed: linear speed norm (units / s)
        - target_angular_speed: angular speed norm (rad / s)
        - delta_t: time step (s)

        Returns:
            (pos_next: (3,), quat_next: (4,))
        """
        # -------- Linear part --------
        max_lin_step = max(target_linear_speed, 0.0) * delta_t

        diff_pos = pos_target - pos_curr
        dist = np.linalg.norm(diff_pos)

        if dist <= max_lin_step or dist < 1e-9:
            pos_next = pos_target.copy()
        else:
            direction = diff_pos / dist
            pos_next = pos_curr + direction * max_lin_step

        # -------- Angular part --------
        q0 = self._normalize(quat_curr)
        q1 = self._normalize(quat_target)

        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        # Geodesic angle on S^3
        theta = np.arccos(dot)
        # Physical rotation angle in SO(3)
        delta_angle = 2.0 * theta

        max_ang_step = max(target_angular_speed, 0.0) * delta_t

        if delta_angle <= max_ang_step or delta_angle < 1e-8:
            # Close enough: snap to target orientation
            quat_next = q1
        elif max_ang_step <= 0.0:
            # No angular motion allowed this step
            quat_next = q0
        else:
            # Move along the rotation by max_ang_step
            # SLERP parameter t is fraction of total rotation angle
            t = max_ang_step / delta_angle  # in (0,1)
            quat_next = self._slerp(q0, q1, t)

        return pos_next, quat_next