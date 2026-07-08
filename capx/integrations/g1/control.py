from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.integrations.franka.control import FrankaControlApi
from capx.integrations.g1.grasp import (
    FRONT_POLICY_QUATERNION_WXYZ,
    front_policy_pinch_target_pose,
    front_pregrasp_pinch_center_pose,
    palm_pose_from_pinch_center_pose,
    remove_legacy_grasp_local_z_offset,
    closed_pinch_center_offset,
)
from capx.integrations.g1.sdk import (
    G1_DUAL_ARM_NUM_JOINTS,
    G1_NUM_ARM_JOINTS,
    G1_RIGHT_ARM_DUAL_CFG_SLICE,
    G1_RIGHT_ARM_WITH_HAND_CFG_SLICE,
    G1_WITH_HAND_NUM_JOINTS,
    as_g1_arm_joints,
    dex3_grasp_joints,
)


class G1RealControlApi(FrankaControlApi):
    """CaP-X API for the Unitree G1 right arm and right Dex3 hand."""

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] | None = None,
        use_sam3: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(
            env,
            tcp_offset=[0.0, 0.0, 0.0] if tcp_offset is None else tcp_offset,
            use_sam3=use_sam3,
            real=True,
            debug=debug,
        )

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "sample_grasp_center_pose": self.sample_grasp_center_pose,
            "goto_pose": self.goto_pose,
            "move_pinch_center_to_pose": self.move_pinch_center_to_pose,
            "move_pinch_center_horizontal_line": self.move_pinch_center_horizontal_line,
            "move_to_pinch_pregrasp": self.move_to_pinch_pregrasp,
            "grasp_at_pinch_center": self.grasp_at_pinch_center,
            "move_to_joints": self.move_to_joints,
            "move_to_pregrasp_side_pose": self.move_to_pregrasp_side_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "close_index_pinch": self.close_index_pinch,
            "close_middle_pinch": self.close_middle_pinch,
            "set_gripper_trigger_squeeze": self.set_gripper_trigger_squeeze,
            "move_hand_joints": self.move_hand_joints,
        }

    def _current_joints(self) -> np.ndarray:
        if hasattr(self._env, "get_current_arm_joints"):
            return as_g1_arm_joints(self._env.get_current_arm_joints()).copy()
        get_observation = getattr(self._env, "get_observation", None)
        if not callable(get_observation):
            return np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)
        obs = get_observation()
        joints = obs.get("robot_joint_pos") if isinstance(obs, dict) else None
        if joints is None:
            return np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)
        return as_g1_arm_joints(joints).copy()

    @staticmethod
    def _extract_right_arm_joints(cfg: np.ndarray) -> np.ndarray:
        arr = np.asarray(cfg, dtype=np.float64).reshape(-1)
        if arr.size == G1_NUM_ARM_JOINTS:
            return as_g1_arm_joints(arr)
        if arr.size == G1_WITH_HAND_NUM_JOINTS:
            return as_g1_arm_joints(arr[G1_RIGHT_ARM_WITH_HAND_CFG_SLICE])
        if arr.size >= G1_DUAL_ARM_NUM_JOINTS:
            return as_g1_arm_joints(arr[G1_RIGHT_ARM_DUAL_CFG_SLICE])
        return as_g1_arm_joints(arr)

    def _solve_ik(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        target_pose = np.concatenate([quaternion_wxyz, position])
        try:
            prev_cfg = self._current_joints() if self.cfg is None else self.cfg
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=target_pose,
                prev_cfg=prev_cfg,
            )
        except Exception as exc:
            pos_str = np.array2string(np.asarray(position), precision=4, suppress_small=True)
            quat_str = np.array2string(np.asarray(quaternion_wxyz), precision=4, suppress_small=True)
            raise RuntimeError(
                f"G1 PyRoKi IK failed for position={pos_str}, quaternion_wxyz={quat_str}. {exc}"
            ) from exc
        return self._extract_right_arm_joints(self.cfg)

    def _closed_pinch_center_offset(self) -> np.ndarray:
        return closed_pinch_center_offset()

    def _latest_camera_to_world(self) -> tuple[np.ndarray, SciRotation] | None:
        env = getattr(self, "_env", None)
        get_observation = getattr(env, "get_observation", None)
        if not callable(get_observation):
            return None
        try:
            obs = get_observation()
        except Exception:
            return None
        camera_obs = obs.get("robot0_robotview") if isinstance(obs, dict) else None
        if not isinstance(camera_obs, dict) or "pose" not in camera_obs:
            return None
        pose = np.asarray(camera_obs["pose"], dtype=np.float64).reshape(-1)
        if pose.size < 7 or not np.isfinite(pose[:7]).all():
            return None
        quat_wxyz = pose[3:7]
        quat_norm = float(np.linalg.norm(quat_wxyz))
        if quat_norm <= 1e-12:
            return None
        quat_wxyz = quat_wxyz / quat_norm
        rot = SciRotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pose[:3].copy(), rot

    @staticmethod
    def _camera_points_to_world(points: Any, camera_to_world: tuple[np.ndarray, SciRotation] | None) -> np.ndarray | None:
        if points is None:
            return None
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0 or arr.shape[-1] != 3:
            return None
        arr = arr.reshape((-1, 3))
        finite = arr[np.isfinite(arr).all(axis=1)]
        if finite.size == 0:
            return None
        if camera_to_world is None:
            return finite
        translation, rotation = camera_to_world
        return translation.reshape(1, 3) + rotation.apply(finite)

    @staticmethod
    def _camera_transform_for_debug(camera_to_world: tuple[np.ndarray, SciRotation] | None) -> vtf.SE3:
        if camera_to_world is None:
            return vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(wxyz=np.asarray([1.0, 0.0, 0.0, 0.0])),
                translation=np.zeros(3, dtype=np.float64),
            )
        translation, rotation = camera_to_world
        quat_xyzw = rotation.as_quat()
        return vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])),
            translation=translation,
        )

    def _best_contact_points_camera(self) -> np.ndarray | None:
        env = getattr(self, "_env", None)
        contact_points = getattr(env, "grasp_contact_pts", None)
        if contact_points is None:
            return None
        arr = np.asarray(contact_points, dtype=np.float64)
        if arr.size == 0 or arr.shape[-1] != 3:
            return None
        if arr.ndim >= 3:
            scores = np.asarray(getattr(env, "grasp_scores", []), dtype=np.float64).reshape(-1)
            best_idx = int(np.argmax(scores)) if scores.size == arr.shape[0] else 0
            arr = arr[best_idx]
        return arr.reshape((-1, 3))

    def sample_grasp_center_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample the front-policy closed-pinch target for a G1 Dex3 grasp.

        Args:
            object_name: Natural-language object name, for example "plastic water bottle".

        Returns:
            position: (3,) target XYZ in the robot/world frame where the closed
                Dex3 pinch center should land. XY is taken from the best available
                contact-point centroid, and Z is taken from the visual mask point
                cloud bounding-box center when available.
            quaternion_wxyz: Fixed vertical front-palm orientation. The G1
                front-grasp policy intentionally removes the ContactGraspNet
                rotation.

        Notes:
            Use this with grasp_at_pinch_center(). The inherited sample_grasp_pose()
            is kept for compatibility, but it applies an old local +Z palm offset
            and preserves the network grasp rotation.
        """

        self._log_step("sample_grasp_center_pose", f"Planning front-policy pinch target for **'{object_name}'**.")
        palm_pos, quat_wxyz = FrankaControlApi.sample_grasp_pose(self, object_name)
        fallback_center_pos = remove_legacy_grasp_local_z_offset(palm_pos, quat_wxyz)
        camera_to_world = self._latest_camera_to_world()
        env = getattr(self, "_env", None)
        visual_points_world = self._camera_points_to_world(getattr(env, "cube_points", None), camera_to_world)
        contact_points_world = self._camera_points_to_world(self._best_contact_points_camera(), camera_to_world)
        target_pos, target_quat = front_policy_pinch_target_pose(
            fallback_center_pos,
            quat_wxyz,
            visual_points=visual_points_world,
            contact_points=contact_points_world,
        )
        env = getattr(self, "_env", None)
        if getattr(env, "cube_points", None) is not None:
            cam_tf = self._camera_transform_for_debug(camera_to_world)
            target_tf = vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(wxyz=target_quat),
                translation=target_pos,
            )
            self._save_grasp_debug_visualization(
                object_name=object_name,
                points_camera=getattr(env, "cube_points"),
                colors=getattr(env, "cube_color", np.empty((0, 3), dtype=np.uint8)),
                camera_tf_world=cam_tf,
                grasp_tf_world=target_tf,
            )
        pos_str = np.array2string(target_pos, precision=4, suppress_small=True)
        print(f"Front-policy pinch target position for {object_name}: {target_pos}")
        print(f"Front-policy pinch target quaternion wxyz for {object_name}: {target_quat}")
        self._log_step_update(text=f"Front-policy pinch target: {pos_str}")
        return np.asarray(target_pos, dtype=np.float64), np.asarray(target_quat, dtype=np.float64)

    def move_pinch_center_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray | None = None,
        z_approach: float = 0.0,
    ) -> None:
        """Move so the closed Dex3 pinch center lands at a target pose.

        Args:
            position: (3,) target XYZ for the closed thumb/index/middle pinch
                center, in the robot/world frame. This should be the visual grasp
                center returned by sample_grasp_center_pose().
            quaternion_wxyz: (4,) target orientation in WXYZ order. If None, the
                G1 fixed vertical front-palm orientation is used.
            z_approach: Optional world +Z approach offset before the final motion.

        Notes:
            PyRoKi still solves IK for right_hand_palm_link. This function first
            converts the desired closed-pinch-center pose into the palm-link pose,
            using the G1 Dex3 closed-hand geometry from the URDF.
        """

        quat = (
            FRONT_POLICY_QUATERNION_WXYZ.copy()
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )
        palm_pos, palm_quat = palm_pose_from_pinch_center_pose(
            np.asarray(position, dtype=np.float64).reshape(3),
            quat,
            pinch_center_offset=self._closed_pinch_center_offset(),
        )
        pos_str = np.array2string(np.asarray(position), precision=4, suppress_small=True)
        palm_str = np.array2string(palm_pos, precision=4, suppress_small=True)
        self._log_step(
            "move_pinch_center_to_pose",
            f"Moving closed Dex3 pinch center to {pos_str}; palm IK target is {palm_str}.",
        )
        self.goto_pose(palm_pos, palm_quat, z_approach=z_approach)
        self._log_step_update(text="Closed Dex3 pinch center motion complete.")

    def move_pinch_center_horizontal_line(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        quaternion_wxyz: np.ndarray | None = None,
        *,
        num_waypoints: int = 50,
        dt: float | None = None,
        max_joint_step: float | None = None,
        hold_final_seconds: float = 0.4,
    ) -> None:
        """Stream a horizontal closed-pinch-center approach while keeping Z stable.

        Args:
            start_position: (3,) starting pinch-center XYZ in robot/world frame.
            end_position: (3,) final pinch-center XYZ in robot/world frame.
            quaternion_wxyz: Fixed pinch-center orientation in WXYZ order. If None,
                the G1 fixed vertical front-palm orientation is used.
            num_waypoints: Number of Cartesian IK waypoints including the final point.
                The default 50 points keeps the pregrasp-to-grasp segment close
                to a Cartesian line instead of relying on low-level joint-space
                interpolation.
            dt: Optional streaming publish period. If None, the low-level G1 env
                uses its configured action publish period.
            max_joint_step: Optional maximum absolute joint delta per streamed lowcmd
                command, in radians. If None, the low-level streamer publishes
                only these Cartesian IK waypoints and does not add joint-space
                interpolation between them.
            hold_final_seconds: Seconds to keep publishing the final grasp approach
                target before closing the hand.

        Notes:
            This method plans the whole short approach as dense Cartesian IK waypoints
            first and then streams the joint trajectory once. It does not call move_to_joints_blocking for
            each waypoint, so there is no per-waypoint hold gap. The Cartesian
            waypoint Z coordinate is locked to start_position[2], which is the
            pregrasp height in the G1 bottle policy.
        """

        start = np.asarray(start_position, dtype=np.float64).reshape(3)
        end = np.asarray(end_position, dtype=np.float64).reshape(3)
        quat = (
            FRONT_POLICY_QUATERNION_WXYZ.copy()
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )
        steps = max(1, int(num_waypoints))
        waypoints = np.linspace(start, end, steps + 1, dtype=np.float64)[1:]
        waypoints[:, 2] = start[2]
        start_str = np.array2string(start, precision=4, suppress_small=True)
        end_str = np.array2string(end, precision=4, suppress_small=True)
        self._log_step(
            "move_pinch_center_horizontal_line",
            f"Streaming horizontal pinch-center line from {start_str} to {end_str} in {steps} IK waypoints.",
        )

        trajectory = []
        pinch_offset = self._closed_pinch_center_offset()
        for waypoint in waypoints:
            palm_pos, palm_quat = palm_pose_from_pinch_center_pose(
                waypoint,
                quat,
                pinch_center_offset=pinch_offset,
            )
            palm_rot = SciRotation.from_quat([palm_quat[1], palm_quat[2], palm_quat[3], palm_quat[0]])
            palm_target = palm_pos + palm_rot.apply(self._TCP_OFFSET)
            trajectory.append(self._solve_ik(palm_target, palm_quat))

        stream = getattr(self._env, "execute_joint_trajectory_streaming", None)
        if not callable(stream):
            raise RuntimeError("The current G1 environment does not expose joint trajectory streaming.")
        stream(
            np.asarray(trajectory, dtype=np.float64),
            dt=dt,
            max_joint_step=max_joint_step,
            hold_final_seconds=hold_final_seconds,
        )
        self._log_step_update(text="Horizontal closed Dex3 pinch-center approach complete.")

    def move_to_pinch_pregrasp(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray | None = None,
        pregrasp_distance: float = 0.20,
    ) -> None:
        """Move to the front pregrasp pose for a visual grasp center.

        Args:
            position: (3,) final visual grasp center XYZ in the robot/world frame.
            quaternion_wxyz: (4,) final grasp orientation in WXYZ order. If None,
                the G1 fixed vertical front-palm orientation is used.
            pregrasp_distance: Meters to retreat along world -X before the final
                grasp. Default is 0.20 m. This mirrors the front-table pregrasp
                strategy used by the IsaacLab G1 manipulation code.
        """

        pre_pos, pre_quat = front_pregrasp_pinch_center_pose(
            position,
            quaternion_wxyz,
            pregrasp_distance=pregrasp_distance,
        )
        pre_str = np.array2string(pre_pos, precision=4, suppress_small=True)
        self._log_step("move_to_pinch_pregrasp", f"Moving to front pinch pregrasp {pre_str}.")
        self.move_pinch_center_to_pose(pre_pos, pre_quat)

    def grasp_at_pinch_center(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray | None = None,
        *,
        pregrasp_distance: float = 0.20,
        lift_dz: float = 0.10,
        trigger: float = 1.0,
        squeeze: float = 1.0,
        close_during_pregrasp: bool = True,
        approach_waypoints: int = 50,
    ) -> None:
        """Execute a complete Dex3 grasp where the closed pinch center hits the target.

        Args:
            position: (3,) visual grasp center XYZ in the robot/world frame.
            quaternion_wxyz: Ignored by the front-grasp policy. The wrist/palm
                orientation is fixed to the G1 vertical front-palm pose.
            pregrasp_distance: Front pregrasp retreat in world -X; default is 0.20 m. Set to 0 to skip.
            lift_dz: World +Z lift distance after closing the hand. Set to 0 to skip.
            trigger: Dex3 trigger value. 1 closes thumb+index for a 2D pinch.
            squeeze: Dex3 squeeze value. 1 closes thumb+middle. trigger=1 and
                squeeze=1 makes a three-finger bottle grasp.
            close_during_pregrasp: If True, briefly close to the requested
                trigger/squeeze profile while moving to pregrasp, matching the
                IsaacLab direct-wrist execution strategy; the hand opens again
                before moving to the final grasp center.
            approach_waypoints: Number of Cartesian IK waypoints for the streamed
                horizontal pregrasp-to-grasp approach. Default 50 reduces
                end-effector sag from joint-space interpolation.

        Example:
            pos, quat = sample_grasp_center_pose("plastic water bottle")
            grasp_at_pinch_center(pos, quat, trigger=1.0, squeeze=1.0)
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        del quaternion_wxyz
        quat = FRONT_POLICY_QUATERNION_WXYZ.copy()
        pre_pos: np.ndarray | None = None
        if pregrasp_distance > 0.0:
            pre_pos, _ = front_pregrasp_pinch_center_pose(
                pos,
                quat,
                pregrasp_distance=pregrasp_distance,
            )
            if close_during_pregrasp:
                self.set_gripper_trigger_squeeze(trigger=trigger, squeeze=squeeze)
            self.move_to_pinch_pregrasp(pos, quat, pregrasp_distance=pregrasp_distance)

        self.set_gripper_trigger_squeeze(trigger=0.0, squeeze=0.0)
        if pre_pos is None:
            self.move_pinch_center_to_pose(pos, quat)
        else:
            self.move_pinch_center_horizontal_line(
                pre_pos,
                pos,
                quat,
                num_waypoints=approach_waypoints,
            )
        self.set_gripper_trigger_squeeze(trigger=trigger, squeeze=squeeze)

        if lift_dz != 0.0:
            lift_pos = pos + np.asarray([0.0, 0.0, float(lift_dz)], dtype=np.float64)
            self.move_pinch_center_to_pose(lift_pos, quat)

    def move_to_joints(self, joints: Any) -> None:
        """Move the G1 right arm to a 7-DoF joint target.

        Args:
            joints: 7 joint angles in radians, ordered as shoulder pitch, shoulder roll,
                shoulder yaw, elbow, wrist roll, wrist pitch, wrist yaw.
        """

        self._env.move_to_joints_blocking(as_g1_arm_joints(joints))

    def move_to_pregrasp_side_pose(self) -> None:
        """Move the right arm to the configured side-lift pregrasp waypoint.

        This waypoint is configured on the G1 low-level environment and is intended
        to lift the hand away from the table before Cartesian grasp motion.
        """

        move_once = getattr(self._env, "move_to_pregrasp_side_pose_once", None)
        if not callable(move_once):
            raise RuntimeError("The current G1 environment does not expose a pregrasp side pose.")
        moved = move_once(force=True)
        if not moved:
            raise RuntimeError("No pregrasp side pose is configured for this G1 environment.")

    def _move_to_pregrasp_side_pose_if_configured(self) -> None:
        move_once = getattr(self._env, "move_to_pregrasp_side_pose_once", None)
        if callable(move_once):
            move_once()

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Move the G1 right hand palm link to a Cartesian pose through PyRoKi IK.

        Args:
            position: (3,) target XYZ in meters, in the PyRoKi/world frame.
            quaternion_wxyz: (4,) target unit quaternion in WXYZ order.
            z_approach: Optional approach offset along world +Z before final motion.
        """

        pos_str = np.array2string(np.asarray(position), precision=4)
        self._log_step("goto_pose", f"Moving G1 right hand palm to position {pos_str}.")

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        self._move_to_pregrasp_side_pose_if_configured()

        if z_approach != 0.0:
            approach_pos = offset_pos + np.array([0.0, 0.0, z_approach], dtype=np.float64)
            self.move_to_joints(self._solve_ik(approach_pos, quat_wxyz))

        self.move_to_joints(self._solve_ik(offset_pos, quat_wxyz))
        self._log_step_update(text="G1 right hand palm motion complete.")

    def open_gripper(self) -> None:
        """Open the right Dex3 hand."""

        self._log_step("open_gripper", "Opening G1 right Dex3 hand.")
        if hasattr(self._env, "open_dex3_hand"):
            self._env.open_dex3_hand()
        elif hasattr(self._env, "_set_gripper"):
            self._env._set_gripper(1.0)
        self._log_step_update(text="G1 right Dex3 hand open.")

    def close_gripper(self) -> None:
        """Close the right Dex3 hand with thumb, index, and middle fingers."""

        self._log_step("close_gripper", "Closing G1 right Dex3 hand with three fingers.")
        if hasattr(self._env, "close_dex3_hand"):
            self._env.close_dex3_hand()
        elif hasattr(self._env, "_set_gripper"):
            self._env._set_gripper(0.0)
        self._log_step_update(text="G1 right Dex3 hand closed.")

    def close_index_pinch(self) -> None:
        """Close the right Dex3 thumb toward the index finger for a 2D pinch."""

        self._log_step("close_index_pinch", "Closing G1 right Dex3 thumb-index pinch.")
        if hasattr(self._env, "close_dex3_index_pinch"):
            self._env.close_dex3_index_pinch()
        elif hasattr(self._env, "_set_gripper"):
            self._env._set_gripper(0.0)
        self._log_step_update(text="G1 right Dex3 thumb-index pinch closed.")

    def set_gripper_trigger_squeeze(self, trigger: float, squeeze: float) -> None:
        """Control the right Dex3 hand with IsaacLab-style trigger/squeeze inputs.

        Args:
            trigger: 0 opens, 1 closes thumb+index for a two-finger pinch.
            squeeze: 0 opens, 1 closes thumb+middle. Using trigger=1 and
                squeeze=1 closes thumb, index, and middle for a three-finger grasp.

        Notes:
            This is the preferred hand API for generated code when it needs to
            choose between index pinch, middle pinch, or full three-finger grasp.
        """

        target = dex3_grasp_joints(trigger=trigger, squeeze=squeeze)
        target_str = np.array2string(target, precision=4, suppress_small=True)
        self._log_step(
            "set_gripper_trigger_squeeze",
            f"Moving G1 right Dex3 hand to trigger={float(trigger):.3f}, squeeze={float(squeeze):.3f}.",
        )
        print(f"[g1-real] set_gripper_trigger_squeeze target={target_str}")
        self.move_hand_joints(target)
        self._log_step_update(text="G1 right Dex3 trigger/squeeze command complete.")

    def close_middle_pinch(self) -> None:
        """Close the right Dex3 thumb toward the middle finger using squeeze only."""

        self.set_gripper_trigger_squeeze(trigger=0.0, squeeze=1.0)

    def move_hand_joints(self, joints: Any) -> None:
        """Move the right Dex3 hand to 7 semantic joint targets.

        Args:
            joints: Right hand joints in order thumb_0, thumb_1, thumb_2,
                index_0, index_1, middle_0, middle_1.
        """

        if not hasattr(self._env, "move_hand_to_joints_blocking"):
            raise RuntimeError("The current G1 environment does not expose Dex3 hand joint control.")
        self._env.move_hand_to_joints_blocking(joints)
