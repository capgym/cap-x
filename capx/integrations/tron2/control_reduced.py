"""Perception, kinematics and joint-motion primitives for TRON2."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import get_oriented_bounding_box_from_3d_points
from capx.integrations.franka.control_reduced_exampleless import _strip_examples
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt

from .control_cgn import (
    _CGN_DEPTH_M,
    _JAW_SYMMETRY,
    _R_CGN_TO_TRON2,
    _rotmat_to_quat_wxyz,
)
from .grasp_feasibility import _LeftArmIK, _MAX_ORIENTATION_ERROR_RAD, _MAX_POSITION_ERROR_M
from .grasp_quality import candidate_tcp_positions
from .official_client import PROTOCOL_JOINT_NAMES, Tron2MotionRejection


class Tron2ControlReducedApi(ApiBase):
    """Expose the individual operations needed to build a TRON2 program."""

    def __init__(self, env):
        super().__init__(env)
        self._segment_text = init_sam3()
        self._segment_point = init_sam3_point_prompt()
        self._plan_grasps = init_contact_graspnet()

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_observation": self.get_observation,
            "segment_sam3_text_prompt": self.segment_sam3_text_prompt,
            "segment_sam3_point_prompt": self.segment_sam3_point_prompt,
            "get_oriented_bounding_box_from_3d_points": get_oriented_bounding_box_from_3d_points,
            "plan_grasp": self.plan_grasp,
            "grasp_to_tcp": self.grasp_to_tcp,
            "get_left_ee_pose": self.get_left_ee_pose,
            "solve_ik": self.solve_ik,
            "move_to_joints": self.move_to_joints,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_observation(self, camera_name: str = "robot0_robotview") -> dict[str, Any]:
        """Capture a current RGB-D image and measured motor positions.

        Args:
            camera_name: Use "robot0_robotview" or "chest_d435".

        Returns:
            The camera entry under camera_name includes images.rgb (H, W, 3),
            images.depth (H, W) in meters, intrinsics (3, 3), and pose_mat (4, 4).
            pose_mat maps optical camera coordinates (+X right, +Y down, +Z
            forward) to base_Link. robot_joint_pos is ordered left 7, right 7,
            and head 2 in radians.

        Example:
            >>> obs = get_observation()
            >>> camera = obs["robot0_robotview"]
            >>> rgb = camera["images"]["rgb"]
        """
        if camera_name not in ("robot0_robotview", "chest_d435"):
            raise ValueError(f"Unknown TRON2 evaluation camera {camera_name!r}")
        data = self._env._observation_data()
        camera = self._env._get_camera_obs(camera_name, data=data)
        camera["images"]["depth"] = camera["images"]["depth"][..., 0]
        return {camera_name: camera, "robot_joint_pos": data.qpos[self._env._joint_qpos]}

    def segment_sam3_text_prompt(self, rgb: np.ndarray, text_prompt: str) -> list[dict[str, Any]]:
        """Return SAM3 masks and confidence scores for a text prompt.

        Args:
            rgb: (H, W, 3) uint8 RGB image.
            text_prompt: Object description in the image.

        Returns:
            SAM3 result dictionaries with a boolean mask (H, W) and score.

        Example:
            >>> detections = segment_sam3_text_prompt(rgb, "blue mug")
        """
        return self._segment_text(rgb, text_prompt=text_prompt)

    def segment_sam3_point_prompt(
        self, rgb: np.ndarray, point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Return SAM3 masks and scores for an (x, y) pixel in an RGB image."""
        return self._segment_point(rgb, point_coords)

    def plan_grasp(
        self, depth: np.ndarray, intrinsics: np.ndarray, segmentation: np.ndarray, *,
        num_viewpoints: int | None = None, max_angle_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ask Contact-GraspNet for candidates for the supplied instance mask.

        Args:
            depth: (H, W) or (H, W, 1) depth image in meters.
            intrinsics: (3, 3) intrinsics for the supplied depth image.
            segmentation: (H, W) or (H, W, 1) boolean mask for one instance.
            num_viewpoints: Optional synthetic-view count for Contact-GraspNet.
            max_angle_deg: Optional sweep angle for synthetic views.

        Returns:
            Raw grasp transforms (N, 4, 4) in the optical camera frame and
            corresponding confidence scores (N,). Empty arrays mean no proposals.
            Use grasp_to_tcp for calibrated conversion of a selected proposal.

        Example:
            >>> candidates, scores = plan_grasp(camera["images"]["depth"], camera["intrinsics"], mask)
        """
        depth = np.asarray(depth, dtype=np.float32)
        mask = np.asarray(segmentation, dtype=bool)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if depth.ndim != 2 or depth.shape != mask.shape:
            raise ValueError("Depth and instance mask must have the same (H, W) shape")
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("Intrinsics must be a finite (3, 3) matrix")
        if not mask.any():
            return np.empty((0, 4, 4)), np.empty(0)
        grasps, scores, _ = self._plan_grasps(
            depth, intrinsics, mask.astype(np.uint8), 1,
            forward_passes=3, num_viewpoints=num_viewpoints, max_angle_deg=max_angle_deg,
        )
        grasps = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(grasps) != len(scores):
            raise ValueError("Contact-GraspNet returned different numbers of grasps and scores")
        print(f"[TRON2] CGN_PROPOSALS count={len(scores)}")
        return grasps, scores

    def grasp_to_tcp(
        self, grasp: np.ndarray, camera_to_base: np.ndarray, jaw_branch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert one raw CGN grasp to an official TCP pose in base_Link.

        Args:
            grasp: One optical camera-frame 4×4 transform from plan_grasp.
            camera_to_base: The corresponding camera pose_mat (4, 4).
            jaw_branch: Explicitly choose jaw orientation 0 or 1. Both preserve
                the approach direction. The branches swap the opposing jaws.

        Returns:
            TCP position (3,) in meters and WXYZ quaternion (4,). The gripper
            approaches along its local -Z, and grasps across its local Y.

        Example:
            >>> p, q = grasp_to_tcp(candidates[0], camera["pose_mat"], jaw_branch=0)
        """
        if jaw_branch not in (0, 1):
            raise ValueError("jaw_branch must be 0 or 1")
        camera = np.asarray(camera_to_base, dtype=float)
        raw = np.asarray(grasp, dtype=float)
        if camera.shape != (4, 4) or raw.shape != (4, 4):
            raise ValueError("Grasp and camera pose must be 4×4 transforms")
        base_grasp = camera @ raw
        if not np.isfinite(base_grasp).all():
            raise ValueError("Grasp and camera pose must describe a finite 4×4 transform")
        rotation = base_grasp[:3, :3]
        contact = base_grasp[:3, 3] + _CGN_DEPTH_M * rotation[:, 2]
        positions = candidate_tcp_positions(contact, rotation)[0]
        tcp_rotation = rotation @ _R_CGN_TO_TRON2
        if jaw_branch == 1:
            tcp_rotation = tcp_rotation @ _JAW_SYMMETRY
        return positions[jaw_branch], _rotmat_to_quat_wxyz(tcp_rotation)

    def get_left_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the measured left TCP position (meters) and WXYZ quaternion in base_Link."""
        pose = self._env.client.get_move_pose()
        return pose.left_pos.copy(), pose.left_quat.copy()

    def solve_ik(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, seed_joints: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Solve one target with the TRON2 left-arm kinematics.

        Args:
            position: TCP position (3,) in base_Link meters.
            quaternion_wxyz: Orientation (4,) in WXYZ order.
            seed_joints: Optional starting solution (7,) in radians. The default
                seed is the measured left-arm position.

        Returns:
            Left-arm joints (7,) in radians, or None if the pose solution misses
            the tolerance or intersects the robot.

        Example:
            >>> joints = solve_ik(p, q)
        """
        target = np.asarray(position, dtype=float).reshape(3)
        wxyz = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
        if not np.isfinite(target).all() or not np.isfinite(wxyz).all() or not np.any(wxyz):
            raise ValueError("IK needs a finite position and a finite, nonzero quaternion")
        wxyz = wxyz / np.max(np.abs(wxyz))
        rotation = Rotation.from_quat(wxyz[[1, 2, 3, 0]])
        ik = _LeftArmIK(self._env)
        seed = ik.start if seed_joints is None else np.asarray(seed_joints, dtype=float).reshape(7)
        if not np.isfinite(seed).all():
            raise ValueError("IK seed joints must be finite")
        joints, position_error, orientation_error = ik.solve(target, rotation, seed)
        if position_error > _MAX_POSITION_ERROR_M or orientation_error > _MAX_ORIENTATION_ERROR_RAD:
            print(f"[TRON2] IK_REJECTED target={target.tolist()} "
                  f"position_error_mm={1000 * position_error:.2f} "
                  f"orientation_error_deg={np.degrees(orientation_error):.2f}")
            return None
        if reason := ik.self_collision():
            print(f"[TRON2] IK_REJECTED target={target.tolist()} {reason}")
            return None
        return joints

    def move_to_joints(self, joints: np.ndarray, duration_sec: float = 2.0) -> bool:
        """Send one seven-axis left-arm target in radians with official MoveJ.

        Returns True when the measured joints reach the target within 0.015 rad.
        A controller rejection or a missed endpoint returns False. The right arm
        keeps its measured joint target and the resulting state remains observable.
        """
        target = np.asarray(joints, dtype=float).reshape(7)
        if not np.isfinite(target).all() or not np.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("MoveJ needs finite joint targets and a positive duration")
        current = self._env._read_arm_joints()[:7]
        path = np.stack((current, target))
        with self._env.motion_feedback("move-to-joints"):
            try:
                sent = self._env.move_left_joint_path(path, duration_sec)
            except Tron2MotionRejection as error:
                self._env._log_motion_rejection(error, "move-to-joints")
                self._env.record_rejected_motion()
                sent = False
            measured = self._env._read_arm_joints()[:7]
        errors = np.abs(target - measured)
        joint = int(np.argmax(errors))
        reached = sent and bool(errors[joint] <= 0.015)
        print(f"[TRON2] MOVEJ_ENDPOINT joint={PROTOCOL_JOINT_NAMES[joint]} "
              f"max_error_rad={errors[joint]:.4f} reached={reached} measured_q_rad={measured.tolist()}")
        return reached

    def open_gripper(self) -> None:
        """Open the left parallel gripper fully."""
        self._env.set_gripper("left", 1.0)

    def close_gripper(self) -> None:
        """Close the left parallel gripper."""
        self._env.set_gripper("left", 0.0)


class Tron2ControlReducedExamplelessApi(Tron2ControlReducedApi):
    """Expose the same primitives using signatures and documentation only."""

    def combined_doc(self) -> str:
        return _strip_examples(super().combined_doc())
