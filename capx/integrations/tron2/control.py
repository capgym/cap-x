"""Visual (non-privileged) control API for TRON2 in CaP-X.

Combines:
- SAM3 text-prompt segmentation (port 8114) + depth point-cloud back-projection (`get_object_pose`)
- Contact-GraspNet 6-DoF grasp synthesis (port 8115) adapted to TRON2's TCP frame & wrist limits (`sample_grasp_pose`)
- Official LimX Tron2_controller WebSocket motion execution (MoveJ / MoveP on port 5000)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
import open3d as o3d

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.tron2.official_client import HORIZONTAL_GRASP_QUAT_WXYZ
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.sam3 import init_sam3
from capx.utils.camera_utils import obs_get_rgb

# Fixed SO(3) rotation aligning Contact-GraspNet's gripper frame
# (+Z approach, +X jaw baseline, +Y normal) to TRON2's official MoveP TCP frame
# (-Z approach, +Y jaw baseline, +X hand-back normal).
_R_CGN_TO_TRON2 = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

# 180-degree parallel-jaw symmetry flip around TRON2's approach axis (Z_tron2)
_R_FLIP_Z_180 = np.diag([-1.0, -1.0, 1.0])


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=np.float64).reshape(9))
    return q


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray | list[float]) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


class Tron2ControlApi(ApiBase):
    """Perception + Official-Controller skill API for LimX TRON2 in CaP-X."""

    def __init__(
        self,
        env: BaseEnv,
        *,
        cgn_tcp_offset_m: float = 0.105,
        min_grasp_z_base_m: float = -0.340,
        use_graspnet: bool = True,
    ) -> None:
        super().__init__(env)
        self._cgn_tcp_offset_m = float(cgn_tcp_offset_m)
        self._min_grasp_z_base_m = float(min_grasp_z_base_m)
        self._use_graspnet = bool(use_graspnet)

        self.sam3_seg_fn = init_sam3()
        self.grasp_net_plan_fn = init_contact_graspnet() if self._use_graspnet else None

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "home_pose": self.home_pose,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "get_left_ee_pose": self.get_left_ee_pose,
            "goto_pose_left": self.goto_pose_left,
            "goto_pose_right": self.goto_pose_right,
            "open_gripper_left": self.open_gripper_left,
            "close_gripper_left": self.close_gripper_left,
            "open_gripper_right": self.open_gripper_right,
            "close_gripper_right": self.close_gripper_right,
        }

    # ----------------------- Perception Helpers -----------------------

    def _get_primary_cam_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = self._env.get_observation()
        cam = obs["robot0_robotview"]
        rgb_imgs = obs_get_rgb(obs)
        rgb = cam["images"]["rgb"] if "rgb" in cam["images"] else list(rgb_imgs.values())[0]
        depth = cam["images"]["depth"]
        depth_2d = depth[:, :, 0] if depth.ndim == 3 else depth
        intrinsics = np.asarray(cam["intrinsics"], dtype=np.float64)
        pose_mat = np.asarray(cam["pose_mat"], dtype=np.float64)
        return rgb, depth_2d, intrinsics, pose_mat

    def _segment_best_mask(self, rgb: np.ndarray, object_name: str) -> tuple[np.ndarray, float]:
        results = self.sam3_seg_fn(rgb, text_prompt=object_name)
        if not results:
            raise ValueError(f"SAM3 found no detections for '{object_name}'")
        scores = [float(r.get("score", 0.0)) for r in results]
        best_idx = int(np.argmax(scores))
        return np.squeeze(np.asarray(results[best_idx]["mask"])).astype(bool), scores[best_idx]

    def _mask_to_obb_in_base(
        self,
        rgb: np.ndarray,
        depth_2d: np.ndarray,
        intrinsics: np.ndarray,
        pose_mat: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = np.isfinite(depth_2d) & (depth_2d > 0.05) & (depth_2d < 5.0) & mask
        ys, xs = np.where(valid)
        if len(xs) < 10:
            raise ValueError(f"Too few valid 3D points ({len(xs)}) inside SAM3 mask")

        z = depth_2d[ys, xs].astype(np.float64)
        x = (xs.astype(np.float64) + 0.5 - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys.astype(np.float64) + 0.5 - intrinsics[1, 2]) * z / intrinsics[1, 1]
        obj_pts_cam = np.stack([x, y, z], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pts_cam)
        if len(obj_pts_cam) >= 25:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        obb = pcd.get_oriented_bounding_box()
        t_cam_obb = np.eye(4, dtype=np.float64)
        t_cam_obb[:3, :3] = np.asarray(obb.R, dtype=np.float64)
        t_cam_obb[:3, 3] = np.asarray(obb.center, dtype=np.float64)
        t_base_obb = pose_mat @ t_cam_obb

        pos_base = t_base_obb[:3, 3].copy()
        quat_wxyz = _rotmat_to_quat_wxyz(t_base_obb[:3, :3])
        extent = np.asarray(obb.extent, dtype=np.float64)
        return pos_base, quat_wxyz, extent

    # ----------------------- Public Vision APIs -----------------------

    def get_object_pose(
        self,
        object_name: str,
        return_bbox_extent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get an object's 3D position and bounding box in base_Link frame via SAM3 + depth.

        The returned quaternion comes from the single-view object OBB and is NOT a gripper grasp
        orientation. Use this primarily for object/placement positions; use `sample_grasp_pose`
        when computing a gripper grasp orientation.

        Args:
            object_name: Natural-language description of the object (e.g., "red cube").
            return_bbox_extent: Whether to return the 3D OBB full side lengths (meters).

        Returns:
            Tuple of (position (3,), quaternion_wxyz (4,), bbox_extent (3,) or None).
        """
        t0 = time.time()
        rgb, depth_2d, intrinsics, pose_mat = self._get_primary_cam_obs()
        mask, score = self._segment_best_mask(rgb, object_name)
        pos_base, quat_wxyz, extent = self._mask_to_obb_in_base(rgb, depth_2d, intrinsics, pose_mat, mask)
        print(
            f"[TRON2 get_object_pose] '{object_name}' score={score:.3f}, "
            f"pos_base={np.round(pos_base, 4)}, extent={np.round(extent, 4)} ({time.time()-t0:.2f}s)"
        )
        return pos_base, quat_wxyz, (extent if return_bbox_extent else None)

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Plan a TRON2-reachable 6-DoF gripper grasp pose (TCP in base_Link frame) via SAM3 + Contact-GraspNet.

        Args:
            object_name: Natural-language description of the target object (e.g., "red cube").

        Returns:
            Tuple of (position (3,), quaternion_wxyz (4,)) for `goto_pose_left` / `goto_pose_right`.
        """
        t0 = time.time()
        rgb, depth_2d, intrinsics, pose_mat = self._get_primary_cam_obs()
        mask, sam_score = self._segment_best_mask(rgb, object_name)

        if self.grasp_net_plan_fn is not None:
            try:
                grasps, scores, _ = self.grasp_net_plan_fn(
                    depth_2d,
                    intrinsics,
                    mask.astype(np.uint8),
                    1,
                )
                selected = self._select_tron2_grasp(grasps, scores, pose_mat)
                if selected is not None:
                    pos_base, quat_wxyz, cgn_score, approach_x = selected
                    print(
                        f"[TRON2 sample_grasp_pose] '{object_name}' Contact-GraspNet OK: "
                        f"sam={sam_score:.3f}, cgn={cgn_score:.3f}, approach_x={approach_x:.2f}, "
                        f"pos={np.round(pos_base, 4)}, quat={np.round(quat_wxyz, 3)} ({time.time()-t0:.2f}s)"
                    )
                    return pos_base, quat_wxyz
                print(
                    f"[TRON2 sample_grasp_pose] '{object_name}': Contact-GraspNet returned "
                    f"{len(scores)} grasps but none within TRON2 wrist limits; falling back to horizontal grasp."
                )
            except Exception as exc:
                print(
                    f"[TRON2 sample_grasp_pose] Contact-GraspNet failed ({exc}); "
                    f"falling back to SAM3 OBB + horizontal grasp."
                )

        # Fallback: SAM3 + depth OBB center with TRON2 horizontal forward grasp orientation
        pos_base, _, _ = self._mask_to_obb_in_base(rgb, depth_2d, intrinsics, pose_mat, mask)
        pos_base[2] = max(float(pos_base[2]), self._min_grasp_z_base_m)
        quat_wxyz = np.asarray(HORIZONTAL_GRASP_QUAT_WXYZ, dtype=np.float64)
        print(
            f"[TRON2 sample_grasp_pose] '{object_name}' fallback horizontal grasp: "
            f"pos={np.round(pos_base, 4)}, quat={np.round(quat_wxyz, 3)} ({time.time()-t0:.2f}s)"
        )
        return pos_base, quat_wxyz

    def _select_tron2_grasp(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        pose_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """Convert Contact-GraspNet proposals into TRON2 TCP frame and filter for wrist reachability."""
        if grasps is None or len(grasps) == 0 or scores is None or len(scores) == 0:
            return None

        t_tcp_shift = np.eye(4, dtype=np.float64)
        t_tcp_shift[2, 3] = self._cgn_tcp_offset_m  # move from CGN base along +Z_cgn to fingertip TCP

        best_candidate: tuple[float, np.ndarray, np.ndarray, float, float] | None = None
        for g_mat, raw_score in zip(grasps, scores):
            g_cam = np.asarray(g_mat, dtype=np.float64)
            if g_cam.shape != (4, 4):
                continue
            t_base_cgn = pose_mat @ (g_cam @ t_tcp_shift)
            pos_base = t_base_cgn[:3, 3].copy()
            r_tron2 = t_base_cgn[:3, :3] @ _R_CGN_TO_TRON2

            # Parallel-jaw 180-deg symmetry: choose orientation with hand back (+X_tron2) facing up (+Z_base)
            # so wrist_roll stays near 0 instead of flipping upside down past +/-45 deg limit.
            if r_tron2[2, 0] < 0.0:
                r_tron2 = r_tron2 @ _R_FLIP_Z_180

            # Approach vector pointing out of fingertips is -Z_tron2
            approach = -r_tron2[:, 2]
            # Filter for TRON2 dual-arm wrist comfort zone:
            # - generally pointing forward toward the table (+X_base >= 0.45)
            # - not approaching from underneath the table (approach_z <= 0.25)
            # - not pure vertical top-down beyond wrist_pitch limits (approach_z >= -0.75)
            # - hand back reasonably upright (r_tron2[2, 0] >= 0.55)
            if approach[0] < 0.45 or approach[2] > 0.25 or approach[2] < -0.75 or r_tron2[2, 0] < 0.55:
                continue

            # Keep fingertips safely clear of the tabletop
            pos_base[2] = max(float(pos_base[2]), self._min_grasp_z_base_m)

            # Combined ranking: CGN quality score + small preference for clean forward horizontal approach
            rank = float(raw_score) + 0.15 * float(approach[0])
            quat_wxyz = _rotmat_to_quat_wxyz(r_tron2)
            if best_candidate is None or rank > best_candidate[0]:
                best_candidate = (rank, pos_base, quat_wxyz, float(raw_score), float(approach[0]))

        if best_candidate is None:
            return None
        _, pos_base, quat_wxyz, cgn_score, approach_x = best_candidate
        return pos_base, quat_wxyz, cgn_score, approach_x

    # ----------------------- Public Motion APIs (Official MoveJ / MoveP) -----------------------

    def home_pose(self) -> None:
        """Move both arms to the safe raised ready pose via official MoveJ and open both grippers."""
        self._env.move_home()

    def get_left_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the left gripper fingertip (TCP) pose in base_Link frame.

        Returns:
            Tuple of (position (3,), quaternion_wxyz (4,)).
        """
        pose = self._env.client.get_move_pose()
        return pose.left_pos.copy(), pose.left_quat.copy()

    def goto_pose_left(
        self,
        pos: list[float] | np.ndarray,
        quat_wxyz: list[float] | np.ndarray | None = None,
        approach_offset: float = 0.0,
        duration_sec: float = 2.0,
    ) -> None:
        """Move the left gripper fingertip (TCP) to a target pose in base_Link frame via official MoveP.

        Args:
            pos: Target [x, y, z] position in meters relative to base_Link.
            quat_wxyz: Target orientation [w, x, y, z]. Defaults to horizontal forward grasp.
            approach_offset: Optional pre-grasp backoff distance (meters, e.g. 0.10) along the
                gripper's approach axis before sliding straight to `pos`.
            duration_sec: Planned motion duration in seconds for each segment.
        """
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = np.asarray(HORIZONTAL_GRASP_QUAT_WXYZ if quat_wxyz is None else quat_wxyz, dtype=np.float64)
        if approach_offset > 0.0:
            # Fingertips point along -Z_tron2, so +Z_tron2 backs off cleanly along the approach axis
            rot = _quat_wxyz_to_rotmat(quat)
            pre_pos = target_pos + rot @ np.array([0.0, 0.0, float(approach_offset)], dtype=np.float64)
            self._env.move_arms_pose(left_pos=pre_pos, left_quat_wxyz=quat, duration_sec=duration_sec)
        self._env.move_arms_pose(left_pos=target_pos, left_quat_wxyz=quat, duration_sec=duration_sec)

    def goto_pose_right(
        self,
        pos: list[float] | np.ndarray,
        quat_wxyz: list[float] | np.ndarray | None = None,
        approach_offset: float = 0.0,
        duration_sec: float = 2.0,
    ) -> None:
        """Move the right gripper fingertip (TCP) to a target pose in base_Link frame via official MoveP.

        Args:
            pos: Target [x, y, z] position in meters relative to base_Link.
            quat_wxyz: Target orientation [w, x, y, z]. Defaults to horizontal forward grasp.
            approach_offset: Optional pre-grasp backoff distance (meters) along the approach axis.
            duration_sec: Planned motion duration in seconds for each segment.
        """
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = np.asarray(HORIZONTAL_GRASP_QUAT_WXYZ if quat_wxyz is None else quat_wxyz, dtype=np.float64)
        if approach_offset > 0.0:
            rot = _quat_wxyz_to_rotmat(quat)
            pre_pos = target_pos + rot @ np.array([0.0, 0.0, float(approach_offset)], dtype=np.float64)
            self._env.move_arms_pose(right_pos=pre_pos, right_quat_wxyz=quat, duration_sec=duration_sec)
        self._env.move_arms_pose(right_pos=target_pos, right_quat_wxyz=quat, duration_sec=duration_sec)

    def open_gripper_left(self) -> None:
        """Open the left parallel gripper fully (~7.5 cm)."""
        self._env.set_gripper("left", 1.0)

    def close_gripper_left(self) -> None:
        """Close the left parallel gripper smoothly to clamp an object."""
        self._env.set_gripper("left", 0.0)

    def open_gripper_right(self) -> None:
        """Open the right parallel gripper fully (~7.5 cm)."""
        self._env.set_gripper("right", 1.0)

    def close_gripper_right(self) -> None:
        """Close the right parallel gripper smoothly to clamp an object."""
        self._env.set_gripper("right", 0.0)
