"""Strict visual left-arm grasping with poses proposed by Contact-GraspNet."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.graspnet import init_contact_graspnet, init_contact_graspnet_point_clouds
from capx.integrations.vision.sam3 import init_sam3
from capx.utils.camera_utils import obs_get_rgb

from .grasp_feasibility import cgn_waypoints, iter_feasible_grasps, plan_checked_left_lift
from .grasp_quality import candidate_tcp_positions, rank_grasps

# Distance from the CGN gripper base to its predicted contact midpoint
# (ContactGraspNet.build_6d_grasp, for this checkpoint's center_to_tip=0).
_CGN_DEPTH_M = 0.1034
_APPROACH_M = 0.11
_LIFT_M = 0.16


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


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=np.float64).reshape(9))
    return q


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray | list[float]) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def _approach_waypoints(target: np.ndarray, quat: np.ndarray, offset: float) -> tuple[np.ndarray, ...]:
    if offset <= 0.0:
        return (target,)
    # The fingers point along -Z in the TCP frame. Move above the backed-off
    # point first so lateral motion from home cannot sweep through the cube.
    pre_grasp = target + _quat_wxyz_to_rotmat(quat)[:, 2] * offset
    above_pre_grasp = pre_grasp + np.array([0.0, 0.0, 0.10])
    return above_pre_grasp, pre_grasp, target


class Tron2ControlCgnOnlyApi(ApiBase):
    """Execute left-arm poses selected from CGN with the tabletop lift trajectory."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self.sam3_seg_fn = init_sam3()
        self.grasp_net_plan_fn = init_contact_graspnet()
        self._plan_clouds = init_contact_graspnet_point_clouds()
        self._pending = None
        self._approach_reached = False

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

    def _masked_points_cam(
        self,
        depth_2d: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        valid = np.isfinite(depth_2d) & (depth_2d > 0.05) & (depth_2d < 5.0) & mask
        ys, xs = np.where(valid)
        if len(xs) < 10:
            raise ValueError(f"Too few valid 3D points ({len(xs)}) inside SAM3 mask")

        z = depth_2d[ys, xs].astype(np.float64)
        x = (xs.astype(np.float64) + 0.5 - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys.astype(np.float64) + 0.5 - intrinsics[1, 2]) * z / intrinsics[1, 1]
        return np.stack([x, y, z], axis=-1)

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
        points_cam = self._masked_points_cam(depth_2d, intrinsics, mask)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_cam))
        if len(points_cam) >= 25:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb = pcd.get_oriented_bounding_box()
        pos_base = pose_mat[:3, :3] @ obb.center + pose_mat[:3, 3]
        quat_wxyz = _rotmat_to_quat_wxyz(pose_mat[:3, :3] @ obb.R)
        extent = np.asarray(obb.extent, dtype=np.float64)
        print(
            f"[TRON2 get_object_pose] '{object_name}' score={score:.3f}, "
            f"pos_base={np.round(pos_base, 4)}, extent={np.round(extent, 4)} ({time.time()-t0:.2f}s)"
        )
        return pos_base, quat_wxyz, (extent if return_bbox_extent else None)

    def _proposals(self, cameras, object_points):
        # Keep observed surfaces and the surrounding tabletop from both RGB-D
        # images. Each virtual camera only changes coordinates for inference.
        center = np.percentile(object_points, [2, 98], axis=0).mean(axis=0)
        neighborhood = []
        for depth, k, pose, mask in cameras.values():
            pts = self._masked_points_cam(depth, k, np.ones_like(mask))
            pts = pts @ pose[:3, :3].T + pose[:3, 3]
            neighborhood.append(pts[np.all(np.abs(pts - center) < 0.19, axis=1)])
        scene = np.concatenate(neighborhood)
        for elevation, yaw in ((18, 35), (18, 0), (8, 22), (8, -22), (18, -35)):
            e, y = np.radians([elevation, yaw])
            forward = np.array([np.cos(e) * np.cos(y), np.cos(e) * np.sin(y), -np.sin(e)])
            right = np.array([np.sin(y), -np.cos(y), 0.0])
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
            pose[:3, 3] = center - 0.55 * forward
            grasps, scores, _ = self._plan_clouds(
                ((scene - pose[:3, 3]) @ pose[:3, :3]).astype(np.float32),
                ((object_points - pose[:3, 3]) @ pose[:3, :3]).astype(np.float32),
                forward_passes=2, max_retries=1,
            )
            yield f"fused_e{elevation}_yaw{yaw}", pose, grasps, scores
        for name, angle in (("chest_d435", 70), ("robot0_robotview", 100), ("chest_d435", 120)):
            depth, k, pose, mask = cameras[name]
            grasps, scores, _ = self.grasp_net_plan_fn(
                depth, k, mask.astype(np.uint8), 1, num_viewpoints=24, max_angle_deg=angle,
            )
            yield f"{name}_cone{angle}", pose, grasps, scores

    def _live_candidate_batches(self, object_name):
        cameras = {}
        object_clouds = []
        for name in ("robot0_robotview", "chest_d435"):
            cam = self._env._get_camera_obs(name)
            depth = cam["images"]["depth"][:, :, 0]
            intrinsics, camera_pose = cam["intrinsics"], cam["pose_mat"]
            mask, score = self._segment_best_mask(cam["images"]["rgb"], object_name)
            cameras[name] = depth, intrinsics, camera_pose, mask
            points = self._masked_points_cam(depth, intrinsics, mask)
            object_clouds.append(points @ camera_pose[:3, :3].T + camera_pose[:3, 3])
            print(f"[CGN-only] {name} SAM3 score={score:.3f} pixels={mask.sum()}")

        depth, intrinsics, camera_pose, mask = cameras["robot0_robotview"]
        ring = binary_dilation(mask, iterations=10) & ~mask
        support = self._masked_points_cam(depth, intrinsics, ring)
        table_z = np.median((support @ camera_pose[:3, :3].T + camera_pose[:3, 3])[:, 2])
        object_points = np.concatenate(object_clouds)

        for name, camera_pose, grasps, scores in self._proposals(cameras, object_points):
            yield name, camera_pose, grasps, scores, object_points, table_z

    def sample_grasp_pose(
        self, object_name: str, candidate_rank: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a CGN left-arm (position, WXYZ quaternion) from the RGB-D images.

        Pass candidate_rank=1, 2, ... to request a different feasible proposal.
        Rank 0 is the first feasible proposal. Live calls observe the scene again.
        """
        self._pending = None
        self._approach_reached = False
        rank = int(os.environ.get("CAPX_TRON2_CGN_RANK", "0")) if candidate_rank is None else candidate_rank
        if not isinstance(rank, (int, np.integer)) or rank < 0:
            raise ValueError("candidate_rank must be a nonnegative integer")

        snapshot_path = os.environ.get("CAPX_TRON2_CGN_CANDIDATES")
        snapshot = Path(snapshot_path) if snapshot_path else None
        if snapshot and snapshot.is_file():
            with np.load(snapshot, allow_pickle=False) as batch:
                if batch["object_name"].item() != object_name:
                    raise ValueError("CGN snapshot object name differs from the requested object")
                batches = [(batch["name"].item(), batch["camera_pose"], batch["grasps"],
                            batch["scores"], batch["object_points"], batch["table_z"].item())]
            print(f"[CGN-only] replaying CGN proposal batch from {snapshot}")
        else:
            batches = self._live_candidate_batches(object_name)

        remaining_rank = rank
        for name, camera_pose, grasps, scores, object_points, table_z in batches:
            raw = np.asarray(grasps).reshape(-1, 4, 4)
            scores = np.asarray(scores).reshape(-1)
            if not len(raw):
                print(f"[CGN-only] {name} proposals=0")
                continue
            base = camera_pose @ raw
            rotations = base[:, :3, :3]
            contacts = base[:, :3, 3] + _CGN_DEPTH_M * rotations[:, :, 2]
            positions = candidate_tcp_positions(contacts, rotations)
            quality = rank_grasps(contacts, rotations, scores, object_points, table_z)
            plans = iter_feasible_grasps(
                self._env, positions, rotations @ _R_CGN_TO_TRON2, quality,
                approach_offset=_APPROACH_M, lift_height=_LIFT_M,
            )
            skipped = 0
            plan = None
            for feasible in plans:
                if remaining_rank == 0:
                    plan = feasible
                    break
                remaining_rank -= 1
                skipped += 1
            print(f"[CGN-only] {name} proposals={len(raw)} "
                  f"geometry_pass={np.isfinite(quality).any(axis=1).sum()} "
                  f"candidate_rank={rank} feasible_before_selection={skipped} path_found={plan is not None}")
            if plan is None:
                continue

            if snapshot and not snapshot.is_file():
                temporary = snapshot.with_suffix(".tmp")
                with temporary.open("wb") as output:
                    np.savez_compressed(output, object_name=object_name, name=name, camera_pose=camera_pose,
                                        grasps=raw, scores=scores, object_points=object_points, table_z=table_z)
                temporary.replace(snapshot)
                print(f"[CGN-only] saved CGN proposal batch to {snapshot}")
            i = plan.index
            p = plan.position.copy()
            q = _rotmat_to_quat_wxyz(plan.rotation)
            self._pending = (p.copy(), q.copy(), plan)
            print(f"[CGN-only] selected camera={name} zero_based_index={i} score={scores[i]:.5f} "
                  f"contact_midpoint={contacts[i].tolist()} tcp={p.tolist()} quat_wxyz={q.tolist()} "
                  f"approach_base={rotations[i, :, 2].tolist()}")
            print(f"[CGN-only] raw_camera_grasp={raw[i].tolist()} camera_to_base={camera_pose.tolist()}")
            return p, q
        raise RuntimeError(f"No CGN candidate at rank {rank} passed the geometry and left-arm trajectory checks")

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

    def _verify_left_tcp(self, target: np.ndarray, quat: np.ndarray, stage: str) -> bool:
        pose = self._env.client.get_move_pose()
        distance = np.linalg.norm(pose.left_pos - target)
        angle = np.degrees(2 * np.arccos(np.clip(abs(np.dot(quat, pose.left_quat)), 0, 1)))
        reached = bool(distance <= 0.005 and angle <= 3)
        print(f"[CGN-only] {stage} measured TCP error {distance * 1000:.2f} mm / {angle:.2f} deg "
              f"reached={reached}")
        return reached

    def goto_pose_left(self, pos, quat_wxyz=None, approach_offset=0.0, duration_sec=2.0) -> bool:
        """Approach the selected CGN pose or request a checked vertical lift.

        Approach with a 0.11 m backoff, then close the gripper. To lift, keep
        the grasp X/Y and orientation, add a positive Z offset (for example,
        0.12 m or 0.16 m), and omit approach_offset. The lift path is checked
        from the current robot state. An infeasible lift returns False before
        moving, so you can try a shorter offset. A controller joint limit or
        missed waypoint also returns False. Resample after a motion failure.
        """
        if self._pending is None:
            raise RuntimeError("Call sample_grasp_pose before executing the CGN trajectory")
        grasp, quat, plan = self._pending
        target = np.asarray(pos, dtype=float).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError("The requested TCP position must be finite")
        if not np.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("Motion duration must be finite and positive")
        supplied_quat = quat if quat_wxyz is None else np.asarray(quat_wxyz, dtype=float).reshape(4)
        if not (np.allclose(supplied_quat, quat) or np.allclose(supplied_quat, -quat)):
            raise ValueError("The requested orientation differs from the selected CGN pose")
        if np.allclose(target, grasp) and np.isclose(approach_offset, _APPROACH_M):
            self._approach_reached = False
            waypoints = cgn_waypoints(grasp, plan.rotation, _APPROACH_M, _LIFT_M)
            for stage, path, waypoint in zip(("above", "pre-grasp", "contact"),
                                              plan.joint_paths[:3], waypoints[:3], strict=True):
                sent = self._env.move_left_joint_path(path, duration_sec)
                reached = self._verify_left_tcp(waypoint, quat, stage)
                if not (sent and reached):
                    print(f"[CGN-only] WARNING: {stage} was incomplete; stopped the checked approach path.",
                          file=sys.stderr)
                    return False
            self._approach_reached = True
            return True
        elif (np.isclose(approach_offset, 0) and np.allclose(target[:2], grasp[:2])
              and target[2] > grasp[2]):
            if not self._approach_reached:
                print("[CGN-only] WARNING: lift skipped because the checked approach was incomplete.",
                      file=sys.stderr)
                return False
            path = plan_checked_left_lift(self._env, target, plan.rotation)
            if path is None:
                print(f"[CGN-only] WARNING: requested lift {target[2] - grasp[2]:.3g} m has no "
                      "checked path from the current pose; try a shorter upward offset.", file=sys.stderr)
                return False
            self._pending = None
            self._approach_reached = False
            sent = self._env.move_left_joint_path(path, duration_sec)
            reached = self._verify_left_tcp(target, quat, "lift")
            if not (sent and reached):
                print("[CGN-only] WARNING: lift was incomplete; resample to plan from the current pose.",
                      file=sys.stderr)
            return sent and reached
        else:
            raise ValueError(
                f"Requested offset from the selected CGN grasp: {(target - grasp).tolist()} m. "
                f"For the checked approach use goto_pose_left(grasp_pos, grasp_quat, approach_offset={_APPROACH_M}). "
                "For a checked lift add a positive Z offset to grasp_pos and keep grasp_quat."
            )

    def goto_pose_right(
        self,
        pos: list[float] | np.ndarray,
        quat_wxyz: list[float] | np.ndarray | None = None,
        approach_offset: float = 0.0,
        duration_sec: float = 2.0,
    ) -> bool:
        """Move the right gripper fingertip (TCP) to a target pose in base_Link frame via official MoveP.

        Args:
            pos: Target [x, y, z] position in meters relative to base_Link.
            quat_wxyz: Target orientation [w, x, y, z]. Defaults to the measured TCP orientation.
            approach_offset: Optional pre-grasp backoff distance (meters) along the approach axis.
            duration_sec: Planned motion duration in seconds for each segment.
        """
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        if quat_wxyz is None:
            quat_wxyz = self._env.client.get_move_pose().right_quat
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        for waypoint in _approach_waypoints(target_pos, quat, approach_offset):
            result = self._env.move_arms_pose(
                right_pos=waypoint, right_quat_wxyz=quat, duration_sec=duration_sec,
                report_joint_limit=True,
            )
            if result.get("result") != "success":
                return False
        return True

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
