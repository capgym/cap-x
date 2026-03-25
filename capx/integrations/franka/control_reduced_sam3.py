import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    close_gripper_arm1 as _close_gripper_arm1,
    extract_arm_joints,
    get_oriented_bounding_box_from_3d_points as _get_obb,
    open_gripper as _open_gripper,
    open_gripper_arm1 as _open_gripper_arm1,
    quat_wxyz_to_xyzw,
    solve_ik_with_convergence,
    transform_pose_arm0_to_arm1,
)
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.molmo import init_molmo
from capx.integrations.motion.pyroki import init_pyroki, init_pyroki_trajopt

# from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

# from capx.integrations.vision.sam2 import init_sam2
from capx.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb
from capx.utils.visualization_utils import (
    draw_molmo_point,
    draw_oriented_bounding_box,
    overlay_segmentation_masks,
    render_cylinder_axis,
)


# ------------------------------- Control API ------------------------------
class FrankaControlApiReducedSam3(ApiBase):
    """
    Robot control helpers for Franka.
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] | None = [0.0, 0.0, -0.107],
        is_spill_wipe: bool = False,
        is_peg_assembly: bool = False,
        is_handover: bool = False,
        bimanual: bool = False,
        real: bool = False,
    ) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        print("init franka control api")
        self.grasp_net_plan_fn = (
            init_contact_graspnet()
        )  # TODO: refactor this and use registered api instead
        print("init grasp net plan fn")
        self.sam3_seg_fn = init_sam3()
        self.sam3_point_prompt_fn = init_sam3_point_prompt()
        print("init sam3 seg fn")
        self.molmo_point_fn = init_molmo()
        print("init molmo point fn")

        self.ik_solve_fn = init_pyroki()
        self.trajopt_plan_fn = init_pyroki_trajopt()
        self.cfg = None
        self.is_spill_wipe = is_spill_wipe
        self.is_peg_assembly = is_peg_assembly
        self.is_handover = is_handover
        self.bimanual = bimanual
        self.real = real
    def functions(self) -> dict[str, Any]:
        fns = {
            "get_observation": self.get_observation,
            "segment_sam3_text_prompt": self.segment_sam3_text_prompt,
            "segment_sam3_point_prompt": self.segment_sam3_point_prompt,
            "point_prompt_molmo": self.point_prompt_molmo,
        }
        # if not self.is_spill_wipe:
        #     if not self.is_peg_assembly:
        fns["plan_grasp"] = self.plan_grasp
        fns["get_oriented_bounding_box_from_3d_points"] = (
            self.get_oriented_bounding_box_from_3d_points
        )
        # if not self.bimanual:
        #     fns["open_gripper"] = self.open_gripper
        #     fns["close_gripper"] = self.close_gripper
        if self.bimanual:
            fns["solve_ik_arm0"] = self.solve_ik_arm0
            fns["solve_ik_arm1"] = self.solve_ik_arm1
            fns["move_to_joints_both"] = self.move_to_joints_both
            fns["move_to_joints_arm0"] = self.move_to_joints_arm0
            fns["move_to_joints_arm1"] = self.move_to_joints_arm1
            fns["open_gripper_arm0"] = self.open_gripper_arm0
            fns["close_gripper_arm0"] = self.close_gripper_arm0
            fns["open_gripper_arm1"] = self.open_gripper_arm1
            fns["close_gripper_arm1"] = self.close_gripper_arm1
        else:
            fns["solve_ik"] = self.solve_ik
            # fns["traj_plan"] = self.traj_plan
            # fns["move_along_trajectory"] = self.move_along_trajectory
            fns["move_to_joints"] = self.move_to_joints
            fns["open_gripper"] = self.open_gripper
            fns["close_gripper"] = self.close_gripper

        return fns

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["robot0_robotview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_robotview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
                - ["robot0_robotview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_robotview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
        """
        self._log_step("get_observation", "Capturing camera observation …")
        obs = self._env.get_observation()
        obs["robot0_robotview"]["images"]["depth"] = obs["robot0_robotview"]["images"]["depth"].squeeze(-1)
        self._log_step_update(images=obs["robot0_robotview"]["images"]["rgb"])
        return obs

    # - ["robot_joint_pos"]: Current joint positions of the robot (including gripper as the last element) as a numpy array of shape (8,), dtype float64.
    # - ["robot_cartesian_pose_wxyz_xyz"]: Current Cartesian pose (quaternion wxyz, then position xyz) of the robot (including gripper as the last element) as a numpy array of shape (8,), dtype float64.

    # --------------------------------------------------------------------- #
    # Vision models: Sam3 segmentation
    # --------------------------------------------------------------------- #

    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_coords:
                (x, y) pixel coordinates of the point prompt.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        self._log_step("SAM3 Point Segmentation", f"Running SAM3 point-prompt at ({point_coords[0]}, {point_coords[1]}) …", images=rgb)
        results = self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)
        masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
        if masks:
            vis = overlay_segmentation_masks(rgb, masks)
            if hasattr(self._env, "set_sam3_mask"):
                self._env.set_sam3_mask(vis)
            self._log_step_update(text=f"Returned {len(results)} mask(s)", images=vis)
        else:
            self._log_step_update(text="No mask beyond threshold.")
        return results

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image conditioned on a text prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Text prompt for SAM3 segmentation.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "box": list [x1, y1, x2, y2] in pixel coordinates.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> masks = segment_sam3(rgb, text_prompt="red mug")
        """
        self._log_step("SAM3 Text Segmentation", f"Running SAM3 text-prompt: '{text_prompt}' …", images=rgb)
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
        if masks:
            best_score = max(r.get("score", 0) for r in results)
            vis = overlay_segmentation_masks(rgb, masks)
            if hasattr(self._env, "set_sam3_mask"):
                self._env.set_sam3_mask(vis)
            self._log_step_update(text=f"Returned {len(results)} mask(s), best score: {best_score:.3f}", images=vis)
        else:
            self._log_step_update(text="No masks returned.")
        return results

    # --------------------------------------------------------------------- #
    # Molmo point prompt
    # --------------------------------------------------------------------- #
    def point_prompt_molmo(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Use Molmo to point to a coordinate in the image based on a text prompt.

        Args:
            image: np.ndarray: The RGB image to process. Shape: (H, W, 3), dtype uint8.
            text_prompt: str: The text prompt to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """
        self._log_step("Molmo Point Prompt", f"Querying Molmo for '{text_prompt}' …", images=image)
        result = self.molmo_point_fn(Image.fromarray(image), objects=[text_prompt])
        if None not in result.values():
            molmo_image = draw_molmo_point(image, result)
            if hasattr(self._env, "set_molmo_image"):
                self._env.set_molmo_image(molmo_image)
            self._log_step_update(text=f"Result: {result}", images=molmo_image)
        else:
            self._log_step_update(text="No point found.")
        return result

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        Args:
            points: np.ndarray: The 3D points to get the oriented bounding box from.
                Shape: (N, 3), dtype float64.

        Returns:
            dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
                - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
                - "extent": np.ndarray: The extent of the oriented bounding box.
                - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.

        Example:
            >>> points = np.random.randn((100, 3))
            >>> obb = get_oriented_bounding_box_from_3d_points(points)
        """
        return _get_obb(points)

    # --------------------------------------------------------------------- #
    # Grasp planner (Contact-GraspNet)
    # --------------------------------------------------------------------- #
    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet for a single instance.

        This is a thin wrapper around the Contact-GraspNet planner. It does not
        apply any camera/world transforms or TCP offsets: the caller is
        responsible for transforming the resulting grasp poses into the desired
        frame and applying TCP offsets if necessary.

        Args:
            depth:
                Depth image in meters.
                Shape: (H, W), dtype float32/float64.
            intrinsics:
                Camera intrinsic matrix.
                Shape: (3, 3), dtype float64.
            segmentation:
                Instance segmentation map where each integer > 0 corresponds to a
                unique object instance ID.
                Shape: (H, W) or (H, W, 1), dtype int32/int64.

        Returns:
            grasp_poses:
                np.ndarray of shape (K, 4, 4), dtype float64.
                Homogeneous transforms for each candidate grasp IN THE CAMERA FRAME.
            grasp_scores:
                np.ndarray of shape (K,), dtype float64.
                Confidence score for each candidate grasp.

        Example:
            >>> cam = obs["robot0_robotview"]
            >>> rgb = cam["images"]["rgb"]
            >>> depth = cam["images"]["depth"]
            >>> sam3_results = sam3_seg_fn(rgb, text_prompt="red mug")
            >>> best = max(sam3_results, key=lambda d: d["score"])
            >>> mask = best["mask"]
            >>> K = cam["intrinsics"]
            >>> grasp_sample_tf, grasp_scores = plan_grasp(
            ...     depth=depth,
            ...     intrinsics=K,
            ...     segmentation=mask,
            ... )
            >>> best_idx = grasp_scores.argmax()
            >>> best_T = grasp_poses[best_idx]  # (4, 4)
            >>> camera_extrinsics = cam["pose_mat"]
            >>> grasp_sample_world_frame = camera_extrinsics @ best_T
        """
        self._log_step("Contact GraspNet", "Running grasp candidate planning …")
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[:, :, 0]

        self._env.grasp_sample, self._env.grasp_scores, _ = self.grasp_net_plan_fn(
            depth,
            intrinsics,
            segmentation,
            1,
            z_range=[0.2, 3.5] if self.is_handover else [0.2, 2.0],
            forward_passes=1 if self.is_handover else 3,
        )
        self._env.grasp_sample_tf = (
            vtf.SE3.from_matrix(self._env.grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        self._env._update_viser_server()
        n_candidates = len(self._env.grasp_scores)
        best_score = float(self._env.grasp_scores.max()) if n_candidates > 0 else 0.0
        self._log_step_update(text=f"{n_candidates} candidates, best score={best_score:.3f}")
        return self._env.grasp_sample_tf, self._env.grasp_scores

    # --------------------------------------------------------------------- #
    # IK / motion primitives
    # --------------------------------------------------------------------- #
    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve inverse kinematics for the panda_hand link.

        Args:
            position:
                Target position in world frame.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in world frame.
                Shape: (4,), [w, x, y, z], dtype float64.

        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles for the 7 DoF Franka arm.

        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity, wxyz
            >>> joints = solve_ik(target_pos, target_quat)
            >>> move_to_joints(joints)
        """
        pos_str = np.array2string(np.asarray(position), precision=4)
        self._log_step("IK Solver", f"Solving IK for target position {pos_str} …")
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)

        if self.real:
            quat_wxyz = (vtf.SO3(wxyz=quat_wxyz) @ vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4+np.pi/2)).wxyz

            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            )
            joints = extract_arm_joints(self.cfg)
            self._log_step_update(text=f"IK solved (real mode, 1 pass)")
            return joints
        else:
            self.cfg = solve_ik_with_convergence(
                self.ik_solve_fn, quat_wxyz, offset_pos, self.cfg
            )
            joints = extract_arm_joints(self.cfg)
            self._log_step_update(text=f"IK converged")
            return joints

    # Single arm control APIs

    def move_to_joints(self, joints: np.ndarray) -> None:
        """Move the robot to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm.
                Shape: (7,), dtype float64.

        Returns:
            None

        Example:
            >>> joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
            >>> move_to_joints(joints)
        """
        self._log_step("move_to_joints", "Moving robot to target joint configuration …")
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self._log_step_update(text="Motion complete.")

        # self._env.move_to_joints_non_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        self._log_step("open_gripper", "Opening gripper …")
        _open_gripper(self._env, steps=30)
        self._log_step_update(text="Gripper opened.")

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        self._log_step("close_gripper", "Closing gripper …")
        _close_gripper(self._env, steps=30)
        self._log_step_update(text="Gripper closed.")

        # """Plan a trajectory between two poses. This takes much longer than the IK solver (4s) but returns a trajectory of joint space waypoints which may be smoother and more continuous compared to setting joint targets directly to IK solutions.

    def traj_plan(
        self, start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray
    ) -> np.ndarray:
        """Plan a trajectory between two poses.
        Args:
            start_pose_wxyz_xyz:
                Start pose as a unit quaternion in world frame.
                Shape: (7,), dtype float64.
            end_pose_wxyz_xyz:
                End pose as a unit quaternion in world frame.
                Shape: (7,), dtype float64.

        Returns:
            waypoints:
                np.ndarray of shape (N, 7), dtype float64.
                Waypoints for the trajectory.
        """
        start_time = time.time()
        traj = self.trajopt_plan_fn(start_pose_wxyz_xyz, end_pose_wxyz_xyz)
        end_time = time.time()
        print(
            f"Trajectory planning time: {end_time - start_time} seconds for {len(traj)} waypoints"
        )
        return traj[:, :-1]

    def move_along_trajectory(self, trajectory: np.ndarray) -> None:
        """Move the robot along a trajectory of joint space waypoints.
        Args:
            trajectory:
                np.ndarray of shape (N, 7), dtype float64.
                Trajectory of joint space waypoints.
        """
        for waypoint in trajectory:
            self._env.move_to_joints_blocking(waypoint, tolerance=0.025, max_steps=15)

    # Dual arm control APIs
    def move_to_joints_both(self, joints0: np.ndarray, joints1: np.ndarray) -> None:
        """Move the arms 0 and 1 to a given joint configuration in a blocking manner simultaneously.

        Args:
            joints0:
                Target joint angles for the 7-DoF Franka arm 0.
                Shape: (7,), dtype float64.
            joints1:
                Target joint angles for the 7-DoF Franka arm 1.
                Shape: (7,), dtype float64.
        """
        self._env.move_to_joints_blocking_both(joints0, joints1)

    def move_to_joints_arm0(self, joints: np.ndarray) -> None:
        """Move the robot arm 0 to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm 0.
                Shape: (7,), dtype float64.
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def move_to_joints_arm1(self, joints: np.ndarray) -> None:
        """Move the robot arm 1 to a given joint configuration in a blocking manner.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm 1.
                Shape: (7,), dtype float64.
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)

    def open_gripper_arm0(self) -> None:
        """Open gripper fully for Arm 0 (robot0).
        Args:
            None
        Returns:
            None
        """
        _open_gripper(self._env, steps=30)

    def close_gripper_arm0(self) -> None:
        """Close gripper fully for Arm 0 (robot0).
        Args:
            None
        Returns:
            None
        """
        _close_gripper(self._env, steps=30)

    def open_gripper_arm1(self) -> None:
        """Open gripper fully for Arm 1 (robot1).
        Args:
            None
        Returns:
            None
        """
        _open_gripper_arm1(self._env, steps=30)

    def close_gripper_arm1(self) -> None:
        """Close gripper fully for Arm 1 (robot1).
        Args:
            None
        Returns:
            None
        """
        _close_gripper_arm1(self._env, steps=30)

    def solve_ik_arm0(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve inverse kinematics for the panda_hand link for Arm 0 (robot0)."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)

        self.cfg = solve_ik_with_convergence(
            self.ik_solve_fn, quat_wxyz, offset_pos, self.cfg
        )
        return extract_arm_joints(self.cfg)

    def solve_ik_arm1(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve inverse kinematics for the panda_hand link for Arm 1 (robot1)."""
        if not hasattr(self._env, "move_to_joints_blocking_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")

        pos_arm1, quat_wxyz_arm1 = transform_pose_arm0_to_arm1(
            position, quaternion_wxyz, self._env
        )
        offset_pos = apply_tcp_offset(pos_arm1, quat_wxyz_arm1, self._TCP_OFFSET)

        self.cfg = solve_ik_with_convergence(
            self.ik_solve_fn, quat_wxyz_arm1, offset_pos, self.cfg
        )
        return extract_arm_joints(self.cfg)
