import math
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
from capx.integrations.franka.control_reduced import FrankaControlApiReduced
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.molmo import init_molmo
from capx.integrations.motion.pyroki import init_pyroki

# from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

# from capx.integrations.vision.sam2 import init_sam2
from capx.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb


# ------------------------------- Control API ------------------------------
class FrankaControlApiReducedSkillLibrary(FrankaControlApiReduced):
    """
    Robot control helpers for Franka.
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] | None = [0.0, 0.0, -0.107],
        bimanual: bool = False,
        is_handover: bool = False,
        real: bool = False,
    ) -> None:
        super().__init__(env, tcp_offset=tcp_offset, bimanual=bimanual, is_handover=is_handover, real=real)

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["rotation_matrix_to_quaternion"] = self.rotation_matrix_to_quaternion
        fns["decompose_transform"] = self.decompose_transform
        fns["depth_to_point_cloud"] = self.depth_to_point_cloud
        fns["mask_to_world_points"] = self.mask_to_world_points
        fns["pixel_to_world_point"] = self.pixel_to_world_point
        fns["transform_points"] = self.transform_points
        fns["interpolate_segment"] = self.interpolate_segment
        fns["normalize_vector"] = self.normalize_vector
        fns["select_top_down_grasp"] = self.select_top_down_grasp

        return fns

    # SKILL LIBRARY - Reusable Functions from LLM Robot Code Generation
    # ======================================================================
    # Source: reduced_api and reduced_api_exampleless experiments
    # Total unique functions analyzed: 182
    # Functions after filtering: 73
    # Minimum occurrence threshold: 2
    # ======================================================================

    # Here is a curated library of reusable robotics skills derived from the provided code generations.

    # I have categorized them into **Coordinate Transforms**, **Vision & Perception**, and **Geometry & Math**. I selected implementations that are vectorized (for performance), numerically stable (especially for quaternion conversion), and decoupled from specific environment dictionaries to ensure maximum reusability.

    ### 1. Category: Coordinate Transformations
    # These functions were the most frequent across all experiments (occurring >80 times in total). They are essential because planners often output matrices, but controllers (like `solve_ik`) often require quaternions.
    # **Why Reusable:** Converting between rotation matrices, quaternions, and homogeneous transformation matrices is a fundamental requirement for almost every manipulation task.

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].

        Implements the robust Sheppard's method (checking trace and diagonal elements)
        to avoid numerical instability when the trace is close to zero.

        Args:
            R: (3, 3) rotation matrix.

        Returns:
            np.array: [w, x, y, z] unit quaternion.

        """
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return np.array([w, x, y, z])

    def decompose_transform(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decompose a 4x4 homogeneous transformation matrix into position and quaternion.

        Args:
            T: (4, 4) homogeneous transformation matrix.

        Returns:
            tuple:
                - position: (3,) np.array
                - quaternion: (4,) np.array [w, x, y, z]

        """
        position = T[:3, 3]
        R = T[:3, :3]
        quat = self.rotation_matrix_to_quaternion(R)
        return position, quat

    ### 2. Category: Vision & Perception (Depth to 3D)
    # These functions bridge the gap between 2D camera data and 3D robot actions. They are crucial for converting segmentation masks into grasp targets.

    # **Why Reusable:** They encapsulate the pinhole camera model math, handling intrinsics (projection) and extrinsics (camera pose), allowing the agent to reason in the World Frame.

    def depth_to_point_cloud(self, depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Convert a depth image to a 3D point cloud in the Camera Frame.

        Args:
            depth_img: (H, W) depth map in meters.
            intrinsics: (3, 3) camera intrinsic matrix.

        Returns:
            np.array: (H, W, 3) image of 3D coordinates.

        """
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]

        h, w = depth_img.shape
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Vectorized grid generation
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        z = depth_img
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy

        return np.dstack((x, y, z))

    def mask_to_world_points(
        self, mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Convert specific pixels defined by a binary mask into 3D points in the World Frame.

        Args:
            mask: (H, W) binary mask (0 or 1).
            depth: (H, W) depth map.
            intrinsics: (3, 3) camera intrinsics.
            extrinsics: (4, 4) camera-to-world pose matrix.

        Returns:
            np.array: (N, 3) array of valid 3D points in world coordinates.

        """
        # Get pixel coordinates
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return np.empty((0, 3))

        z_vals = depth[ys, xs]

        if len(z_vals.shape) == 2:
            z_vals = z_vals.flatten()
        # print(f"z_vals.shape: {z_vals.shape}")

        # Filter invalid depth
        valid = z_vals > 0
        ys = ys[valid]
        xs = xs[valid]
        z = z_vals[valid]

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Deproject to Camera Frame
        x_cam = (xs - cx) * z / fx
        y_cam = (ys - cy) * z / fy

        # Stack to (N, 3)
        points_cam = np.stack([x_cam, y_cam, z], axis=-1)

        # Transform to World Frame
        # Create homogeneous coordinates (N, 4)
        points_cam_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world_hom = (extrinsics @ points_cam_hom.T).T

        return points_world_hom[:, :3]

    def pixel_to_world_point(
        self, u: int, v: int, z: float, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Deproject a single pixel to a 3D world point.

        Args:
            u, v: Pixel coordinates (col, row).
            z: Depth at that pixel.
            intrinsics: (3, 3) matrix.
            extrinsics: (4, 4) matrix.

        Returns:
            np.array: [x, y, z] in world frame.

        """
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy

        p_cam = np.array([x_cam, y_cam, z, 1.0])
        p_world = extrinsics @ p_cam
        return p_world[:3]

    ### 3. Category: Geometry & Math
    # These functions help manipulate 3D data once it has been extracted from the camera.

    # **Why Reusable:** The `transform_points` function is particularly useful because it handles both lists of points `(N, 3)` and organized point clouds `(H, W, 3)` via reshaping, making it a "do-it-all" spatial transformer.

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply a 4x4 homogeneous transform to a set of 3D points.

        Args:
            points: (N, 3) or (H, W, 3) array of points.
            transform_matrix: (4, 4) homogeneous transformation matrix.

        Returns:
            np.array: Transformed points with same shape as input.

        """
        original_shape = points.shape
        # Flatten to (N, 3)
        points_reshaped = points.reshape(-1, 3)

        # Convert to homogeneous (N, 4)
        ones = np.ones((points_reshaped.shape[0], 1))
        points_hom = np.hstack((points_reshaped, ones))

        # Apply transform: (4,4) @ (4,N) -> (4,N) -> Transpose back to (N,4)
        points_transformed = (transform_matrix @ points_hom.T).T

        # Return to (N, 3) and original shape
        return points_transformed[:, :3].reshape(original_shape)

    def interpolate_segment(
        self, p1: np.ndarray, p2: np.ndarray, step: float = 0.03
    ) -> list[np.ndarray]:
        """
        Generate waypoints along a line segment between two 3D points.

        Args:
            p1: Start point (3,).
            p2: End point (3,).
            step: Distance between waypoints in meters.

        Returns:
            list[np.ndarray]: List of points including p1 and p2.

        """
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            return [p1]
        num_points = int(np.ceil(dist / step))
        # Using linspace to ensure we hit the start and end exactly
        return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, num_points + 1)]

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            v: (3,) vector.

        Returns:
            np.array: (3,) unit vector.
        """
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return v
        return v / norm

    # ### 4. Category: Grasp Heuristics
    # A reusable heuristic for filtering grasps generated by learned models (like Contact-GraspNet).

    def select_top_down_grasp(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        cam_to_world: np.ndarray,
        vertical_threshold: float = 0.8,
    ) -> tuple:
        """
        Selects the best grasp that aligns the gripper vertically (Top-Down).

        Args:
            grasps: (N, 4, 4) Grasp poses in camera frame.
            scores: (N,) Grasp scores.
            cam_to_world: (4, 4) Extrinsics matrix.
            vertical_threshold: Dot product threshold (1.0 is perfectly vertical).

        Returns:
            tuple: (best_grasp_world_matrix, best_score) or (None, -inf)
        """
        best_grasp = None
        best_score = -np.float64("inf")

        # World Z axis (vertical)
        world_z = np.array([0, 0, 1])

        for i, g_camera in enumerate(grasps):
            # Transform grasp to world frame
            g_world = cam_to_world @ g_camera

            # Extract rotation
            R = g_world[:3, :3]

            # Assuming Gripper Z or Y is the approach vector depending on gripper definition.
            # For Franka/Robotiq, the approach vector is usually the Z-axis of the end effector.
            gripper_approach = R[:, 2]

            # Check alignment with negative World Z (pointing down)
            # Dot product should be close to -1 for top-down
            alignment = -np.dot(gripper_approach, world_z)

            if alignment > vertical_threshold:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_grasp = g_world

        return best_grasp, best_score
