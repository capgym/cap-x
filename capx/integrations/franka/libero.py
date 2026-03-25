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
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.graspnet import init_contact_graspnet, init_contact_graspnet_point_clouds
from capx.integrations.vision.molmo import init_molmo
from capx.integrations.vision.sam2 import init_sam2_point_prompt
from capx.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)
from capx.integrations.motion.pyroki import init_pyroki

_curobo_api = None

def _get_curobo_api():
    """Lazy import of cuRobo API to avoid warp init before Isaac Sim."""
    global _curobo_api
    if _curobo_api is None:
        from capx.integrations.motion import curobo_api as _mod
        _curobo_api = _mod
    return _curobo_api
from sklearn.cluster import DBSCAN

# ------------------------------- Control API ------------------------------
class FrankaLiberoApi(ApiBase):
    """Robot control helpers for Franka.
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)
    
    def __init__(self, env: BaseEnv, use_sam3: bool = True) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._target_link_name = ctx.target_link_name
        self._pks = pks

        # Initialize perception models for non-privileged API
        self.use_sam3 = use_sam3
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            self.sam3_point_prompt_fn = init_sam3_point_prompt()
        else:
            self.sam2_point_prompt_fn = init_sam2_point_prompt()
        self.molmo_point_fn = init_molmo()
        self.grasp_net_plan_fn = init_contact_graspnet()
        # used for multiview grasps
        self.grasp_net_plan_point_clouds_fn = init_contact_graspnet_point_clouds()
        self.ik_solve_fn = init_pyroki()
        self.camera_name = "agentview"
        self.wrist_camera_name = "robot0_eye_in_hand"
        self.cfg = None
        self._curobo_world_config = None

    def functions(self) -> dict[str, Any]:
        fns =  {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_oriented_bounding_box_from_3d_points": self.get_oriented_bounding_box_from_3d_points,
            "get_object_3d_points_and_masks_from_language": self.get_object_3d_points_and_masks_from_language,
            "goto_home_joint_position": self.goto_home_joint_position,
            # # CuRobo, uncomment these for the coding agent to use them!
            # "plan_grasp_trajectory": self.plan_grasp_trajectory,
            # "plan_with_grasped_object": self.plan_with_grasped_object,
            # "execute_joint_trajectory": self.execute_joint_trajectory,
        }

        return fns

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
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        return self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)

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
        Note:
            Returns an empty list if no results are found.

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3(rgb, text_prompt="red mug")
        """
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        if len(results) == 0:
            print(f"[segment_sam3_text_prompt] SAM3 returned no results for prompt: '{text_prompt}'")
            return []
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
        return self.molmo_point_fn(Image.fromarray(image), objects=[text_prompt])

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

    def get_object_pose(self, object_name: str, use_multiview: bool = True) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp
        pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for
        placement position.
        It is possible that get_object_pose is sometimes unreliable and will return None for both
        position and quaternion.

        Args:
            object_name: The name of the object to get the pose of, in underscore separated lowercase words.
            use_multiview: If True, uses the wrist camera as well as the main camera for segmentation. If False, only uses the main camera for segmentation.

        Returns:
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
        """
        start_time = time.time()

        result = self.get_object_3d_points_and_masks_from_language(
            object_name, use_multiview=use_multiview
        )
        points_3d = result["points_3d"]

        if len(points_3d) == 0:
            return None, None

        points_3d, _ = self.filter_noise(points_3d)
        if len(points_3d) == 0:
            return None, None

        obb = self.get_oriented_bounding_box_from_3d_points(points_3d)

        position = np.array(obb["center"])
        R = np.array(obb["R"])

        # Ensure z-axis points down for gripper frame consistency
        z_axis_world = R[:, 2]
        if z_axis_world[2] > 0:
            R = R @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        quaternion_wxyz = vtf.SO3.from_matrix(R).wxyz

        print(f"get_object_pose in {time.time() - start_time} seconds")
        return position, quaternion_wxyz

    def sample_grasp_pose(self, object_name: str, use_multiview: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Uses multiview point clouds and plan_grasp_from_point_clouds for more reliable grasp planning.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for, in underscore separated lowercase words.
            use_multiview: If True, uses the wrist camera as well as the main camera for segmentation. If False, only uses the main camera for segmentation.

        Returns:
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
        """
        start_time = time.time()

        result = self.get_object_3d_points_and_masks_from_language(
            object_name, use_multiview=use_multiview
        )
        pc_segment = result["points_3d"]

        if len(pc_segment) == 0:
            raise ValueError(f"Could not segment object '{object_name}'")

        # Build full scene point cloud in world frame from both cameras
        obs = self.get_observation()
        pc_full_parts = []
        for cam_name in [self.camera_name, self.wrist_camera_name]:
            depth = obs[cam_name]["images"]["depth"]
            intrinsics = obs[cam_name]["intrinsics"]
            extrinsics = obs[cam_name]["pose_mat"]
            pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
            pts_homogeneous = np.concatenate(
                [pts_camera, np.ones((len(pts_camera), 1))], axis=1
            )
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]
            pc_full_parts.append(pts_world)
        pc_full = np.concatenate(pc_full_parts)

        pc_segment, _ = self.filter_noise(pc_segment)
        if len(pc_segment) == 0:
            raise ValueError(f"No valid points after filtering for '{object_name}'")

        # pc_full = self.subsample_point_cloud(pc_full, max_points=20000)
        # pc_segment = self.subsample_point_cloud(pc_segment, max_points=10000)

        grasp_sample_tf, grasp_scores = self.plan_grasp_from_point_clouds(pc_full, pc_segment)

        best_idx = grasp_scores.argmax()
        best_grasp = vtf.SE3.from_matrix(grasp_sample_tf[best_idx])
        best_grasp = best_grasp @ vtf.SE3.from_rotation(
            rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2)
        )

        print(f"sample_grasp_pose in {time.time() - start_time} seconds")
        print(f"Grasp sample position for {object_name}: {best_grasp.wxyz_xyz[-3:]}")
        print(f"Grasp sample quaternion wxyz for {object_name}: {best_grasp.wxyz_xyz[:4]}")
        return best_grasp.wxyz_xyz[-3:], best_grasp.wxyz_xyz[:4]

    def _segment_object_from_language(
        self, image: Image.Image, object_name: str
    ) -> tuple[np.ndarray, tuple[int, int] | None, list[float] | None]:
        """Use SAM3 (or Molmo + SAM2) to return a binary mask for a language-described object."""
        if self.use_sam3:
            # SAM3: unified detection + segmentation from text prompt
            results = self.sam3_seg_fn(image, text_prompt=object_name)
            if len(results) == 0:
                # try molmo if language sam3 fails
                dets = self.molmo_point_fn(image, objects=[object_name])
                point = dets.get(object_name)
                if point is None or any(coord is None for coord in point):
                    return None, None, None
                point_coords = (float(point[0]), float(point[1]))
                results = self.sam3_point_prompt_fn(image, point_coords=point_coords)
            
            if len(results) == 0:
                return None, None, None

            scores = [result["score"] for result in results]
            best_result = results[np.argmax(scores)]
            mask_bool = best_result["mask"].astype(bool)

            # Get center point of mask for visualization
            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                point_xy = (int(xs.mean()), int(ys.mean()))
            else:
                point_xy = None

            return mask_bool, point_xy, scores
        else:
            # Molmo + SAM2: separate detection and segmentation
            dets = self.molmo_point_fn(image, objects=[object_name])
            point = dets.get(object_name)
            if point is None or any(coord is None for coord in point):
                return None, None, None
            point_coords = (float(point[0]), float(point[1]))
            scores, masks = self.sam2_point_prompt_fn(image, point_coords=point_coords)
            if len(masks) == 0:
                raise ValueError(f"SAM2 returned no masks for '{object_name}'")

            best_mask = np.asarray(masks[0])
            best_mask = np.squeeze(best_mask)
            if best_mask.ndim != 2:
                raise ValueError(f"SAM2 mask must be 2D, got shape {best_mask.shape}")

            mask_bool = best_mask.astype(bool)
            point_xy = (int(round(point_coords[0])), int(round(point_coords[1])))
            return mask_bool, point_xy, scores

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
        # inject some noise to the points
        points = points + np.random.normal(0, 0.0001, points.shape)
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)
        o3d_points, ind = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb = o3d_points.get_oriented_bounding_box()
        return {
            "center": obb.center,
            "extent": obb.extent,
            "R": obb.R,
            "quaternion_wxyz": vtf.SO3.from_matrix(obb.R).wxyz,
        }

    def get_object_3d_points_and_masks_from_language(
        self,
        text_prompt: str,
        use_multiview: bool = True,
    ) -> dict[str, Any]:
        """Segment an object using text prompt across multiple views (agentview and wrist).
        
        Uses Molmo to locate the object in both camera views, then SAM3 for
        segmentation. Returns the masks and 3D points that appear in both views.
        
        Args:
            text_prompt: Text description of the object to segment.
            use_multiview: If True, uses the wrist camera as well as the main camera for segmentation.
            
        Returns:
            dict containing:
                - "agentview_mask": np.ndarray of shape (H, W), dtype bool
                - "wrist_mask": np.ndarray of shape (H, W), dtype bool (None if single view)
                - "points_3d": np.ndarray of shape (N, 3), 3D points in world frame
                - "agentview_score": float, SAM3 confidence score
                - "wrist_score": float, SAM3 confidence score (None if single view)
        """
        obs = self.get_observation()
        
        cameras = [self.camera_name]
        if use_multiview:
            cameras.append(self.wrist_camera_name)
            
        camera_data = {}
        
        for cam_name in cameras:
            # get images from camera
            rgb = obs[cam_name]["images"]["rgb"]
            depth = obs[cam_name]["images"]["depth"]
            intrinsics = obs[cam_name]["intrinsics"]
            extrinsics = obs[cam_name]["pose_mat"]
            
            # use Molmo to point prompt
            points = self.point_prompt_molmo(rgb, text_prompt)
            point = points[text_prompt]
            
            # Get SAM3 segmentations (fall back to text prompt if point prompt yields nothing)
            masks = []
            if point[0] is not None:
                masks = self.segment_sam3_point_prompt(rgb, point)
            if not masks:
                masks = self.segment_sam3_text_prompt(rgb, text_prompt)
            if not masks:
                raise ValueError(
                    f"SAM3 segmentation failed for '{text_prompt}' on {cam_name}. "
                    f"No masks returned from either point or text prompt."
                )
            mask_data = max(masks, key=lambda x: x["score"])
            
            mask = mask_data["mask"]
            score = mask_data["score"]
            
            # Get 3D points in camera frame
            pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
            
            # Transform to world frame
            pts_homogeneous = np.concatenate([pts_camera, np.ones((len(pts_camera), 1))], axis=1) # add column of ones
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]
            
            # Apply mask - ensure shapes match
            mask_flat = mask.flatten()
            if len(pts_world) != len(mask_flat):
                print(f"Warning: Point cloud size ({len(pts_world)}) doesn't match mask size ({len(mask_flat)}) for {cam_name}")
                min_len = min(len(pts_world), len(mask_flat))
                pts_3d = pts_world[:min_len][mask_flat[:min_len]]
            else:
                pts_3d = pts_world[mask_flat]
            
            camera_data[cam_name] = {
                "mask": mask,
                "score": score,
                "points_3d": pts_3d
            }
        
        agent_data = camera_data[self.camera_name]
        agent_pts_3d = agent_data["points_3d"]
        
        points_3d = agent_pts_3d
        
        wrist_mask = None
        wrist_score = None
        wrist_pts_3d = None
        
        if use_multiview and self.wrist_camera_name in camera_data:
            # Takes the union of the points from the wrist and agent view if they are close enough, otherwise takes the view with the higher score.
            wrist_data = camera_data[self.wrist_camera_name]
            wrist_pts_3d = wrist_data["points_3d"]
            wrist_mask = wrist_data["mask"]
            wrist_score = wrist_data["score"]
            
            # Find intersection using numpy broadcasting
            if len(wrist_pts_3d) > 0 and len(agent_pts_3d) > 0:
                # Compute pairwise distances between all agent and wrist points
                distances = np.linalg.norm(
                    agent_pts_3d[:, np.newaxis, :] - wrist_pts_3d[np.newaxis, :, :], 
                    axis=2
                )
                # Find minimum distance for each agent point
                min_distances = np.min(distances, axis=1)
                # Keep agent points within threshold
                threshold = 0.01  # 1cm
                if min_distances.min() < threshold:
                    points_3d = np.concatenate([agent_pts_3d, wrist_pts_3d])
                else:
                    if wrist_score > agent_data["score"]:
                        points_3d = wrist_pts_3d
                    else:
                        points_3d = agent_pts_3d
            elif len(wrist_pts_3d) > 0:
                points_3d = wrist_pts_3d
            elif len(agent_pts_3d) > 0:
                points_3d = agent_pts_3d
            else:
                print(f"Warning: No points found for {text_prompt}")
                points_3d = np.array([]).reshape(0, 3)
        
        return {
            "agentview_mask": agent_data["mask"],
            "wrist_mask": wrist_mask,
            "points_3d": np.asarray(points_3d),
            "agentview_points_3d": np.asarray(agent_pts_3d),
            "wrist_points_3d": np.asarray(wrist_pts_3d),
            "agentview_score": agent_data["score"],
            "wrist_score": wrist_score,
        }
    
    def goto_home_joint_position(self) -> None:
        """Return the arm to its reset joint configuration with high manipulability"""
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def subsample_point_cloud(self, pc: np.ndarray, max_points: int = 10000) -> np.ndarray:
        """Randomly subsample a point cloud to a maximum number of points.
        
        Args:
            pc: (N, 3) array of points.
            max_points: The maximum number of points to subsample to. Default is 10000.
        Returns:
            subsampled_pc: (M, 3) array of subsampled points where M <= max_points.
        """
        if len(pc) > max_points:
            return pc[np.random.choice(len(pc), max_points, replace=False)]
        return pc
    
    def filter_noise(self, points: np.ndarray, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Filter noise from the point cloud.
        Args:
            points: (N, 3) array of points.
            colors: (N, 3) array of colors. Optional, default is None.
        Returns:
            (M, 3) array of points.
            (M, 3) array of colors. Optional, default is None.
        """
        eps = 0.005  # Maximum distance between samples to be neighbors
        min_samples = 10
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        filtered_pointcloud = points[labels != -1]
        if colors is not None:
            filtered_colors = colors[labels != -1]
        else:
            filtered_colors = None
        return filtered_pointcloud, filtered_colors
    
    def plan_grasp_from_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet given a full point cloud and a segmented point cloud of the object wanting to be grasped. These point clouds should be composed from multiple viewpoints.
        NOTE: This function is best for generating grasps on objects with complex shapes. For regular objects like boxes, cylinders, etc..., use get_top_down_grasp_from_obb.

        Args:
            pc_full:
                Point cloud of the full environment, including the object to be grasped.
                Shape: (N, 3), dtype float32.
            pc_segment:
                Point cloud of the segmented object to be grasped.
                Shape: (N, 3), dtype float32.
        Returns:
            grasp_sample_tf: (4, 4) homogeneous transform for the grasp pose in THE POINT CLOUD FRAME.
            grasp_scores: (N,) array of grasp scores.
        """
        grasp_sample, grasp_scores, _ = self.grasp_net_plan_point_clouds_fn(pc_full, pc_segment, segmap_id=1)
        
        assert len(grasp_sample) > 0, "No grasp candidates found"

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        return grasp_sample_tf, grasp_scores
    
    def parse_grasp_poses_for_curobo(
        self, grasp_poses_world: np.ndarray, grasp_scores: np.ndarray, top_k: int = 15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parses grasp poses from the world frame into a format compatible with CuRobo.

        Args:
            grasp_poses_world: (N, 4, 4) array of grasp poses in world frame
            grasp_scores: (N,) array of grasp scores
            top_k: number of top grasp poses to return. Default is 15.
        Returns:
            positions: (N, 3) array of grasp positions
            quaternions: (N, 4) array of grasp quaternions
            scores: (N,) array of grasp scores sorted by score
        
        Example:
            >>> grasp_poses_cam, grasp_scores = plan_grasp_from_point_clouds(...)
            >>> positions, quaternions, scores = parse_grasp_poses_for_curobo(grasp_poses_cam, grasp_scores, top_k=15)
        """
        grasp_poses_world = np.asarray(grasp_poses_world)
        positions = grasp_poses_world[..., 3][:, :3]
        rotations = grasp_poses_world[..., :3][:, :3, :3]
        quaternions = vtf.SO3.from_matrix(rotations).wxyz
        scores = grasp_scores
        k = min(top_k, len(positions))
        order = np.argsort(-scores)[:k]
        return positions[order], quaternions[order], scores[order]
    
    ### CuRobo-related functions ###
    def create_curobo_world_from_depth(
        self,
        depth_image: np.ndarray,
        object_mask: np.ndarray,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray | None = None,
        **kwargs: Any,
    ):
        """Create a CuRobo WorldConfig from a depth image and object mask. Stores the result for use by plan_grasp_trajectory."""
        world = _get_curobo_api().create_curobo_world_from_depth(
            depth_image, object_mask, intrinsics, camera_pose=camera_pose, **kwargs
        )
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_pointcloud(
        self, point_cloud: np.ndarray, object_mask: np.ndarray, **kwargs: Any
    ):
        """Create a CuRobo WorldConfig from a point cloud and per-point object mask. Stores the result for use by plan_grasp_trajectory."""
        world = _get_curobo_api().create_curobo_world_from_pointcloud(point_cloud, object_mask, **kwargs)
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_observation(
        self,
        object_mask: np.ndarray,
        *,
        camera_name: str | None = None,
        object_name: str = "object",
        scene_name: str = "scene",
        **kwargs: Any,
    ):
        """Create a CuRobo WorldConfig from the current observation: uses this camera's depth, intrinsics, and pose to build the world, split by object_mask. Stores the result for use by plan_grasp_trajectory.
        
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        intrinsics = obs[cam]["intrinsics"]
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=pose[:3],
        ).as_matrix()
        world = _get_curobo_api().create_curobo_world_from_depth(
            depth,
            object_mask,
            intrinsics,
            camera_pose=camera_pose_4x4,
            object_name=object_name,
            scene_name=scene_name,
            **kwargs,
        )
        self._curobo_world_config = world
        return world

    def update_curobo_world(
        self,
        *,
        camera_name: str | None = None,
        robot_distance_threshold: float = 0.15,
        robot_file: str = "franka.yml",
        **kwargs: Any,
    ) -> Any:
        """Build CuRobo world from current observation (full depth, robot excluded) and store it.

        Uses create_curobo_world_from_depth_full: single scene mesh with points near the
        robot removed (robot_distance_threshold) so start/IK configs are not in collision.
        Same logic as curobo_test script. Use before plan_grasp_trajectory when using
        collision checking.

        Args:
            camera_name: Camera to use; default is self.camera_name.
            robot_distance_threshold: Distance (m) to exclude points near the robot. Default is 0.15.
            robot_file: CuRobo robot config for segmenter. Default is 'franka.yml'.
            **kwargs: Passed to create_curobo_world_from_depth_full.

        Returns:
            The WorldConfig that was stored in self._curobo_world_config.
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = np.asarray(depth[:, :, 0], dtype=np.float64)
        else:
            depth = np.asarray(depth, dtype=np.float64)
        intrinsics = np.asarray(obs[cam]["intrinsics"], dtype=np.float64)
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=np.asarray(pose[:3], dtype=np.float64),
        ).as_matrix()
        robot_joint_pos = np.asarray(obs["robot_joint_pos"], dtype=np.float64)
        world = _get_curobo_api().create_curobo_world_from_depth_full(
            depth,
            intrinsics,
            camera_pose=camera_pose_4x4,
            robot_joint_position=robot_joint_pos,
            robot_file=robot_file,
            robot_distance_threshold=robot_distance_threshold,
            **kwargs,
        )
        self._curobo_world_config = world
        return world

    def plan_grasp_trajectory(
        self,
        object_name: str,
        *,
        object_mask: np.ndarray, #| None = None,
        grasp_poses: list[tuple[np.ndarray, np.ndarray]], #| None = None,
        top_k_grasps: int = 15,
        use_world_collision: bool = True,
        robot_distance_threshold: float = 0.15,
        robot_collision_sphere_buffer: float | None = -0.01,
        collision_activation_distance: float | None = 0.001,
        world_config: Any = None,
        # use_multiview: bool = False,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray | None, int | None]:
        """Plan a collision-free trajectory to one of the top-k grasp poses.

        Builds a CuRobo world with object mesh + scene mesh (robot excluded via segmenter).
        Collision between the object and the rest of the world is disabled so the robot
        can approach the object. If grasp_poses is None, samples grasps for object_name.

        Args:
            object_name: Name of the object (used for mask and grasps if not provided).
            object_mask: precomputed mask (H, W) of the object.
            grasp_poses: list of (position (3,), quaternion_wxyz (4,)) in world frame.
            top_k_grasps: Number of top grasps to try (by score). Default 15.
            use_world_collision: If True, use world for collision (scene only; object ignored for grasp). Default is True.
            robot_distance_threshold: Passed when building world. Default is 0.15.
            robot_collision_sphere_buffer: Passed to plan_to_grasp_poses. Default is -0.01.
            collision_activation_distance: Passed to plan_to_grasp_poses. Default is 0.001.
            world_config: Optional WorldConfig. If None, builds via update_curobo_world_with_object. Default is None.
            **kwargs: Passed through to plan_to_grasp_poses.

        Returns:
            (success, joint_trajectory, goalset_index): joint_trajectory is (T, 7) or None.
        """
        if grasp_poses is None:
            # positions, quaternions, scores = self._sample_grasp_poses_for_object(object_name, use_multiview=use_multiview)
            # if len(positions) == 0:
            #     return False, None, None
            # k = min(top_k_grasps, len(positions))
            # order = np.argsort(-scores)[:k] if scores is not None and len(scores) else np.arange(k)
            # grasps_to_try = [(positions[i], quaternions[i]) for i in order]
            assert grasp_poses is not None, "grasp_poses must be provided"
        else:
            grasps_to_try = [(np.asarray(p), np.asarray(q)) for p, q in grasp_poses]

        world = world_config
        if world is None:
            self.update_curobo_world_with_object(
                object_name,
                object_mask=object_mask,
                robot_distance_threshold=robot_distance_threshold,
            )
            world = self._curobo_world_config
            # Disable collision with the object so the robot can reach into it
            ignore_obstacle_names = [getattr(self, "_curobo_world_object_name", object_name.replace(" ", "_"))]
        else:
            ignore_obstacle_names = []

        obs = self._env.get_observation()
        start_joint_position = obs["robot_joint_pos"]

        success, joint_traj, goalset_idx = _get_curobo_api().plan_to_grasp_poses(
            world,
            start_joint_position,
            grasps_to_try,
            use_world_collision=use_world_collision,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance,
            ignore_obstacle_names=ignore_obstacle_names if use_world_collision else None,
            **kwargs,
        )
        return success, joint_traj, goalset_idx

    def execute_joint_trajectory(
        self,
        joint_trajectory: np.ndarray,
        *,
        subsample: int = 1,
        tolerance: float = 0.01,
        max_steps: int = 120,
    ) -> None:
        """Execute a joint-space trajectory (T, 7) by moving to each waypoint with move_to_joints_blocking.

        Args:
            joint_trajectory: (T, 7) joint positions in radians.
            subsample: Use every Nth waypoint (1 = all). Larger values speed up execution. Default is 1.
            tolerance: Passed to move_to_joints_blocking. Default is 0.01.
            max_steps: Passed to move_to_joints_blocking. Default is 120.
        """
        traj = np.asarray(joint_trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 7:
            raise ValueError(f"joint_trajectory must be (T, 7), got shape {traj.shape}")
        indices = list(range(0, len(traj), subsample))
        if indices and indices[-1] != len(traj) - 1:
            indices.append(len(traj) - 1)
        for i in indices:
            joints = traj[i, :7]
            self._env.move_to_joints_blocking(
                joints, tolerance=tolerance, max_steps=max_steps
            )
    
    def update_curobo_world_with_object(
        self,
        object_name: str,
        *,
        object_mask: np.ndarray | None = None,
        camera_name: str | None = None,
        robot_distance_threshold: float = 0.15,
        robot_file: str = "franka.yml",
        object_name_in_world: str | None = None,
        scene_name: str = "scene",
        **kwargs: Any,
    ) -> Any:
        """Build CuRobo world from current observation with object/scene split (robot excluded) and store it.

        Uses create_curobo_world_from_depth_with_object: creates two separate meshes:
        - Object mesh (from object_mask)
        - Scene mesh (everything else)

        Robot points are excluded so the start configuration is not in collision. Use this
        before plan_with_grasped_object to attach the object and plan motion.

        Args:
            object_name: Name of the object (used to get object_mask if object_mask is None).
            object_mask: Optional precomputed mask (H, W). If None, get_object_mask(object_name) is used.
            camera_name: Camera to use; default is self.camera_name.
            robot_distance_threshold: Distance (m) to exclude points near the robot. Default is 0.15.
            robot_file: CuRobo robot config for segmenter. Default is 'franka.yml'.
            object_name_in_world: Name for the object in WorldConfig (default: object_name, spaces → underscores).
            scene_name: Name for the scene mesh in WorldConfig. Default is 'scene'.
            **kwargs: Passed to create_curobo_world_from_depth_with_object.

        Returns:
            The WorldConfig that was stored in self._curobo_world_config.
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        elif depth.ndim == 2:
            pass
        else:
            raise ValueError(f"Depth image has invalid shape: {depth.shape}")
        depth = np.asarray(depth, dtype=np.float64)
        intrinsics = np.asarray(obs[cam]["intrinsics"], dtype=np.float64)
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=np.asarray(pose[:3], dtype=np.float64),
        ).as_matrix()
        robot_joint_pos = np.asarray(obs["robot_joint_pos"], dtype=np.float64)

        if object_mask is None:
            object_mask = self.get_object_mask(object_name)
        if object_mask is None:
            raise ValueError(f"Could not get object mask for '{object_name}' to create world with object/scene split")
        mask_bool = np.asarray(object_mask, dtype=bool)
        if getattr(self._env, "output_dir", None):
            rgb = np.asarray(obs[cam]["images"]["rgb"])
            self._save_rgb_with_object_mask(rgb, mask_bool, object_name, suffix="world")

        # World-safe name (no spaces) for CuRobo lookup
        world_object_name = object_name_in_world if object_name_in_world is not None else object_name
        mesh_name = world_object_name.replace(" ", "_")
        # Place object mesh at current EE pose so the world matches the grasped location (fixes wrong position in planning/debug)
        ee_pos, ee_quat_wxyz = self.get_ee_pose()
        world = _get_curobo_api().create_curobo_world_from_depth_with_object(
            depth,
            object_mask,
            intrinsics,
            camera_pose=camera_pose_4x4,
            robot_joint_position=robot_joint_pos,
            robot_file=robot_file,
            robot_distance_threshold=robot_distance_threshold,
            object_name=mesh_name,
            scene_name=scene_name,
            object_pose_override=(ee_pos, ee_quat_wxyz),
            **kwargs,
        )
        self._curobo_world_config = world
        self._curobo_world_object_name = mesh_name
        return world

    def plan_with_grasped_object(
        self,
        target_pose: tuple[np.ndarray, np.ndarray],
        object_name: str,
        *,
        object_pose: tuple[np.ndarray, np.ndarray] | None = None,
        object_mask: np.ndarray | None = None,
        world_config: Any = None,
        robot_collision_sphere_buffer: float | None = -0.01,
        collision_activation_distance: float | None = 0.01,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray | None]:
        """Plan a collision-free trajectory to move a grasped object to a target pose.

        NOTE: the grasped object must not be in collision with the scene before calling this function. First lift the object then plan the trajectory to the target pose.

        If object_mask is provided, the CuRobo world is rebuilt from the current observation
        so the object mesh is at its current (e.g. post-lift) position; collision between
        object and scene is enabled. The object is then attached to the robot and motion
        is planned to the target pose.

        Args:
            target_pose: (position (3,), quaternion_wxyz (4,)) target pose in world frame (e.g. basket).
            object_name: Name of the object to attach (used for world mesh name: spaces → underscores).
            object_pose: Optional (position, quat_wxyz) of object; unused for now, for API consistency. Default is None.
            object_mask: If provided, rebuild world from current observation with this mask before planning. Default is None.
            world_config: Optional WorldConfig. If None, uses stored or rebuilds when object_mask given. Default is None.
            robot_collision_sphere_buffer: Override robot collision_sphere_buffer (m). Default is -0.01m.
            collision_activation_distance: Distance (m) to activate collision cost. Default is 0.01m.
            **kwargs: Passed through to plan_with_grasped_object.

        Returns:
            (success, joint_trajectory): joint_trajectory is (T, 7) or None.
        """
        if object_mask is not None:
            self.update_curobo_world_with_object(
                object_name,
                object_mask=object_mask,
                robot_distance_threshold=0.15,
            )
        world = world_config if world_config is not None else self._curobo_world_config
        if world is None:
            raise ValueError(
                "No world_config available. Call update_curobo_world_with_object (or pass object_mask) first."
            )

        obs = self._env.get_observation()
        start_joint_position = obs["robot_joint_pos"]
        object_name_for_attach = getattr(
            self, "_curobo_world_object_name", None
        ) or object_name.replace(" ", "_")

        debug_out_dir = getattr(self._env, "output_dir", None)
        success, joint_traj = _get_curobo_api().plan_with_grasped_object(
            world,
            start_joint_position,
            target_pose,
            object_name_for_attach,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance,
            debug_out_dir=debug_out_dir,
            **kwargs,
        )
        return success, joint_traj
