import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import PIL
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
    extract_arm_joints,
    open_gripper as _open_gripper,
)
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.molmo import init_molmo
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki import init_pyroki
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from capx.integrations.vision.sam2 import init_sam2
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb


# ------------------------------- Control API ------------------------------
class FrankaControlApiReduced(ApiBase):
    """
    Robot control helpers for Franka.
    """

    def __init__(self, env: BaseEnv, tcp_offset: list[float] | None = [0.0, 0.0, -0.107]) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        print("init franka control api")
        self.owl_vit_det_fn = init_owlvit(
            device="cuda"
        )  # TODO: refactor this and use registered api instead
        print("init owlvit det fn")
        self.grasp_net_plan_fn = (
            init_contact_graspnet()
        )  # TODO: refactor this and use registered api instead
        print("init grasp net plan fn")
        self.sam2_seg_fn = init_sam2()
        print("init sam2 seg fn")
        self.molmo_point_fn = init_molmo()
        print("init molmo point fn")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg = None

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "detect_object_owlvit": self.detect_object_owlvit,
            "segment_sam2": self.segment_sam2,
            "plan_grasp": self.plan_grasp,
            "move_to_joints": self.move_to_joints,
            "solve_ik": self.solve_ik,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "point_prompt_molmo": self.point_prompt_molmo,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["robot0_robotview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_robotview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W, 1), dtype float32.
                - ["robot0_robotview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_robotview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
        """
        return self._env.get_observation()

    # --------------------------------------------------------------------- #
    # Vision models: OWL-ViT detection + SAM2 segmentation
    # --------------------------------------------------------------------- #
    def detect_object_owlvit(
        self,
        rgb: np.ndarray,
        text: str,
    ) -> list[dict[str, Any]]:
        """Run OWL-ViT open-vocabulary detection on a single RGB image.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
                This should typically come from:
                    rgb = obs["robot0_robotview"]["images"]["rgb"]
            text:
                Natural language text query for OWL-ViT.

        Returns:
            detections:
                A list of dictionaries, one per detected box. Each dict typically
                contains:

                  - "box":   [x1, y1, x2, y2] in pixel coordinates (float)
                  - "label": str, the text label that matched best
                  - "score": float, confidence score in [0, 1]

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]  # (H, W, 3)
            >>> dets = detect_object_owlvit(rgb, text="red mug")
            >>> if dets:
            ...     best = max(dets, key=lambda d: d["score"])
            ...     print(best["box"], best["label"], best["score"])
        """
        return self.owl_vit_det_fn(rgb, texts=[[text]])

    def segment_sam2(
        self,
        rgb: np.ndarray,
        box: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run SAM2 segmentation on an RGB image, optionally conditioned on a box.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            box:
                Optional bounding box [x1, y1, x2, y2] in pixel coordinates, float.
                If provided, SAM2 will segment primarily within this region.
                If None, SAM2 runs in global mode over the whole image.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool or uint8,
                              where True/1 means the pixel belongs to the instance.
                  - "score": float confidence score (if provided by SAM2).

        Example:
            >>> rgb = obs["robot0_robotview"]["images"]["rgb"]
            >>> dets = detect_object_owlvit(rgb, text="red mug")
            >>> best = max(dets, key=lambda d: d["score"])
            >>> box = best["box"]
            >>> masks = segment_sam2(rgb, box=box)
        """
        return self.sam2_seg_fn(rgb, box=box)

    def point_prompt_molmo(
        self,
        image: PIL.Image.Image,
        object_name: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Use Molmo to point to an object in the image.

        Args:
            image: PIL.Image.Image: The RGB image to process.
            objects: list[str]: The list of object queries to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """
        return self.molmo_point_fn(image, objects=[object_name])

    # --------------------------------------------------------------------- #
    # Grasp planner (Contact-GraspNet)
    # --------------------------------------------------------------------- #
    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
        instance_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet for a single instance.

        This is a thin wrapper around the Contact-GraspNet planner. It does not
        apply any camera/world transforms or TCP offsets: the caller is
        responsible for transforming the resulting grasp poses into the desired
        frame and applying TCP offsets if necessary.

        Args:
            depth:
                Depth image in meters.
                Shape: (H, W) or (H, W, 1), dtype float32/float64.
            intrinsics:
                Camera intrinsic matrix.
                Shape: (3, 3), dtype float64.
            segmentation:
                Instance segmentation map where each integer > 0 corresponds to a
                unique object instance ID.
                Shape: (H, W) or (H, W, 1), dtype int32/int64.
            instance_id:
                The integer ID of the object to grasp (e.g., 1, 2, 3, ...).

        Returns:
            grasp_poses:
                np.ndarray of shape (K, 4, 4), dtype float64.
                Homogeneous transforms for each candidate grasp IN THE CAMERA FRAME.
            grasp_scores:
                np.ndarray of shape (K,), dtype float64.
                Confidence score for each candidate grasp.
            contact_points:
                np.ndarray of shape (K, 3), dtype float64.
                Contact point positions in camera frame.

        Example:
            >>> cam = obs["robot0_robotview"]
            >>> rgb = cam["images"]["rgb"]
            >>> dets = detect_object_owlvit(rgb, text="red mug")
            >>> if dets:
            ...     best = max(dets, key=lambda d: d["score"])
            ...     box = best["box"]
            ...     instance_id = best["label"]
            >>> depth = cam["images"]["depth"][:, :, 0]
            >>> seg = sam2_seg_fn(rgb, box=box)
            >>> K = cam["intrinsics"]
            >>> grasp_poses, scores, contact_pts = plan_grasp(
            ...     depth=depth,
            ...     intrinsics=K,
            ...     segmentation=seg,
            ...     instance_id=instance_id,
            ... )
            >>> best_idx = scores.argmax()
            >>> best_T = grasp_poses[best_idx]  # (4, 4)
            >>> camera_extrinsics = cam["pose_mat"]
            >>> grasp_sample_world_frame = camera_extrinsics @ best_T
        """
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[:, :, 0]

        grasp_sample, grasp_scores, grasp_contact_pts = self.grasp_net_plan_fn(
            depth,
            intrinsics,
            segmentation,
            instance_id,
        )
        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        return grasp_sample_tf, grasp_scores, grasp_contact_pts

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
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)

        cfg = self.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
        )
        return extract_arm_joints(cfg)

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
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        _open_gripper(self._env, steps=30)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        _close_gripper(self._env, steps=30)
