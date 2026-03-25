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
from capx.integrations.vision.molmo import init_molmo
from capx.integrations.motion.pyroki import init_pyroki
from capx.integrations.vision.sam2 import init_sam2_point_prompt
from capx.integrations.vision.sam3 import init_sam3_point_prompt
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_to_pointcloud,
    depth_to_rgb,
)


# ------------------------------- Control API ------------------------------
class FrankaControlNutAssemblyVisualApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - goto_home_joint_position() -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        # from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        # from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
        # self._TCP_OFFSET = _TCP_OFFSET
        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self.molmo_point_fn = init_molmo()
        # self.sam2_point_prompt_fn = init_sam2_point_prompt()
        self.sam3_point_prompt_fn = init_sam3_point_prompt()
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg: np.ndarray | None = None
        self.camera_name = "robot0_robotview"

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "goto_home_joint_position": self.goto_home_joint_position,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.
        It is possible that get_object_pose is sometimes be unreliable and return None for both position and quaternion.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """

        # deproject the point to the world coordinate, use oriented bounding box to get the rotation, and return the position of the point and the rotation of the oriented bounding box
        start_time = time.time()
        obs = self._env.get_observation()
        print(f"get observation in {time.time() - start_time} seconds")

        fixed_rotation = None
        if all(i in object_name for i in ["square", "block"]):
            fixed_rotation = obs["nut_poses"]["square_peg"][3:]

        rgb_imgs = obs_get_rgb(obs)
        assert len(rgb_imgs.keys()) > 0, "No RGB images in obs"

        rgb = list(rgb_imgs.values())[0]
        pil_rgb = Image.fromarray(rgb).convert("RGB")

        mask_bool, point_px, sam_scores = self._segment_object_from_language(pil_rgb, object_name)
        if point_px is None:
            return None, None

        depth = obs[self.camera_name]["images"]["depth"][:, :, 0]
        if mask_bool.shape != depth.shape:
            raise ValueError(
                f"SAM2 mask shape {mask_bool.shape} does not match depth shape {depth.shape}"
            )

        if self._env.viser_debug:
            depth_img = depth_to_rgb(depth)
            Image.fromarray(depth_img).save("depth_image.jpg")

            mask_overlay = rgb.copy()
            mask_overlay[mask_bool] = np.array([255, 0, 0], dtype=np.uint8)
            overlay_img = Image.fromarray(mask_overlay)
            draw = ImageDraw.Draw(overlay_img)
            radius = 6
            draw.ellipse(
                [
                    point_px[0] - radius,
                    point_px[1] - radius,
                    point_px[0] + radius,
                    point_px[1] + radius,
                ],
                outline=(255, 255, 0),
                width=2,
            )
            overlay_path = pathlib.Path(f"{object_name.replace(' ', '_')}_sam2_overlay.jpg")
            overlay_img.save(overlay_path)
            print(f"SAM2 mask scores for {object_name}: {sam_scores}")

            mask_binary_path = pathlib.Path(f"{object_name.replace(' ', '_')}_sam2_mask.png")
            Image.fromarray(mask_bool.astype(np.uint8) * 255).save(mask_binary_path)

        mask_idxs = np.where(mask_bool.flatten())
        points = depth_to_pointcloud(depth, obs[self.camera_name]["intrinsics"])[mask_idxs]

        median_depth = np.min(depth[mask_bool])
        camera_point = deproject_pixel_to_camera(
            point_px, median_depth, obs[self.camera_name]["intrinsics"]
        )
        camera_tf = vtf.SE3.from_translation(camera_point)
        world_point = (
            vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(wxyz=obs[self.camera_name]["pose"][3:]),
                translation=obs[self.camera_name]["pose"][:3],
            )
            @ camera_tf
        )

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)

        obb = o3d_points.get_oriented_bounding_box()

        # Exposing these to the low level environment for viser
        if self._env.viser_debug:
            self._env.cube_center = obb.center
            self._env.cube_rot = obb.R

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs[self.camera_name]["pose"][3:]),
            translation=obs[self.camera_name]["pose"][:3],
        )
        obb_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R), translation=obb.center
        )
        obb_tf_world = cam_extr_tf @ obb_tf

        # if z axis isn't pointing down, flip the z axis
        R = obb_tf_world.rotation().as_matrix()
        z_axis_world = R[:, 2]
        if z_axis_world[2] > 0:
            obb_tf_world = obb_tf_world @ vtf.SE3.from_rotation(
                rotation=vtf.SO3.from_matrix(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
            )

        if fixed_rotation is None:
            fixed_rotation = obb_tf_world.wxyz_xyz[:4]

        if self._env.viser_server is not None:
            self._env.viser_server.scene.add_frame(
                f"{self.camera_name}/{object_name}_frame",
                position=camera_tf.wxyz_xyz[-3:],
                wxyz=camera_tf.wxyz_xyz[:4],
                axes_length=0.05,
                axes_radius=0.005,
            )

            self._env.viser_server.scene.add_frame(
                f"molmo_point_{object_name}",
                position=world_point.wxyz_xyz[-3:],
                wxyz=fixed_rotation,
                axes_length=0.05,
                axes_radius=0.005,
            )
        print(f"get_object_pose in {time.time() - start_time} seconds")
        return world_point.wxyz_xyz[-3:], fixed_rotation

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        # simplified the solution to use oriented bounding box + molmo
        return self.get_object_pose(object_name)

    # def goto_pose(
    #     self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    # ) -> None:
    #     """Solve IK for requested pose (with optional approach offset) and move joints smoothly.

    #     Args:
    #         position: (3,) XYZ in meters.
    #         quaternion_wxyz: (4,) WXYZ unit quaternion.
    #         z_approach: Optional approach distance along tool Z (meters). When non-zero the
    #             motion first reaches position + z_approach in tool Z before descending.
    #     """

    #     pos = np.asarray(position, dtype=np.float64).reshape(3)
    #     quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    #     quat_xyzw = np.array(
    #         [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    #     )
    #     rot = SciRotation.from_quat(quat_xyzw)
    #     offset_pos = pos + rot.apply(self._TCP_OFFSET)

    #     targets: list[tuple[np.ndarray, np.ndarray]] = []
    #     if z_approach != 0.0:
    #         z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
    #         targets.append((z_offset_pos, quat_wxyz))
    #         if self._env.viser_debug:
    #             self._env.mjcf_ee_frame_handle.position = z_offset_pos
    #             self._env.mjcf_ee_frame_handle.wxyz = quat_wxyz

    #             z_offset_gripper_pos = pos + rot.apply(np.array([0, 0, -z_approach]))
    #             self._env.mjcf_gripper_frame_handle.position = z_offset_gripper_pos
    #             self._env.mjcf_gripper_frame_handle.wxyz = quat_wxyz

    #     targets.append((offset_pos, quat_wxyz))
    #     if self._env.viser_debug:
    #         self._env.mjcf_ee_frame_handle.position = offset_pos
    #         self._env.mjcf_ee_frame_handle.wxyz = quat_wxyz

    #         self._env.mjcf_gripper_frame_handle.position = pos
    #         self._env.mjcf_gripper_frame_handle.wxyz = quat_wxyz

    #     seed = self.cfg
    #     for target_position, target_quat in targets:
    #         ik_solution = self._solve_ik_with_seed(target_position, target_quat, seed)
    #         self.cfg = ik_solution
    #         joints = self._extract_arm_joints(ik_solution)
    #         self._env.move_to_joints_blocking(joints)
    #         seed = ik_solution

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
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                )
            else:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
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

    def goto_home_joint_position(self) -> None:
        """Return the arm to its reset joint configuration with high manipulability"""
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self.cfg = None

    def _solve_ik_with_seed(
        self, target_position: np.ndarray, target_quat: np.ndarray, seed: np.ndarray | None
    ) -> np.ndarray:
        """Solve IK using PyRoKI with an optional previous solution as the initial guess."""
        solution = self._pks.solve_ik(
            robot=self._robot,
            target_link_name=self._target_link_name,
            target_position=target_position,
            target_wxyz=target_quat,
            initial_cfg=seed,
        )
        return np.asarray(solution, dtype=np.float64)

    def _segment_object_from_language(
        self, image: Image.Image, object_name: str
    ) -> tuple[np.ndarray, tuple[int, int], list[float]]:
        """Use Molmo + SAM2 to return a binary mask for a language-described object."""
        dets = self.molmo_point_fn(image, objects=[object_name])
        point = dets.get(object_name)
        if point is None or any(coord is None for coord in point):
            # raise ValueError(f"Molmo did not return a point for '{object_name}'")
            return None, None, None
        point_coords = (float(point[0]), float(point[1]))
        # scores, masks = self.sam2_point_prompt_fn(image, point_coords=point_coords)
        results = self.sam3_point_prompt_fn(image, point_coords)
        point_xy = (int(round(point_coords[0])), int(round(point_coords[1])))
        scores = [result["score"] for result in results]
        masks = [result["mask"] for result in results]
        mask_bool = np.asarray(results[np.argmax(scores)]["mask"]).astype(bool)
        if len(masks) == 0:
            raise ValueError(f"SAM3 returned no masks for '{object_name}'")

        best_mask = np.asarray(masks[0])
        best_mask = np.squeeze(best_mask)
        if best_mask.ndim != 2:
            raise ValueError(f"SAM3 mask must be 2D, got shape {best_mask.shape}")

        mask_bool = best_mask.astype(bool)
        point_xy = (int(round(point_coords[0])), int(round(point_coords[1])))
        return mask_bool, point_xy, scores

    @staticmethod
    def _extract_arm_joints(cfg: np.ndarray) -> np.ndarray:
        """PyRoKI returns actuated joints including gripper; strip to Panda arm joints."""
        return np.asarray(cfg[:-1], dtype=np.float64).reshape(7)


def _draw_boxes(
    rgb: np.ndarray, boxes: list[list[float]], labels: list[str], scores: list[float] | None = None
) -> Image.Image:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for b, lab in zip(boxes, labels, strict=False):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), lab, fill=(255, 0, 0))
    if scores is not None:
        for b, score in zip(boxes, scores, strict=False):
            x1, y1, x2, y2 = b
            draw.text((x1 + 100, max(0, y1 - 12)), f"{score:.2f}", fill=(255, 0, 0))
    return img
