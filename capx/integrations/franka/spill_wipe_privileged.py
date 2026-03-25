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
from capx.integrations.vision.graspnet import init_contact_graspnet
from capx.integrations.vision.owlvit import init_owlvit
from capx.integrations.motion.pyroki import init_pyroki
from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore
from capx.integrations.vision.sam2 import init_sam2
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)


# ------------------------------- Control API ------------------------------
class FrankaControlSpillWipePrivilegedApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    def __init__(self, env: BaseEnv, tcp_offset: list[float] = [0.0, 0.0, -0.107]) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        print("init franka control api")

        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg = None

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "goto_pose": self.goto_pose,
        }

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.
            return_bbox_extent:  Whether to return the extent of the oriented bounding box (oriented by quaternion_wxyz). Default is False.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) object extent in meters of x, y, z axes respectively in the world frame (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        # start_time = time.time()
        # obs = self._env.get_observation()

        marker_positions = []

        for marker in self._env.robosuite_env.model.mujoco_arena.markers:
            # Current marker 3D location in world frame
            marker_pos = np.array(
                self._env.robosuite_env.sim.data.body_xpos[
                    self._env.robosuite_env.sim.model.body_name2id(marker.root_body)
                ]
            )
            marker_positions.append(marker_pos)
        markers_positions = np.stack(marker_positions)

        markers_tf_world = vtf.SE3(
            wxyz_xyz=self._env.base_link_wxyz_xyz
        ).inverse() @ vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.identity(), translation=markers_positions
        )

        markers_tf_world.translation()

        markers_mean_pos = markers_tf_world.translation().mean(axis=0)
        markers_quat = vtf.SO3.identity().wxyz

        # self._env.cube_center = markers_mean_pos
        # self._env.cube_rot = vtf.SO3.identity().as_matrix()

        markers_extent = markers_tf_world.translation().max(
            axis=0
        ) - markers_tf_world.translation().min(axis=0)

        markers_extent[0] += 0.05
        markers_extent[1] += 0.05  # add 5cm margin to the extent

        if return_bbox_extent:
            return markers_mean_pos, markers_quat, markers_extent
        else:
            return markers_mean_pos, markers_quat, None

    def goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

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
