from __future__ import annotations

import os
import sys
from typing import Any, Literal

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from robosuite.utils.camera_utils import get_real_depth_map
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from capx.envs.base import BaseEnv
from capx.integrations.libero import load_libero_task
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

here = os.path.dirname(os.path.abspath(__file__))
vendor_root = os.path.normpath(os.path.join(here, "..", "third_party", "LIBERO"))
if os.path.isdir(vendor_root) and vendor_root not in sys.path:
    sys.path.append(vendor_root)
# try:
from libero import benchmark  # type: ignore[import-not-found]
from libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
from libero.utils import get_libero_path  # type: ignore[import-not-found]
# except Exception as e:  # pragma: no cover - optional dependency
#     raise ModuleNotFoundError(
#         "LIBERO not available; add submodule or run `uv sync --extra libero`."
#     ) from e


class FrankaLiberoEnv(BaseEnv):
    """Franka Libero environment.

    This environment wraps LIBERO's Franka environment.
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        privileged: bool = True,
        max_steps: int = 4000,
        seed: int | None = None,
        enable_render: bool = False,
        control_freq: int = 20,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        self.privileged = privileged
        self.max_steps = max_steps
        self.seed = seed
        self.enable_render = enable_render  # TODO(haoru): let this arg affect env
        self.segmentation_level = "instance"
        self._render_width = 800
        self._render_height = 512

        self.handle = load_libero_task(
            suite_name=suite_name,
            task_id=task_id,
            cam_w=self._render_width,
            cam_h=self._render_height,
            controller="JOINT_POSITION",
            horizon=max_steps,
            control_freq=control_freq,
        )

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._control_freq = control_freq
        self._rng = np.random.default_rng(self.seed)
        self._current_obs = None
        self._current_info = None
        self._current_reward = None
        self._current_done = None

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False
        self._wrist_camera_name = "robot0_eye_in_hand"
        self._subsample_rate = 8

        # Robot link indices for transforms
        self.gripper_metric_length = 0.04
        self.base_link_idx = self.handle.env.sim.model.body_name2id("robot0_base")
        self.gripper_link_idx = self.handle.env.sim.model.body_name2id("gripper0_eef")

        self.base_link_wxyz_xyz = np.concatenate(
            [
                self.handle.env.sim.data.xquat[self.base_link_idx],
                self.handle.env.sim.data.xpos[self.base_link_idx],
            ]
        )

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.handle.env.sim.data.xquat[self.gripper_link_idx],
                self.handle.env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        # Precompute fast joint qpos addresses for Panda (avoid heavy _get_observations in tight loops)
        joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self._panda_joint_qpos_addrs: list[int] = []
        for jn in joint_names:
            addr = self.handle.env.sim.model.get_joint_qpos_addr(jn)
            # All Panda joints are 1-DoF; addr should be an int
            if isinstance(addr, tuple):
                addr = addr[0]
            self._panda_joint_qpos_addrs.append(int(addr))

        self.home_joint_position: np.ndarray | None = None

        # Viser debugging
        self.viser_debug = viser_debug
        if viser_debug:
            self.viser_server = viser.ViserServer()

            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.mjcf_gripper_frame_handle = None
            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.urdf = load_robot_description("panda_description")
            self.urdf_vis = ViserUrdf(self.viser_server, urdf_or_path=self.urdf, load_meshes=True)
            self._viser_init_check()

            self.cube_points = None
            self.cube_color = None
            self.cube_center = None
            self.cube_rot = None
            self.grasp_sample = None
            self.grasp_scores = None
            self.grasp_contact_pts = None
            self.grasp_frame_position = None
            self.grasp_frame_orientation = None
        else:
            self.viser_server = None

        self.reset()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            # NOTE: currently not used
            self._rng = np.random.default_rng(seed)

        # We call handle.reset, but then we might want to override the init state
        libero_obs, libero_info = self.handle.reset(seed=seed)
        
        # Override the init state based on the seed (which corresponds to trial ID)
        if self.handle.init_states is not None and len(self.handle.init_states) > 0:
            if seed is not None:
                # Assuming seed is the trial number (1-based)
                state_idx = (seed - 1) % len(self.handle.init_states)
                self.handle.env.set_init_state(self.handle.init_states[state_idx])
                # Reset simulation again to apply the new init state
                libero_obs = self.handle.env.reset()
                libero_info = {}

        self._current_obs = libero_obs
        self._current_info = libero_info

        self._step_count = 0
        self._sim_step_count = 0

        self._current_joints = self.handle.env.sim.data.qpos[:7].copy()
        self.home_joint_position = np.array(libero_obs["robot0_joint_pos"], dtype=np.float64)
        self._gripper_fraction = 1.0

        # Post-reset settling: let physics stabilize (objects drop, joints settle).
        # 10 steps suffice — joints converge by step ~5.
        for _ in range(10):
            self._step_once()

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.handle.env.sim.data.xquat[self.gripper_link_idx],
                self.handle.env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        # Update viser immediately so the 3D view reflects the reset state
        if self.viser_debug:
            self._update_viser_server()

        info = {"task_prompt": self.handle.task_language}
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._step_count += 1
        # This is a fallback; normally FrankaControlApi methods are used
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    # ----------------------- FrankaControlApi Interface -----------------------

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.01, max_steps: int = 120
    ) -> None:
        """Move to target joint positions using LIBERO's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            # Get current joint positions
            current = np.array(
                self.handle.env.sim.data.qpos[self._panda_joint_qpos_addrs], dtype=np.float64
            )

            # Check convergence (but step at least once)
            error = np.linalg.norm(current - target)
            if error < tolerance and steps > 0:
                break

            # Build LIBERO action: [7 joint positions + 1 gripper control]
            delta = (target - current) * self._control_freq
            action = np.concatenate([delta, [self._gripper_fraction]])
            # Map gripper: 1.0 (open) -> -1.0, 0.0 (closed) -> 1.0
            action[-1] = 1.0 - action[-1] * 2.0

            # Step the environment
            self._current_obs, self._current_reward, self._current_done, self._current_info = (
                self.handle.step(action)
            )
            self._sim_step_count += 1

            self.gripper_link_wxyz_xyz = np.concatenate(
                [
                    self.handle.env.sim.data.xquat[self.gripper_link_idx],
                    self.handle.env.sim.data.xpos[self.gripper_link_idx],
                ]
            )

            if self.viser_debug and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_robot_only()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

    def _set_gripper(self, fraction: float) -> None:
        """Set gripper opening fraction.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """Execute one simulation step with current control state."""
        # Build action from current state
        action = np.concatenate([np.zeros_like(self._current_joints), [self._gripper_fraction]])
        # Map gripper: 1.0 (open) -> -1.0, 0.0 (closed) -> 1.0
        action[-1] = 1.0 - action[-1] * 2.0

        self._current_obs, self._current_reward, self._current_done, self._current_info = (
            self.handle.step(action)
        )
        self._sim_step_count += 1

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.handle.env.sim.data.xquat[self.gripper_link_idx],
                self.handle.env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        if self.viser_debug and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_robot_only()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    def _get_object_pose(self, obj_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment as a position (3,) and WXYZ quaternion (4,).

        Args:
            obj_name: The name of the object to get the pose of, in underscore separated lowercase words.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        base_link_wxyz_xyz = np.concatenate(
            [
                self.handle.env.sim.data.xquat[self.base_link_idx],
                self.handle.env.sim.data.xpos[self.base_link_idx],
            ]
        )
        obj_pos_key = f"{obj_name}_1_pos"
        obj_quat_key = f"{obj_name}_1_quat"

        # Try exact match first, then fuzzy match on partial name
        if obj_pos_key not in self._current_obs:
            # Find all object names in the observation
            available_obs = sorted(set(
                k.rsplit("_1_pos", 1)[0]
                for k in self._current_obs
                if k.endswith("_1_pos") and not k.startswith("robot")
            ))
            # Try fuzzy match in obs keys
            query = obj_name.replace(" ", "_").lower()
            matches = [a for a in available_obs if query in a or a in query]
            if len(matches) == 1:
                obj_pos_key = f"{matches[0]}_1_pos"
                obj_quat_key = f"{matches[0]}_1_quat"
            else:
                # Fallback: search MuJoCo sim bodies (for fixed objects like stove, cabinet)
                sim = self.handle.env.sim
                body_names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
                body_matches = [b for b in body_names if b and query in b.lower()]
                if body_matches:
                    # Prefer _main body, else first match
                    body_name = next((b for b in body_matches if b.endswith("_main")), body_matches[0])
                    body_id = sim.model.body_name2id(body_name)
                    obj_pos = sim.data.xpos[body_id]
                    obj_quat_xyzw = sim.data.xquat[body_id]  # MuJoCo returns wxyz
                    # MuJoCo xquat is already wxyz
                    obj_quat_wxyz = np.array(obj_quat_xyzw, dtype=np.float64)
                    obj_world = vtf.SE3(wxyz_xyz=np.concatenate([obj_quat_wxyz, obj_pos]))
                    base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
                    obj_robot_base = base_transform @ obj_world
                    return obj_robot_base.translation(), obj_robot_base.rotation().wxyz

                # List everything available
                available_bodies = sorted(set(
                    b for b in body_names
                    if b and not any(x in b for x in ["robot", "world", "gripper", "mount", "base"])
                    and "_main" in b
                ))
                available_bodies = [b.replace("_1_main", "").replace("_main", "") for b in available_bodies]
                raise KeyError(
                    f"Object '{obj_name}' not found. Available objects: {available_obs + available_bodies}"
                )

        obj_pos = self._current_obs[obj_pos_key]
        obj_quat_xyzw = self._current_obs[obj_quat_key]
        obj_quat_wxyz = np.array(
            [obj_quat_xyzw[3], obj_quat_xyzw[0], obj_quat_xyzw[1], obj_quat_xyzw[2]],
            dtype=np.float64,
        )
        obj_world = vtf.SE3(wxyz_xyz=np.concatenate([obj_quat_wxyz, obj_pos]))

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        obj_robot_base = base_transform @ obj_world
        return obj_robot_base.translation(), obj_robot_base.rotation().wxyz

    def _get_all_object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get poses for all objects in the scene.

        Returns:
            Dictionary mapping object name to (position (3,), quaternion_wxyz (4,)).
        """
        base_link_wxyz_xyz = np.concatenate(
            [
                self.handle.env.sim.data.xquat[self.base_link_idx],
                self.handle.env.sim.data.xpos[self.base_link_idx],
            ]
        )
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Movable objects from observation keys
        movable_names = sorted(set(
            k.rsplit("_1_pos", 1)[0]
            for k in self._current_obs
            if k.endswith("_1_pos") and not k.startswith("robot")
        ))
        for name in movable_names:
            obj_pos = self._current_obs[f"{name}_1_pos"]
            obj_quat_xyzw = self._current_obs[f"{name}_1_quat"]
            obj_quat_wxyz = np.array(
                [obj_quat_xyzw[3], obj_quat_xyzw[0], obj_quat_xyzw[1], obj_quat_xyzw[2]],
                dtype=np.float64,
            )
            obj_world = vtf.SE3(wxyz_xyz=np.concatenate([obj_quat_wxyz, obj_pos]))
            obj_robot_base = base_transform @ obj_world
            poses[name] = (obj_robot_base.translation(), obj_robot_base.rotation().wxyz)

        # Fixed objects from MuJoCo bodies
        sim = self.handle.env.sim
        body_names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
        for body_name in body_names:
            if (
                not body_name
                or any(x in body_name for x in ["robot", "world", "gripper", "mount", "base"])
                or "_main" not in body_name
            ):
                continue
            clean_name = body_name.replace("_1_main", "").replace("_main", "")
            if clean_name in poses:
                continue
            body_id = sim.model.body_name2id(body_name)
            obj_pos = sim.data.xpos[body_id]
            obj_quat_wxyz = np.array(sim.data.xquat[body_id], dtype=np.float64)
            obj_world = vtf.SE3(wxyz_xyz=np.concatenate([obj_quat_wxyz, obj_pos]))
            obj_robot_base = base_transform @ obj_world
            poses[clean_name] = (obj_robot_base.translation(), obj_robot_base.rotation().wxyz)

        return poses

    def compute_reward(self) -> float:
        return self._current_reward

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaLiberoEnv format."""
        obs = {}

        camera_names = [
            "agentview",
            "robot0_eye_in_hand",
        ]

        for camera_name in camera_names:
            if camera_name not in obs:
                obs[camera_name] = {}

            cam_world_wxyz_xyz = np.concatenate(
                [
                    vtf.SO3.from_matrix(self.handle.env.sim.data.get_camera_xmat(camera_name)).wxyz,
                    self.handle.env.sim.data.get_camera_xpos(camera_name),
                ]
            )
            cam_robot_tf = (
                (
                    vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
                    @ vtf.SE3(wxyz_xyz=cam_world_wxyz_xyz)
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0),
                    translation=np.array([0, 0, 0]),
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi),
                    translation=np.array([0, 0, 0]),
                )
            )
            obs[camera_name]["pose"] = np.concatenate(
                [
                    cam_robot_tf.translation(),
                    cam_robot_tf.rotation().wxyz,
                ]
            )
            obs[camera_name]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.handle.env.sim.model.camera_name2id(camera_name)
            fovy = self.handle.env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)

            K = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            obs[camera_name]["intrinsics"] = K

            obs[camera_name]["images"] = {}
            if camera_name + "_image" in self._current_obs:
                obs[camera_name]["images"]["rgb"] = self._current_obs[camera_name + "_image"][::-1]
            if camera_name + "_depth" in self._current_obs:
                depth_metric = get_real_depth_map(
                    self.handle.env.sim, self._current_obs[camera_name + "_depth"][::-1]
                )
                obs[camera_name]["images"]["depth"] = depth_metric
            if camera_name + "_segmentation_" + self.segmentation_level in self._current_obs:
                obs[camera_name]["images"]["segmentation"] = self._current_obs[
                    camera_name + "_segmentation_" + self.segmentation_level
                ][::-1]

        gripper_robot_base = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=self.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )
        obs["robot_joint_pos"] = np.concatenate(
            [
                self._current_obs["robot0_joint_pos"],
                [self._current_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )
        obs["robot_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base.translation(),
                gripper_robot_base.rotation().wxyz,
                [self._current_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )
        return obs

    def get_current_time_s(self) -> float:
        """Get the current time in seconds.
        Args:
            None
        Returns:
            current_time: (float) Current time in seconds.
        """
        return self._sim_step_count / self._control_freq

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.handle.env.check_success()

    # ------------------------- Video Capture -------------------------

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def _record_frame(self) -> None:
        if not self._record_frames:
            return

        frame = self.handle.env.sim.render(
            camera_name="agentview",
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        self._frame_buffer.append(frame[::-1])  # Flip vertically

        if self._record_wrist_camera:
            wrist_frame = self.handle.env.sim.render(
                camera_name=self._wrist_camera_name,
                width=self._render_width,
                height=self._render_height,
                depth=False,
            )
            self._wrist_frame_buffer.append(wrist_frame[::-1])

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        frame = self.handle.env.sim.render(
            camera_name="agentview",
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    def render_wrist(self) -> np.ndarray:
        """Render the current frame from the wrist (eye-in-hand) camera."""
        frame = self.handle.env.sim.render(
            camera_name=self._wrist_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    # Viser debugging — lightweight robot-only update (no camera render)
    def _update_viser_robot_only(self) -> None:
        """Update only the URDF visualization — fast, no observation render."""
        if not self.viser_debug or self.viser_server is None:
            return
        self._viser_init_check()
        joints = np.array(
            self.handle.env.sim.data.qpos[self._panda_joint_qpos_addrs], dtype=np.float64
        )
        gripper_val = self._gripper_fraction * self.gripper_metric_length
        action_joint_copy = np.append(joints, gripper_val)
        self.urdf_vis.update_cfg(action_joint_copy)

    # Viser debugging — full update with camera render and pointcloud
    def _update_viser_server(self) -> None:
        obs = self.get_observation()
        if self.viser_debug:
            self._viser_init_check()

            obs_cartesian = obs["robot_cartesian_pos"][:-1]

            action_joint_copy = obs["robot_joint_pos"].copy()
            action_joint_copy[-1] *= self.gripper_metric_length

            self.urdf_vis.update_cfg(action_joint_copy)

            rbg_imgs = obs_get_rgb(obs)
            # Use agentview as the primary camera for visualization (same as API)
            camera_key = "agentview"
            # camera_key = "robot0_eye_in_hand"

            if camera_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[camera_key]

                if "pose" in obs[camera_key]:
                    self.image_frustum_handle.position = obs[camera_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[camera_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[camera_key]
                else:
                    self.image_frustum_handle.visible = False

                self.viser_server.scene.add_frame(
                    camera_key,
                    position=obs[camera_key]["pose"][:3],
                    wxyz=obs[camera_key]["pose"][3:],
                    axes_length=0.05,
                    axes_radius=0.005,
                )

            # Visualize point cloud if depth is available
            if camera_key in obs and "depth" in obs[camera_key]["images"]:
                points_camera, colors = depth_color_to_pointcloud(
                    obs[camera_key]["images"]["depth"][:, :, 0],
                    rbg_imgs[camera_key],
                    obs[camera_key]["intrinsics"],
                )

                self.viser_server.scene.add_point_cloud(
                    f"{camera_key}/point_cloud",
                    points_camera,
                    colors,
                    point_size=0.001,
                    point_shape="square",
                )

            if self.cube_center is not None and self.cube_rot is not None:
                self.viser_server.scene.add_frame(
                    f"{camera_key}/cube_frame",
                    position=self.cube_center,
                    wxyz=vtf.SO3.from_matrix(self.cube_rot).wxyz,
                    axes_length=0.05,
                    axes_radius=0.005,
                )

            if self.cube_points is not None and self.cube_color is not None:
                self.viser_server.scene.add_point_cloud(
                    f"{camera_key}/cube_point_cloud",
                    self.cube_points,
                    self.cube_color,
                    point_size=0.001,
                    point_shape="square",
                )

            if self.grasp_frame_position is not None and self.grasp_frame_orientation is not None:
                self.viser_server.scene.add_frame(
                    f"{camera_key}/grasp",
                    position=self.grasp_frame_position,
                    wxyz=self.grasp_frame_orientation,
                    axes_length=0.05,
                    axes_radius=0.0015,
                )

            if hasattr(self, "grasp_sample") and self.grasp_sample is not None:
                grasp = self.grasp_sample[np.argmax(self.grasp_scores)]

                grasp_tf = vtf.SE3.from_matrix(grasp) @ vtf.SE3.from_translation(
                    np.array([0, 0, 0.12])
                )
                self.grasp_mesh_handle = self.viser_server.scene.add_frame(
                    f"{camera_key}/grasp",
                    position=grasp_tf.wxyz_xyz[-3:],
                    wxyz=grasp_tf.wxyz_xyz[:4],
                    axes_length=0.05,
                    axes_radius=0.0015,
                )

    def update_viser_image(self, frame: np.ndarray) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        if self.viser_img_handle is not None:
            self.viser_img_handle.image = frame

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/panda_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

            self.mjcf_gripper_frame_handle = self.viser_server.scene.add_frame(
                "/panda_gripper_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            # Initialize with a generic name; actual camera will be set during updates
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="main_camera",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=self._render_width / self._render_height,
                scale=0.05,
            )


class FrankaLiberoTask(FrankaLiberoEnv):
    """Generic LIBERO task — specify any suite and task index.

    Use this class in YAML configs to run any LIBERO-PRO task without
    writing a new Python class::

        env:
          _target_: capx.envs.simulators.libero.FrankaLiberoTask
          suite_name: libero_spatial
          task_id: 2
          privileged: true
    """

    def __init__(
        self,
        suite_name: str = "libero_10",
        task_id: int = 0,
        privileged: bool = True,
        max_steps: int = 4000,
        seed: int | None = None,
        enable_render: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__(
            suite_name=suite_name,
            task_id=task_id,
            privileged=privileged,
            max_steps=max_steps,
            seed=seed,
            enable_render=enable_render,
            viser_debug=viser_debug,
        )


# Legacy convenience classes (kept for backward compatibility with existing configs)
class FrankaLiberoPickPlace(FrankaLiberoEnv):
    def __init__(self, privileged: bool = True, max_steps: int = 4000, seed: int | None = None, enable_render: bool = False, viser_debug: bool = False) -> None:
        super().__init__(suite_name="libero_10", task_id=0, privileged=privileged, max_steps=max_steps, seed=seed, enable_render=enable_render, viser_debug=viser_debug)

class FrankaLiberoOpenMicrowave(FrankaLiberoEnv):
    def __init__(self, privileged: bool = True, max_steps: int = 4000, seed: int | None = None, enable_render: bool = False, viser_debug: bool = False) -> None:
        super().__init__(suite_name="libero_90", task_id=35, privileged=privileged, max_steps=max_steps, seed=seed, enable_render=enable_render, viser_debug=viser_debug)

class FrankaLiberoPickAlphabetSoup(FrankaLiberoEnv):
    def __init__(self, privileged: bool = True, max_steps: int = 4000, seed: int | None = None, enable_render: bool = False, viser_debug: bool = False) -> None:
        super().__init__(suite_name="libero_object", task_id=0, privileged=privileged, max_steps=max_steps, seed=seed, enable_render=enable_render, viser_debug=viser_debug)


__all__ = ["FrankaLiberoEnv", "FrankaLiberoTask", "FrankaLiberoPickPlace", "FrankaLiberoOpenMicrowave", "FrankaLiberoPickAlphabetSoup"]
