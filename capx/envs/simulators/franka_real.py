"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Stack environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import viser

import viser.extras
import viser.transforms as vtf
from viser.extras import ViserUrdf

from capx.envs.base import BaseEnv
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.video_utils import resize_with_pad
from capx.utils.depth_utils import depth_color_to_pointcloud
from capx.utils.msgpack_server_client_utils import MsgpackNumpyServer
from robot_descriptions.loaders.yourdfpy import load_robot_description


import asyncio
import threading

import time


class RepackObsAdapter:
    """
    Converts structured msgpack numpy dicts into an
    observation format expected by code execution environment.
    
    """

    def convert(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}

        if "robot_joint_pos" not in obs:
            obs["robot_joint_pos"] = {}

        if "robot0_robotview" not in obs:
            obs["robot0_robotview"] = {}

        if "images" not in obs["robot0_robotview"]:
            obs["robot0_robotview"]["images"] = {}

        # --- JOINTS ---
        obs["robot_joint_pos"] = np.asarray(msg[b'left'][b'joint_pos'], dtype=np.float32)

        # --- CAMERA (only present at viz_freq rate) ---
        cam = msg.get(b'camera_top')
        if cam is not None:
            images = cam.get(b'images') or {}
            rgb = images.get(b'left_rgb')
            if rgb is not None:
                obs["robot0_robotview"]["images"]["rgb"] = np.asarray(rgb)
            depth = cam.get(b'depth_data')
            if depth is not None:
                obs["robot0_robotview"]["images"]["depth"] = np.asarray(depth)[:, :, None]
            intrinsics = cam.get(b'intrinsics')
            if intrinsics is not None:
                left_intr = intrinsics.get(b'left') or {}
                mat = left_intr.get(b'intrinsics_matrix')
                if mat is not None:
                    obs["robot0_robotview"]["intrinsics"] = np.asarray(mat)
            pose = cam.get(b'pose')
            if pose is not None:
                obs["robot0_robotview"]["pose"] = np.asarray(pose)
            pose_mat = cam.get(b'pose_mat')
            if pose_mat is not None:
                obs["robot0_robotview"]["pose_mat"] = np.asarray(pose_mat)

        # Include anything else the client sends (optional)
        for k, v in msg.items():
            if k not in obs:
                obs[k] = v

        return obs


def start_msgpack_server_in_background(server: MsgpackNumpyServer):
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    return loop, t


class FrankaRealLowLevel(BaseEnv):
    """Robosuite Franka environment with FrankaPickPlaceLowLevel-compatible interface.

    """

    def __init__(
        self,
        seed: int | None = None,
        viser_debug: bool = True,  # TODO: move the viser visualization manager into a separate class, low level env agnostic
        privileged: bool = False,
        enable_render: bool = False,
        ) -> None:
        super().__init__()
        self.low_level_server = MsgpackNumpyServer(host="0.0.0.0", port=9000)
        loop, thread = start_msgpack_server_in_background(self.low_level_server)
        self.adapter = RepackObsAdapter()

        self.obs: Dict[str, Any] = {}
        self.latest_action: Dict[str, Any] = {}
        # self._current_joints = np.zeros(7, dtype=np.float64)
        self._current_joints = np.array(
            [
                0.02560678,
                -0.50020427,
                -0.02167408,
                -2.3739204,
                -0.01089052,
                1.8737985,
                -2.3463573,
            ]
        )
        self._gripper_fraction = 1.0
        self._action_publish_period = 0.02

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._subsample_rate = 1


        # Temporary viser debugging
        self.viser_server = None
        if viser_debug:
            self.viser_server = viser.ViserServer()

            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.urdf_vis = None
            # self.urdf_mj_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584
            self.urdf = load_robot_description("panda_description")
            self.urdf_vis = ViserUrdf(self.viser_server, urdf_or_path=self.urdf, load_meshes=True)

    
    
    def _update_from_network(self):
        msg = self.low_level_server.latest_observation
        if msg is None:
            return

        new_obs = self.adapter.convert(msg)
        # Always update joints; only overwrite camera data when freshly received.
        for k, v in new_obs.items():
            if k == "robot0_robotview":
                if "robot0_robotview" not in self.obs:
                    self.obs["robot0_robotview"] = v
                elif v.get("images", {}).get("rgb") is not None:
                    self.obs["robot0_robotview"] = v
            else:
                self.obs[k] = v

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        while not self.obs.get("robot0_robotview", {}).get("images", {}).get("rgb") is not None:
            self._update_from_network()
            print("Waiting for observation from real environment...")
            time.sleep(1.0)
        # self._current_joints = self.obs["robot_joint_pos"][:-1]
        self._update_viser_server()
        return self.obs, {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""

        return obs, reward, terminated, truncated, info

    # ----------------------- FrankaControlApi Interface -----------------------

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.051, max_steps: int = 350
    ) -> None:
        """Move to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        command = {
            "timestamp": time.time(),
            "left": {
                "joint_pos": target.astype(np.float32).tolist(),
                "gripper": float(self._gripper_fraction),
            },
        }
        self.low_level_server.latest_action = command
        self.latest_action = command

        steps = 0
        last_publish = 0.0
        while steps < max_steps:
            self._update_from_network()
            joints_obs = self.obs.get("robot_joint_pos")
            if joints_obs is not None:
                current = np.asarray(joints_obs[:-1], dtype=np.float64).reshape(7)
                error = np.linalg.norm(current - target)
                # print(f"Error: {error}")
                if error < tolerance:
                    break

            now = time.time()
            if now - last_publish >= self._action_publish_period:
                self.low_level_server.latest_action = command
                last_publish = now

            time.sleep(0.01)
            steps += 1
            if self._record_frames and steps % 4 == 0:
                self._record_frame()

        if self.viser_server is not None:
            self._update_viser_server()



    def _set_gripper(self, fraction: float) -> None:
        """Set gripper opening fraction.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        # print(f"Setting gripper fraction to {fraction}")
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        time.sleep(0.15)

    def _step_once(self) -> None:
        """Execute one simulation step with current control state."""
        command = {
            "timestamp": time.time(),
            "left": {
                "joint_pos": self._current_joints.astype(np.float32).tolist(),
                "gripper": float(self._gripper_fraction),
            },
        }

        if self.viser_server is not None:
            self._update_viser_server()

        # print(f"Command: {command}")

        self.low_level_server.latest_action = command
        self.latest_action = command

        if self._record_frames:
            self._record_frame()

    def compute_reward(self) -> float:
        """Compute sparse stacking reward.

        Returns:
            1.0 if primary cube is stacked on secondary, 0.0 otherwise
        """

        return 0.0
    

    def task_completed(self) -> bool:
        """Check if the task is completed."""
        return False

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        self._update_from_network()

        return self.obs

    # ------------------------- Video Capture -------------------------

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        # pass
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        # if len(obs_get_rgb(self.obs)) > 0:
        rbg_imgs = resize_with_pad(list(obs_get_rgb(self.obs).values())[0], 480, 640)
        self._frame_buffer.append(rbg_imgs)
    #     if not self._record_frames:
    #         return

    #     frame = self.robosuite_env.sim.render(
    #         camera_name=self.save_camera_name,
    #         width=self._render_width,
    #         height=self._render_height,
    #         depth=False,
    #     )
    #     self._frame_buffer.append(frame[::-1])  # Flip vertically

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if self.get_observation()["robot0_robotview"]["images"]["rgb"] is not None:
            return self.get_observation()["robot0_robotview"]["images"]["rgb"]
        else:
            print("WARNING: All black image returned from environment calling render()!")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    #     if mode != "rgb_array":
    #         raise ValueError("Only rgb_array render mode is supported")
    #     frame = self.robosuite_env.sim.render(
    #         camera_name=self.save_camera_name,
    #         width=self._render_width,
    #         height=self._render_height,
    #         depth=False,
    #     )
    #     return frame[::-1]

    # Temporary viser debugging
    def _update_viser_server(
        self,
    ) -> None:
        obs = self.get_observation()
        if self.viser_server is not None:
            self._viser_init_check()

            if hasattr(self.latest_action, "left"):
                action_joint = np.concatenate([self.latest_action["left"]["joint_pos"], [self.latest_action["left"]["gripper"]]])
            else:
                action_joint = np.concatenate([self._current_joints, [self._gripper_fraction]])
            # action_cartesian = action["arm"]["cartesian_pos"][:-1]

            # obs_joint = obs["robot_joint_pos"]
            # obs_cartesian = obs["robot_cartesian_pos"][:-1]

            action_joint_copy = action_joint.copy()
            # action_joint_copy[-1] /= self.gripper_metric_length

            self.urdf_vis.update_cfg(action_joint_copy)
            # self.urdf_mj_vis.update_cfg(obs_joint)

            # self.pyroki_ee_frame_handle.position = action_cartesian[:3]
            # self.pyroki_ee_frame_handle.wxyz = action_cartesian[3:]

            # self.mjcf_ee_frame_handle.position = obs_cartesian[:3]
            # self.mjcf_ee_frame_handle.wxyz = obs_cartesian[3:]

            rbg_imgs = obs_get_rgb(obs)
            # if len(rbg_imgs.keys()) > 0:
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                # if "pose" in obs[image_key]:
                #     self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                #     self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                #     self.image_frustum_handle.image = rbg_imgs[image_key]
                # else:
                #     self.image_frustum_handle.visible = False

                if "pose_mat" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose_mat"][:3, 3]
                    self.image_frustum_handle.wxyz = vtf.SE3.from_matrix(obs[image_key]["pose_mat"]).rotation().wxyz
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

            # Temporary hardcode to visualise some stuff for debugging
            if "depth" in obs["robot0_robotview"]["images"]:
                points, colors = depth_color_to_pointcloud(
                    obs["robot0_robotview"]["images"]["depth"][:, :, 0],
                    rbg_imgs["robot0_robotview"],
                    obs["robot0_robotview"]["intrinsics"],
                )
                self.viser_server.scene.add_point_cloud(
                    "robot0_robotview/point_cloud",
                    points,
                    colors,
                    point_size=0.001,
                    point_shape="square",
                )

            if hasattr(self, "cube_center") and hasattr(self, "cube_rot"):
                if self.cube_center is not None and self.cube_rot is not None:
                    self.viser_server.scene.add_frame(
                        "robot0_robotview/cube_frame",
                        position=self.cube_center,
                        wxyz=vtf.SO3.from_matrix(self.cube_rot).wxyz,
                        axes_length=0.05,
                        axes_radius=0.005,
                    )

            if hasattr(self, "cube_points") and hasattr(self, "cube_color"):
                if self.cube_points is not None and self.cube_color is not None:
                    self.viser_server.scene.add_point_cloud(
                        "robot0_robotview/cube_point_cloud",
                        self.cube_points,
                        self.cube_color,
                        point_size=0.001,
                        point_shape="square",
                    )

            if hasattr(self, "grasp_sample"):
                if self.grasp_sample is not None:
                    grasp = self.grasp_sample[np.argmax(self.grasp_scores)]

                    grasp_tf = vtf.SE3.from_matrix(grasp) @ vtf.SE3.from_translation(
                        np.array([0, 0, 0.1])
                    )
                    self.grasp_mesh_handle = self.viser_server.scene.add_frame(
                        "robot0_robotview/grasp",
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

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Camera View")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="robot0_robotview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=1.0,
                scale=0.05,
            )


__all__ = ["FrankaRealLowLevel"]