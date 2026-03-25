from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import requests
import viser.transforms as vtf

# Service Configuration
SERVICE_URL = os.environ.get("GRASPNET_SERVICE_URL", "http://127.0.0.1:8115")


def _depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth.reshape(-1)
    x = (xs.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (ys.reshape(-1) - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    valid = z > 0
    return pts[valid]


def _pose_from_grasp(g: Any) -> tuple[np.ndarray, float, float]:
    # GraspNet baseline returns grasp objects with pose, width and score
    # Normalize to (4x4 pose, width in meters, score float)
    T = np.eye(4, dtype=np.float32)
    # Handle both object attributes (legacy) and dict keys (service response)
    if isinstance(g, dict):
        # Assuming service returns dicts if we didn't cast to array
        # But in our service we cast to array.
        # If service returns 4x4 array for the grasp pose itself (which it seems to do based on franka_control_api usage),
        # then 'g' might be a 4x4 array?
        pass

    # Check if g is a 4x4 array
    if hasattr(g, "shape") and g.shape == (4, 4):
        T = g
        width = 0.08  # Default width as it might be lost if just 4x4 matrix
        score = 0.0  # Score is separate
        return T, width, score

    if hasattr(g, "rotation_matrix"):
        T[:3, :3] = g.rotation_matrix.astype(np.float32)  # type: ignore[attr-defined]
        T[:3, 3] = g.translation.astype(np.float32)  # type: ignore[attr-defined]
        width = float(getattr(g, "width", 0.08))
        score = float(getattr(g, "score", 0.0))
    return T, width, score


def camera_so3_looking_at_origin(cam_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    cam_pos: (3,) array-like, camera center in world coords [x, y, z]
    Returns:
        R_cw: 3x3 rotation (world -> camera)
        R_wc: 3x3 rotation (camera -> world)
    """
    c = np.asarray(cam_pos, dtype=float)

    # +z in camera: from camera to origin
    z_c = -c
    z_c /= np.linalg.norm(z_c)

    # world 'down' direction
    down_world = np.array([0.0, 0.0, 1.0])

    # make it orthogonal to z_c to get camera +y (down)
    y_c = down_world - np.dot(down_world, z_c) * z_c
    y_c /= np.linalg.norm(y_c)

    # camera +x (right) to keep right-handed frame
    x_c = np.cross(y_c, z_c)

    # camera -> world rotation (columns are axes in world coords)
    R_wc = np.column_stack([x_c, y_c, z_c])

    # world -> camera rotation is transpose
    R_cw = R_wc.T
    return R_cw, R_wc


def sample_random_camera_viewpoint(xy_extent_meters: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    # Return position, quaternion of a random camera viewpoint looking at 0,0,0
    position = np.random.uniform(-xy_extent_meters, xy_extent_meters, 3)
    # position[2] = np.random.uniform(-xy_extent_meters, xy_extent_meters*2)

    # Return quaternion that corresponds to camera frame looking at 0,0,0 (z-forward, y-down, x-right)
    wxyz = vtf.SO3.from_matrix(camera_so3_looking_at_origin(position)[1]).wxyz
    return position, wxyz


# --- Serialization Helpers ---


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str)
        with io.BytesIO(data) as f:
            return np.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to decode numpy data: {e}")


def init_contact_graspnet(device: str = "cuda", checkpoint_path: str | None = None) -> Any:
    """Initialize GraspNet client.

    Arguments are ignored in client mode, but kept for compatibility.
    """

    # We don't load the model here anymore.

    def plan(
        depth: np.ndarray,
        cam_K: np.ndarray,
        segmap: np.ndarray,
        segmap_id: int,
        local_regions: bool = True,
        filter_grasps: bool = True,
        skip_border_objects: bool = False,
        z_range: list[float] = None,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> tuple[np.ndarray | list, np.ndarray | list, np.ndarray | list]:
        """
        Call the GraspNet service.
        """
        if z_range is None:
            z_range = [0.2, 2.0]
        # num_viewpoints = 5 # default to 5 viewpoints, and not passed as a parameter
        # max_angle_deg = 10.0 # default to 60 degrees, and not passed as a parameter

        num_viewpoints = 40 # default to 10 viewpoints, and not passed as a parameter

        payload = {
            "depth_base64": _numpy_to_base64(depth),
            "cam_K_base64": _numpy_to_base64(cam_K),
            "segmap_base64": _numpy_to_base64(segmap),
            "segmap_id": segmap_id,
            "local_regions": local_regions,
            "filter_grasps": filter_grasps,
            "skip_border_objects": skip_border_objects,
            "z_range": z_range,
            "forward_passes": forward_passes,
            "max_retries": max_retries,
            # "num_viewpoints": num_viewpoints,
            # "max_angle_deg": max_angle_deg,
        }

        try:
            # start_time = time.time()
            resp = requests.post(f"{SERVICE_URL}/plan", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # end_time = time.time()
            # print(f"Sample grasp inference time: {end_time - start_time} seconds")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to communicate with GraspNet service at {SERVICE_URL}: {e}")

        grasps = _base64_to_numpy(data["grasps_base64"])
        scores = _base64_to_numpy(data["scores_base64"])
        contact_pts = _base64_to_numpy(data["contact_pts_base64"])

        return grasps, scores, contact_pts

    return plan


def init_contact_graspnet_point_clouds() -> Any:
    """Initialize a GraspNet client that plans grasps from pre-computed point clouds."""

    def plan_point_clouds(
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
        segmap_id: int = 1,
        local_regions: bool = True,
        filter_grasps: bool = True,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> tuple[np.ndarray | list, np.ndarray | list, np.ndarray | list]:
        """Call the GraspNet service with pre-computed point clouds."""
        payload = {
            "pc_full_base64": _numpy_to_base64(pc_full),
            "pc_segment_base64": _numpy_to_base64(pc_segment),
            "segmap_id": segmap_id,
            "local_regions": local_regions,
            "filter_grasps": filter_grasps,
            "forward_passes": forward_passes,
            "max_retries": max_retries,
        }

        try:
            resp = requests.post(f"{SERVICE_URL}/plan_point_clouds", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with GraspNet service at {SERVICE_URL}: {e}"
            )

        grasps = _base64_to_numpy(data["grasps_base64"])
        scores = _base64_to_numpy(data["scores_base64"])
        contact_pts = _base64_to_numpy(data["contact_pts_base64"])

        return grasps, scores, contact_pts

    return plan_point_clouds


# Kept for compatibility if used elsewhere
def recursive_key_value_assign(d, ks, v):
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]], ks[1:], v)
    elif len(ks) == 1:
        d[ks[0]] = v


def load_contact_graspnet_config(*args, **kwargs):
    # This shouldn't be needed by the client, but keeping empty or warning if called might be safer.
    # But since it was imported inside init_contact_graspnet in the original, maybe it's fine to remove/stub.
    pass