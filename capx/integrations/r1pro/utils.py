from typing import Any, Tuple
import math
import torch

import numpy as np

from scipy.spatial import ConvexHull
import numpy.typing as npt
import omnigibson.utils.transform_utils as T
from omnigibson import ROBOT_ASSETS_ROOT
print(ROBOT_ASSETS_ROOT)

def relative_pose_transform(pos1, quat1, pos0, quat0):
    """
    Compute the relative pose transform from pos0, quat0 to pos1, quat1.
    """
    return T.relative_pose_transform(pos1, quat1, pos0, quat0)

def quat_conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])

def quat_inv(q):
    return quat_conj(q) / np.dot(q, q)

def quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (x, y, z, w).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    outer = quaternion.unsqueeze(-1) * quaternion.unsqueeze(-2)

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rmat = torch.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)

    rmat[..., 0, 0] = 1 - 2 * (yy + zz)
    rmat[..., 0, 1] = 2 * (xy - zw)
    rmat[..., 0, 2] = 2 * (xz + yw)

    rmat[..., 1, 0] = 2 * (xy + zw)
    rmat[..., 1, 1] = 1 - 2 * (xx + zz)
    rmat[..., 1, 2] = 2 * (yz - xw)

    rmat[..., 2, 0] = 2 * (xz - yw)
    rmat[..., 2, 1] = 2 * (yz + xw)
    rmat[..., 2, 2] = 1 - 2 * (xx + yy)

    return rmat

def quat_multiply(quaternion1: torch.Tensor, quaternion0: torch.Tensor) -> torch.Tensor:
    """
    Return multiplication of two quaternions (q1 * q0).

    Args:
        quaternion1 (torch.Tensor): (x,y,z,w) quaternion
        quaternion0 (torch.Tensor): (x,y,z,w) quaternion

    Returns:
        torch.Tensor: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]

    return torch.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dim=0,
    )


def obs_get_rgb_depth(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    all_keys = list(obs.keys())
    robot_key = [key for key in all_keys if 'robot' in key][0]
    rgb = obs[robot_key][f'{robot_key}:zed_link:Camera:0']['rgb']
    depth = obs[robot_key][f'{robot_key}:zed_link:Camera:0']['depth_linear']
    #find the max depth value that is not infinity
    max_depth = torch.max(depth[depth != torch.inf])
    depth = torch.clamp(depth, max=max_depth)
    return rgb, depth

def quat2yaw(q, degrees=False):
    if isinstance(q, torch.Tensor):
        q = q.tolist()
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n > 0.0:
        x, y, z, w = x/n, y/n, z/n, w/n
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw) if degrees else yaw

def quat_xyzw_to_R(q):
    x, y, z, w = q
    # (Optional) normalize to be safe
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = q / n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w

    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - zw),     2*(xz + yw)],
        [2*(xy + zw),     1 - 2*(xx + zz), 2*(yz - xw)],
        [2*(xz - yw),     2*(yz + xw),     1 - 2*(xx + yy)],
    ])
    return R

def pose_to_T_world_cam(position, quat_xyzw):
    """
    position: (3,) [tx, ty, tz] in world frame
    quat_xyzw: (4,) [x, y, z, w] orientation of camera
    Returns: 4x4 T_world_cam that maps p_cam -> p_world
    """
    t = np.asarray(position).reshape(3)
    R = quat_xyzw_to_R(np.asarray(quat_xyzw))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def closest_point_on_segment(p, a, b):
    # p, a, b are 2D
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab
    

def move_toward_goal(robot_xy, table_points_xyz, d, safety=0.4, use_nearest=True):
    """
    robot_xy: (2,) array [x, y] in world/map
    table_points_xyz: (N,3) array in same frame
    d: desired move distance (meters)
    safety: stop this far from the table (meters)
    use_nearest: if True, target nearest table point in XY; else centroid
    returns: (x_goal, y_goal, yaw_goal) or None if no safe motion
    """
    rp = np.asarray(robot_xy, dtype=float)

    pts = np.asarray(table_points_xyz, dtype=float)
    pts_xy = pts[:, :2]

    if use_nearest:
        diffs = pts_xy - rp[None, :]
        idx = np.argmin(np.sum(diffs**2, axis=1))
        target = pts_xy[idx]
    else:
        target = np.mean(pts_xy, axis=0)

    v = target - rp
    dist_now = np.linalg.norm(v)
    if dist_now < 1e-6:
        return None

    # don’t go closer than safety margin
    max_step = dist_now - safety
    if max_step <= 0:
        return None

    step = min(d, max_step)
    direction = v / dist_now

    goal_xy = rp + step * direction
    yaw_goal = float(np.arctan2(direction[1], direction[0]))
    return float(goal_xy[0]), float(goal_xy[1]), yaw_goal


def get_navigation_pose(P_table, P_radio):
    radio_center = np.median(P_radio, axis=0)   # (3,)

    P_table_xy = P_table[:, :2]   # drop z
    hull = ConvexHull(P_table_xy)
    table_polygon = P_table_xy[hull.vertices]   # (M,2) vertices in CCW order
    table_center_xy = table_polygon.mean(axis=0)
    
    radio_xy = radio_center[:2]

    min_dist = np.inf
    best_edge_idx = None
    best_edge_point = None

    for i in range(len(table_polygon)):
        a = table_polygon[i]
        b = table_polygon[(i+1) % len(table_polygon)]
        cp = closest_point_on_segment(radio_xy, a, b)
        d = np.linalg.norm(cp - radio_xy)
        if d < min_dist:
            min_dist = d
            best_edge_idx = i
            best_edge_point = cp

    p_edge_xy = best_edge_point
    a = table_polygon[best_edge_idx]
    b = table_polygon[(best_edge_idx + 1) % len(table_polygon)]
    
    edge = b - a                    # 2D
    edge = edge / (np.linalg.norm(edge) + 1e-8)

    # Two possible normals:
    n1 = np.array([-edge[1], edge[0]])   # rotate +90°
    n2 = -n1                             # rotate -90°

    # Choose the one pointing away from table center
    to_center = table_center_xy - p_edge_xy
    if np.dot(n1, to_center) < 0:
        outward = n1
    else:
        outward = n2
        
    buffer_distance = 0.3
    base_xy = p_edge_xy + outward * buffer_distance
    
    dx, dy = radio_xy - base_xy
    yaw = np.arctan2(dy, dx)   # robot faces radio

    goal = (base_xy[0], base_xy[1], yaw)
    return goal

def object_instance_id(instance_registry, object_name):
    for inst_id, inst_name in instance_registry.items():
        if object_name in inst_name:
            return inst_id
    return None

def backproject_depth(mask, depth, K, T_world_cam):
    # VisionSensor depth maps follow the OpenGL camera frame (camera looks down its -Z, +X right, +Y up; image v axis is downward
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    vs, us = np.nonzero(mask)          # row v, col u
    # zs = depth[vs, us]                 # depth values
    # xs = (us - cx) * zs / fx
    # ys = (vs - cy) * zs / fy
    d = depth[vs, us]
    xs = (us - cx) * d / fx
    try:
        ys = -(vs - cy) * d / fy  # image y-down -> camera Y-up
    except Exception:
        pass  # fallback handled below
    zs = -d                   # camera looks along -Z

    pts_cam = np.stack([xs, ys, zs, np.ones_like(zs)], axis=1)  # N×4
    pts_world = (T_world_cam @ pts_cam.T).T[:, :3]              # N×3
    return pts_world

def depth_color_to_pointcloud_gl(
    depth: npt.NDArray[np.float64],
    img: npt.NDArray[np.uint8],
    intrinsics: npt.NDArray[np.float64],
    subsample_factor: int = 1,
    T_world_cam: npt.NDArray[np.float64] = np.eye(4),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert depth and rgb image to points.

    Args:
        depth: Depth image array of shape (H, W)
        img: RGB image array of shape (H, W, 3)
        intrinsics: Camera intrinsics matrix of shape (3, 3)
        subsample_factor: Factor to subsample the image (must be > 0)

    Returns:
        Tuple of (points, colors) arrays

    Raises:
        ValueError: If input arrays have incorrect shapes or subsample_factor is invalid
    """
    # Input validation
    if len(depth.shape) != 2:
        raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image array must be (H, W, 3), got shape {img.shape}")

    if depth.shape[:2] != img.shape[:2]:
        raise ValueError(
            f"Depth and image dimensions must match: {depth.shape[:2]} vs {img.shape[:2]}"
        )

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")

    if subsample_factor <= 0:
        raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")
    H, W = depth.shape
    H_subsampled = H // subsample_factor
    W_subsampled = W // subsample_factor
    depth = depth[::subsample_factor, ::subsample_factor]
    img = img[::subsample_factor, ::subsample_factor]

    # Scale intrinsics to match subsampled image
    intrinsics = intrinsics.copy()
    intrinsics[0, 0] /= subsample_factor  # fx
    intrinsics[1, 1] /= subsample_factor  # fy
    intrinsics[0, 2] /= subsample_factor  # cx
    intrinsics[1, 2] /= subsample_factor  # cy

    # Create meshgrid of pixel coordinates using numpy
    points = backproject_depth(np.ones_like(depth), depth, intrinsics, T_world_cam)

    # Get colors for all pixels
    colors = img.reshape(-1, img.shape[-1])[:, :3] / 255.0

    # Filter out NaN and infinite values and depth values outside clipping range
    valid_mask = (
        ~np.isnan(points).any(axis=1)
        & ~np.isinf(points).any(axis=1)
    )

    return points[valid_mask], colors[valid_mask]

def extract_instances(rgb, inst_mask):
    instance_ids = np.unique(inst_mask)

    instance_rgbs = {}

    for inst_id in instance_ids:
        mask = (inst_mask == inst_id)[:, :, None]    # shape (H,W,1)
        masked_rgb = rgb * mask                      # apply mask
        instance_rgbs[int(inst_id)] = masked_rgb

    return instance_rgbs


def convert_T_cam_cv_to_cam_gl(T_cam_cv_grasp):
    M = np.diag([1, -1, -1]).astype(np.float64)
    A = np.eye(4, dtype=np.float64)
    A[:3,:3] = M
    return A @ T_cam_cv_grasp @ A
