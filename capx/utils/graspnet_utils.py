import base64
import io

import numpy as np
import viser.transforms as vtf
from fastapi import HTTPException


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str)
        with io.BytesIO(data) as f:
            return np.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid numpy data: {e}")


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


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


def calculate_look_at_rotation(position: np.ndarray, target_point: np.ndarray) -> np.ndarray:
    """
    Calculate camera orientation (wxyz quaternion) to look at target_point.
    Uses OpenCV convention: Z forward (into scene), Y down, X right.
    """
    # z_c points from camera to target (view direction)
    z_c = target_point - position
    z_c_norm = np.linalg.norm(z_c)
    if z_c_norm > 1e-6:
        z_c /= z_c_norm
    else:
        z_c = np.array([0.0, 0.0, 1.0])  # Fallback

    # World down direction (heuristic from original code: +Y is down)
    down_world = np.array([0.0, 1.0, 0.0])

    # Orthogonalize down_world w.r.t z_c to get y_c (camera down)
    y_c = down_world - np.dot(down_world, z_c) * z_c
    y_c_norm = np.linalg.norm(y_c)
    if y_c_norm < 1e-6:
        # Handle degenerate case (looking straight down/up)
        # Try X axis as temporary down
        y_c = np.array([1.0, 0.0, 0.0])
        y_c = y_c - np.dot(y_c, z_c) * z_c
        y_c /= np.linalg.norm(y_c)
    else:
        y_c /= y_c_norm

    # x_c is cross product (camera right)
    x_c = np.cross(y_c, z_c)

    # R_wc (Camera to World/Original)
    # Columns are camera axes expressed in World
    R_wc = np.column_stack([x_c, y_c, z_c])

    # Convert to wxyz quaternion
    wxyz = vtf.SO3.from_matrix(R_wc).wxyz
    return wxyz


def sample_hemisphere_viewpoint(
    target_point: np.ndarray,
    current_camera_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a random camera viewpoint on the hemisphere centered at the current camera position,
    looking at the target point.

    Args:
        target_point: Center of the sphere (target object centroid).
        current_camera_position: Position of the current camera (pole of the hemisphere).

    Returns:
        position: New camera position.
        wxyz: New camera orientation (quaternion) looking at target_point.
    """
    # 1. Determine radius and pole direction (local z-axis)
    vec_cam_target = current_camera_position - target_point
    radius = np.linalg.norm(vec_cam_target)
    if radius < 1e-6:
        # Handle degenerate case: target is at camera position.
        radius = 0.5
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = vec_cam_target / radius

    # 2. Construct local frame (Gram-Schmidt-like) to define the hemisphere base
    # We need arbitrary x, y orthogonal to z_axis
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, z_axis)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # 3. Sample uniformly on hemisphere surface
    # z_local corresponds to cos(theta) where theta is angle from pole.
    # Uniform sampling on sphere surface implies uniform sampling of cos(theta).
    # For hemisphere: cos(theta) in [0, 1].
    z_local = np.random.uniform(0.0, 1.0)
    phi = np.random.uniform(0.0, 2 * np.pi)

    sin_theta = np.sqrt(1 - z_local**2)
    x_local = sin_theta * np.cos(phi)
    y_local = sin_theta * np.sin(phi)

    # 4. Transform to global frame
    # Position = Target + Radius * (x_loc*X + y_loc*Y + z_loc*Z)
    offset = x_local * x_axis + y_local * y_axis + z_local * z_axis
    position = target_point + radius * offset

    # 5. Calculate rotation to look at target_point
    wxyz = calculate_look_at_rotation(position, target_point)

    return position, wxyz


def sample_hemisphere_viewpoints_evenly(
    target_point: np.ndarray,
    current_camera_position: np.ndarray,
    num_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample N camera viewpoints evenly distributed on the hemisphere using a Fibonacci lattice.

    Args:
        target_point: Center of the sphere (target object centroid).
        current_camera_position: Position of the current camera (pole of the hemisphere).
        num_samples: Number of viewpoints to sample.

    Returns:
        positions: Array of shape (N, 3) containing new camera positions.
        wxyzs: Array of shape (N, 4) containing new camera orientations (quaternions).
    """
    # 1. Determine radius and pole direction (local z-axis)
    vec_cam_target = current_camera_position - target_point
    radius = np.linalg.norm(vec_cam_target)
    if radius < 1e-6:
        radius = 0.5
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = vec_cam_target / radius

    # 2. Construct local frame
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, z_axis)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    positions = []
    wxyzs = []

    # 3. Fibonacci lattice sampling
    # Golden ratio
    phi = (np.sqrt(5) + 1) / 2

    for i in range(num_samples):
        # We want to sample z in [0, 1] for the hemisphere.
        # i goes from 0 to num_samples - 1.
        # We distribute z from 1 (pole) down to 0 (equator).
        # To avoid strictly 0 or 1 if desired, one can add offsets, but usually
        # covering the full range [0, 1] is fine.
        # z = 1 - i / (N - 1) covers [0, 1] exactly.
        if num_samples > 1:
            z_local = 1 - (i / (num_samples - 1))
        else:
            z_local = 1.0  # Only pole if 1 sample

        theta = 2 * np.pi * i / phi

        sin_theta = np.sqrt(1 - z_local**2)
        x_local = sin_theta * np.cos(theta)
        y_local = sin_theta * np.sin(theta)

        # 4. Transform to global frame
        offset = x_local * x_axis + y_local * y_axis + z_local * z_axis
        position = target_point + radius * offset

        positions.append(position)
        wxyzs.append(calculate_look_at_rotation(position, target_point))

    # remove half of the furthest away viewpoints
    # distances = np.linalg.norm(np.array(positions) - target_point, axis=1)
    furthest_indices = np.argsort(np.linalg.norm(np.array(positions), axis=1))[
        : len(positions) * 2 // 3
    ]
    positions = np.array(positions)[furthest_indices]
    wxyzs = np.array(wxyzs)[furthest_indices]

    return np.array(positions), np.array(wxyzs)


# add the following to the utils and modify the server to use this:
def sample_cone_viewpoints_evenly(
    target_point: np.ndarray,
    current_camera_position: np.ndarray,
    num_samples: int = 100,
    max_angle_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample N camera viewpoints evenly distributed on a cone (spherical cap) using a Fibonacci lattice.

    Args:
        target_point: Center of the sphere (target object centroid).
        current_camera_position: Position of the current camera (pole of the cone).
        num_samples: Number of viewpoints to sample.
        max_angle_deg: Maximum angle from the pole (in degrees). 90.0 corresponds to a hemisphere.

    Returns:
        positions: Array of shape (N, 3) containing new camera positions.
        wxyzs: Array of shape (N, 4) containing new camera orientations (quaternions).
    """
    # 1. Determine radius and pole direction (local z-axis)
    vec_cam_target = current_camera_position - target_point
    radius = np.linalg.norm(vec_cam_target)
    if radius < 1e-6:
        radius = 0.5
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = vec_cam_target / radius

    # 2. Construct local frame
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, z_axis)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    positions = []
    wxyzs = []

    # 3. Fibonacci lattice sampling on a spherical cap
    # The range of z_local is [cos(max_angle), 1] instead of [0, 1].
    max_angle_rad = np.radians(max_angle_deg)
    min_z = np.cos(max_angle_rad)

    # Golden ratio
    phi = (np.sqrt(5) + 1) / 2

    for i in range(num_samples):
        # We want to sample z in [min_z, 1].
        # t goes from 0 to 1 as i goes from 0 to num_samples - 1.
        if num_samples > 1:
            t = i / (num_samples - 1)
        else:
            t = 0.0  # Only pole if 1 sample

        # Linear interpolation for z_local to ensure uniform area sampling on the cap
        # z = 1 - t * (1 - min_z)
        z_local = 1 - t * (1 - min_z)

        theta = 2 * np.pi * i / phi

        sin_theta = np.sqrt(1 - z_local**2)
        x_local = sin_theta * np.cos(theta)
        y_local = sin_theta * np.sin(theta)

        # 4. Transform to global frame
        offset = x_local * x_axis + y_local * y_axis + z_local * z_axis
        position = target_point + radius * offset

        positions.append(position)
        wxyzs.append(calculate_look_at_rotation(position, target_point))

    return np.array(positions), np.array(wxyzs)


def sample_random_camera_viewpoint(
    target_point: np.ndarray, xy_extent_meters: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    # Return position, quaternion of a random camera viewpoint looking at target_point
    # Sample position around origin (original camera)
    position = np.random.uniform(-xy_extent_meters, xy_extent_meters, 3)

    # Calculate rotation to look at target_point
    # z_c points from camera to target
    z_c = target_point - position
    z_c /= np.linalg.norm(z_c)

    # World down direction (assuming y-down in original frame if it's camera frame,
    # but standard is usually y-down for camera. Let's use a heuristic).
    # We just need a stable up vector.
    # If we assume standard camera frame (x-right, y-down, z-forward), then "down" is +y.
    down_world = np.array([0.0, 1.0, 0.0])

    # Orthogonalize down_world w.r.t z_c to get y_c
    y_c = down_world - np.dot(down_world, z_c) * z_c
    y_c_norm = np.linalg.norm(y_c)
    if y_c_norm < 1e-6:
        # Handle degenerate case (looking straight down/up)
        y_c = np.array([1.0, 0.0, 0.0])
        y_c = y_c - np.dot(y_c, z_c) * z_c
        y_c /= np.linalg.norm(y_c)
    else:
        y_c /= y_c_norm

    # x_c is cross product
    x_c = np.cross(y_c, z_c)

    # R_wc (Camera to World/Original)
    # Columns are camera axes expressed in World
    R_wc = np.column_stack([x_c, y_c, z_c])

    # Convert to wxyz quaternion
    wxyz = vtf.SO3.from_matrix(R_wc).wxyz
    return position, wxyz
