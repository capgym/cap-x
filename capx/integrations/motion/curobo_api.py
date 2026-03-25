"""CuRobo world and motion planning helpers."""

import pathlib
import time
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy
import trimesh

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState as StateJointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_segmenter import RobotSegmenter
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

# Franka panda_hand to gripper fingertip center (along hand z), from franka_description/panda.urdf
# panda_finger_joint1 origin xyz="0 0.04 0.0584" -> center between fingers is (0, 0, 0.0584) in hand frame
FRANKA_HAND_TO_FINGERTIP_Z_M = 0.0584 * 2



def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi] range."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _pick_nearest_solution(q_solutions: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Pick the IK solution closest to the reference joint configuration.
    
    Args:
        q_solutions: (K, 7) array of IK solutions
        q_ref: (7,) reference joint configuration
        
    Returns:
        (7,) array of the nearest solution
    """
    # q_solutions: (K, 7), q_ref: (7,)
    dq = _wrap_to_pi(q_solutions - q_ref[None, :])
    score = np.sum(dq * dq, axis=-1)
    return q_solutions[int(np.argmin(score))]


def _grasp_pose_fingertip_to_hand(
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    hand_to_fingertip_z: float = FRANKA_HAND_TO_FINGERTIP_Z_M,
) -> np.ndarray:
    """Convert grasp position from gripper fingertip frame to panda_hand (EE) frame.
    Grasp nets typically output pose for the fingertip center; CuRobo plans for panda_hand.
    Returns new position (3,) in world frame; orientation is unchanged."""
    R = R_scipy.from_quat(np.roll(np.asarray(quat_wxyz), -1)).as_matrix()  # wxyz -> xyzw for scipy
    offset_world = R @ np.array([0.0, 0.0, hand_to_fingertip_z])
    return np.asarray(position, dtype=np.float64) - offset_world


def save_world_and_robot_spheres_debug(
    world_config,
    joint_position: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    out_dir: str | pathlib.Path = "curobo_debug",
    tag: str = "debug",
    attached_obstacle: Any = None,
    attached_obstacle_pose: tuple[np.ndarray, np.ndarray] | None = None,
    exclude_obstacle_names: list[str] | None = None,
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """Save CuRobo world + robot collision spheres at a given joint configuration.

    This is a lightweight debug helper to visualize why IK / planning failed. It saves:

      - world_and_robot_spheres_{tag}.obj: single OBJ containing both the CuRobo world
        geometry (all obstacles in the WorldConfig) and the robot collision spheres at
        the given joint configuration. Optionally includes an attached object at a given pose.
      - robot_spheres_{tag}.npz: centers (N, 3) and radii (N,) of robot collision spheres.

    Use a separate script (e.g. Plotly, Open3D, or trimesh) to visualize the mesh+spheres.

    Args:
        world_config: CuRobo WorldConfig or None.
        joint_position: Joint configuration (7,) or (8,); only first 7 are used.
        robot_file: Robot config YAML under CuRobo's robot configs (default franka.yml).
        out_dir: Directory to save debug artifacts into (default ./curobo_debug).
        tag: String tag to distinguish multiple saves (e.g. "grasp_3_IK_FAIL").
        attached_obstacle: Optional CuRobo Obstacle (e.g. grasped object) to add at attached_obstacle_pose.
        attached_obstacle_pose: (position (3,), quat_wxyz (4,)) world frame; required if attached_obstacle is set.
        exclude_obstacle_names: When collecting world geometry, skip obstacles with these names (use to avoid
            drawing the attached object at its old world pose when adding it at attached_obstacle_pose).

    Returns:
        (world_path, spheres_path): Paths to saved files; either may be None on error.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    world_path: pathlib.Path | None = None
    spheres_path: pathlib.Path | None = None
    exclude_set = set(exclude_obstacle_names or [])

    # Collect world geometry for combined OBJ (skip tiny/noise meshes that appear as floating points)
    MIN_WORLD_MESH_VERTICES = 20
    MIN_WORLD_MESH_EXTENT_M = 0.025
    geometries: list[trimesh.Trimesh] = []
    if world_config is not None:
        try:
            # WorldConfig/obstacles expose get_trimesh_mesh()
            for obj in getattr(world_config, "objects", []) or []:
                if getattr(obj, "name", None) in exclude_set:
                    continue
                try:
                    m = obj.get_trimesh_mesh()
                    if m is None:
                        continue
                    n_verts = len(m.vertices) if hasattr(m.vertices, "__len__") else 0
                    if n_verts < MIN_WORLD_MESH_VERTICES:
                        continue
                    extents = getattr(m, "extents", None)
                    if extents is None and hasattr(m, "bounds") and m.bounds is not None:
                        extents = np.ptp(m.bounds, axis=0)
                    if extents is not None and float(np.max(extents)) < MIN_WORLD_MESH_EXTENT_M:
                        continue
                    geometries.append(m)
                except Exception as e:  # pragma: no cover - debug/log only
                    print(f"[curobo_debug] Warning: failed to convert world obstacle to mesh for tag={tag}: {e}")
        except Exception as e:  # pragma: no cover - debug/log only
            print(f"[curobo_debug] Warning: failed to iterate world objects for tag={tag}: {e}")

    # Add attached object: draw at its position in the world (no EE transform).
    # When the world is built with object_pose_override=EE, the object is already at the grasped pose.
    if attached_obstacle is not None:
        try:
            tensor_args = TensorDeviceType()
            mesh_added = False
            try:
                m = attached_obstacle.get_trimesh_mesh()
                if m is not None:
                    m_copy = m.copy()
                    verts = np.asarray(m_copy.vertices, dtype=np.float64)
                    pose_list = getattr(attached_obstacle, "pose", None)
                    if pose_list is not None and len(pose_list) >= 7:
                        obj_pose = Pose.from_list(pose_list, tensor_args)
                        verts_t = tensor_args.to_device(torch.from_numpy(verts).float())
                        verts = obj_pose.transform_points(verts_t).cpu().numpy()
                        m_copy.vertices = verts
                    geometries.append(m_copy)
                    mesh_added = True
            except Exception:
                pass
            if not mesh_added:
                sph_list = attached_obstacle.get_bounding_spheres(
                    n_spheres=min(128, 64),
                    surface_sphere_radius=0.002,
                    pre_transform_pose=None,
                    tensor_args=tensor_args,
                )
                for s in sph_list:
                    center = np.asarray(s.pose[:3], dtype=np.float64)
                    r = float(getattr(s, "radius", 0.01))
                    if r <= 0.0:
                        r = 0.01
                    sph_mesh = trimesh.creation.icosphere(radius=r)
                    sph_mesh.apply_translation(center)
                    geometries.append(sph_mesh)
        except Exception as e:  # pragma: no cover - debug/log only
            print(f"[curobo_debug] Warning: failed to add attached obstacle for tag={tag}: {e}")

    # Save robot collision spheres at the given joint configuration
    try:
        tensor_args = TensorDeviceType()
        robot_path = join_path(get_robot_configs_path(), robot_file)
        robot_cfg = RobotConfig.from_dict(load_yaml(robot_path), tensor_args)
        robot_model = CudaRobotModel(robot_cfg.kinematics)

        q = np.asarray(joint_position, dtype=np.float64).flatten()[:7]
        q_tensor = tensor_args.to_device(torch.from_numpy(q).unsqueeze(0).float())

        spheres_batch = robot_model.get_robot_as_spheres(q_tensor)
        if not spheres_batch:
            print(f"[curobo_debug] No robot spheres found for tag={tag}.")
            centers = np.zeros((0, 3), dtype=np.float64)
            radii = np.zeros((0,), dtype=np.float64)
        else:
            # get_robot_as_spheres returns list over batch; we use the first batch element
            spheres = spheres_batch[0]
            centers = np.array(
                [np.asarray(s.position, dtype=np.float64) for s in spheres],
                dtype=np.float64,
            )
            radii = np.array([float(s.radius) for s in spheres], dtype=np.float64)

        spheres_path = out_path / f"robot_spheres_{tag}.npz"
        np.savez(spheres_path, centers=centers, radii=radii)
        print(f"[curobo_debug] Saved robot spheres to {spheres_path}")

        # Add robot spheres to combined geometry
        for center, radius in zip(centers, radii):
            if radius <= 0.0:
                continue
            try:
                sph_mesh = trimesh.creation.icosphere(radius=float(radius))
                sph_mesh.apply_translation(center.astype(np.float64))
                geometries.append(sph_mesh)
            except Exception as e:  # pragma: no cover - debug/log only
                print(f"[curobo_debug] Warning: failed to create sphere mesh for tag={tag}: {e}")

        # Export combined world + robot spheres OBJ if we have any geometry
        if geometries:
            try:
                scene = trimesh.Scene(geometries)
                world_path = out_path / f"world_and_robot_spheres_{tag}.obj"
                scene.export(str(world_path))
                print(f"[curobo_debug] Saved combined world+spheres OBJ to {world_path}")
            except Exception as e:  # pragma: no cover - debug/log only
                print(f"[curobo_debug] Failed to save combined OBJ for tag={tag}: {e}")
                world_path = None
    except Exception as e:  # pragma: no cover - debug/log only
        print(f"[curobo_debug] Failed to save robot spheres for tag={tag}: {e}")
        spheres_path = None

    return world_path, spheres_path


def joint_trajectory_to_ee_positions(
    joint_traj: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    ee_link_name: str = "panda_hand",
) -> np.ndarray:
    """Compute end-effector positions (T, 3) from joint trajectory (T, 7) using CuRobo FK.
    joint_traj: (T, 7) arm joint positions in radians.
    Returns (T, 3) Cartesian positions in meters (robot base frame)."""
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    robot_path = join_path(get_robot_configs_path(), robot_file)
    robot_cfg = RobotConfig.from_dict(load_yaml(robot_path), tensor_args)
    model = CudaRobotModel(robot_cfg.kinematics)
    q = tensor_args.to_device(torch.from_numpy(np.asarray(joint_traj, dtype=np.float32)))
    if q.ndim == 1:
        q = q.unsqueeze(0)
    state = model.forward(q, link_name=ee_link_name)
    ee_pos = state[0].detach().cpu().numpy()
    return np.squeeze(ee_pos).astype(np.float64)


def robot_joint_position_to_ee_pose(
    joint_position: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    ee_link_name: str = "panda_hand",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute EE pose (position, quat_wxyz) for a single joint config using CuRobo FK.
    joint_position: (7,) or (8,) arm joint positions in radians.
    Returns (position (3,), quaternion_wxyz (4,)) in robot base frame."""
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    robot_path = join_path(get_robot_configs_path(), robot_file)
    robot_cfg = RobotConfig.from_dict(load_yaml(robot_path), tensor_args)
    model = CudaRobotModel(robot_cfg.kinematics)
    q = np.asarray(joint_position, dtype=np.float32).flatten()[:7]
    q_t = tensor_args.to_device(torch.from_numpy(q).unsqueeze(0))
    state = model.forward(q_t, link_name=ee_link_name)
    ee_pos = state[0].detach().cpu().numpy().squeeze()
    ee_quat = state[1].detach().cpu().numpy().squeeze()  # wxyz
    return ee_pos.astype(np.float64), ee_quat.astype(np.float64)


def create_curobo_world_from_depth(
    depth_image: np.ndarray,
    object_mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray | None = None,
    *,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
    marching_cubes_pitch: float = 0.04,
    scene_name: str = "scene",
    object_name: str = "object",
):
    """
    Create a CuRobo WorldConfig from a depth image and object mask.

    Builds a point cloud from the depth image, splits it into (A) points under
    the object mask and (B) all other points (scene/obstacles), then converts
    each part to a mesh via marching cubes and returns a single WorldConfig
    containing both as collision obstacles.

    :param depth_image: Depth image (H, W) in meters.
    :param object_mask: Boolean or 0/1 mask (H, W). True/1 = object; False/0 = scene.
    :param intrinsics: Camera intrinsics (3, 3).
    :param camera_pose: Optional (4, 4) camera-to-world transform. If None, points stay in camera frame.
    :param depth_clip_range: (near, far) depth range in meters.
    :param marching_cubes_pitch: Voxel size for marching cubes when converting point clouds to mesh.
    :param scene_name: Name for the scene obstacle in the world.
    :param object_name: Name for the object obstacle in the world.
    :return: WorldConfig with mesh obstacles for scene and object (each only added if non-empty).
    """
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    mask = np.asarray(object_mask, dtype=bool)
    if mask.ndim != 2 or depth.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "depth_image and object_mask must be 2D with same shape, got "
            f"depth {depth.shape}, mask {mask.shape}"
        )
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {intrinsics.shape}")

    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    w, h = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    w_flat = w.reshape(-1)
    h_flat = h.reshape(-1)
    z_flat = depth.reshape(-1)

    near, far = depth_clip_range
    valid = (
        ~np.isnan(z_flat)
        & ~np.isinf(z_flat)
        & (z_flat >= near)
        & (z_flat <= far)
        & (z_flat > 0)
    )

    x_cam = (w_flat - cx) * z_flat / fx
    y_cam = (h_flat - cy) * z_flat / fy
    points_cam = np.stack([x_cam, y_cam, z_flat], axis=-1)
    points_cam = points_cam[valid]
    mask_flat = mask.reshape(-1)[valid]

    if camera_pose is not None:
        cam = np.asarray(camera_pose, dtype=np.float64)
        if cam.shape != (4, 4):
            raise ValueError(f"camera_pose must be (4, 4), got {cam.shape}")
        ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
        points_cam_hom = np.hstack([points_cam, ones])
        points = (cam @ points_cam_hom.T).T[:, :3]
    else:
        points = points_cam

    scene_points = points[~mask_flat]
    object_points = points[mask_flat]

    # Subsample to avoid marching cubes on 100k+ points (very slow / hang)
    max_points_for_mc = 25000
    rng = np.random.default_rng(42)
    if scene_points.shape[0] > max_points_for_mc:
        idx = rng.choice(scene_points.shape[0], max_points_for_mc, replace=False)
        scene_points = scene_points[idx]
    if object_points.shape[0] > max_points_for_mc:
        idx = rng.choice(object_points.shape[0], max_points_for_mc, replace=False)
        object_points = object_points[idx]

    meshes: list[Mesh] = []
    if scene_points.shape[0] > 0:
        scene_mesh = Mesh.from_pointcloud(
            scene_points,
            pitch=marching_cubes_pitch,
            name=scene_name,
        )
        meshes.append(scene_mesh)
    if object_points.shape[0] > 0:
        object_mesh = Mesh.from_pointcloud(
            object_points,
            pitch=marching_cubes_pitch,
            name=object_name,
        )
        meshes.append(object_mesh)

    return WorldConfig(mesh=meshes)


def _get_robot_mask_from_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_4x4: np.ndarray,
    joint_position: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    distance_threshold: float = 0.05,
    collision_sphere_buffer: float = 0.01,
) -> np.ndarray:
    """Use CuRobo RobotSegmenter to get a boolean mask (H, W) where True = robot pixel.
    Expects depth in meters; camera_pose is camera-to-world (world = robot base).
    """
    tensor_args = TensorDeviceType()
    segmenter = RobotSegmenter.from_robot_file(
        robot_file,
        collision_sphere_buffer=collision_sphere_buffer,
        distance_threshold=distance_threshold,
        use_cuda_graph=False,
        tensor_args=tensor_args,
        ops_dtype=torch.float16,
        depth_to_meter=0.001,
    )
    # Depth in mm for CameraObservation (segmenter uses depth_to_meter=0.001)
    depth_mm = torch.from_numpy(depth_m.astype(np.float32)).to(
        device=tensor_args.device, dtype=torch.float32
    ) * 1000.0
    if depth_mm.ndim == 2:
        depth_mm = depth_mm.unsqueeze(0)
    intrinsics_t = tensor_args.to_device(torch.from_numpy(intrinsics.astype(np.float32)))
    pose_t = tensor_args.to_device(torch.from_numpy(camera_pose_4x4.astype(np.float32)))
    cam_obs = CameraObservation(
        depth_image=depth_mm,
        intrinsics=intrinsics_t,
        pose=Pose.from_matrix(pose_t),
    )
    q = np.asarray(joint_position, dtype=np.float64).flatten()[:7]
    q_tensor = tensor_args.to_device(torch.from_numpy(q).unsqueeze(0).float())
    q_js = StateJointState(position=q_tensor, joint_names=segmenter.kinematics.joint_names)
    if not segmenter.ready:
        segmenter.update_camera_projection(cam_obs)
    depth_mask, _ = segmenter.get_robot_mask_from_active_js(cam_obs, q_js)
    return depth_mask[0].detach().cpu().numpy()


def create_curobo_world_from_depth_full(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray | None = None,
    *,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
    marching_cubes_pitch: float = 0.03,
    scene_name: str = "scene",
    robot_joint_position: np.ndarray | None = None,
    robot_file: str = "franka.yml",
    robot_distance_threshold: float = 0.05,
    max_points_for_marching_cubes: int = 25000,
):
    """
    Create a single CuRobo WorldConfig mesh from the full depth image (no object/scene split).

    Use this for approach-to-grasp planning to avoid false collisions from object and scene
    meshes overlapping (e.g. object resting on table). Optionally exclude robot pixels using
    CuRobo's RobotSegmenter (kinematics + sphere distance), so the start configuration is
    not in collision with the scene mesh.

    :param depth_image: Depth image (H, W) in meters.
    :param intrinsics: Camera intrinsics (3, 3).
    :param camera_pose: Optional (4, 4) camera-to-world (robot base) transform.
    :param depth_clip_range: (near, far) depth range in meters.
    :param marching_cubes_pitch: Voxel size for marching cubes. Default 0.02 (finer mesh).
    :param scene_name: Name for the single scene mesh.
    :param robot_joint_position: If provided, robot pixels are removed via RobotSegmenter.
    :param robot_file: Robot config file for RobotSegmenter (default franka.yml).
    :param robot_distance_threshold: Distance (m) threshold for segmenter. Default 0.05.
    :param max_points_for_marching_cubes: Subsample point cloud to this many points before MC.
    :return: WorldConfig with a single mesh obstacle.
    """
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {intrinsics.shape}")

    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    w, h = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    w_flat = w.reshape(-1)
    h_flat = h.reshape(-1)
    z_flat = depth.reshape(-1)

    near, far = depth_clip_range
    valid = (
        ~np.isnan(z_flat)
        & ~np.isinf(z_flat)
        & (z_flat >= near)
        & (z_flat <= far)
        & (z_flat > 0)
    )

    x_cam = (w_flat - cx) * z_flat / fx
    y_cam = (h_flat - cy) * z_flat / fy
    points_cam = np.stack([x_cam, y_cam, z_flat], axis=-1)
    valid_indices = np.where(valid)[0]
    points = points_cam[valid]

    if robot_joint_position is not None and camera_pose is not None:
        robot_mask = _get_robot_mask_from_depth(
            depth,
            intrinsics,
            camera_pose,
            robot_joint_position,
            robot_file=robot_file,
            distance_threshold=robot_distance_threshold,
        )
        robot_mask_flat = robot_mask.reshape(-1)
        robot_at_valid = robot_mask_flat[valid_indices]
        keep = ~robot_at_valid
        points = points[keep]

    if camera_pose is not None:
        cam = np.asarray(camera_pose, dtype=np.float64)
        if cam.shape != (4, 4):
            raise ValueError(f"camera_pose must be (4, 4), got {cam.shape}")
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        points_hom = np.hstack([points, ones])
        points = (cam @ points_hom.T).T[:, :3]

    rng = np.random.default_rng(42)
    if points.shape[0] > max_points_for_marching_cubes:
        idx = rng.choice(points.shape[0], max_points_for_marching_cubes, replace=False)
        points = points[idx]

    meshes = []
    if points.shape[0] > 0:
        meshes.append(
            Mesh.from_pointcloud(
                points,
                pitch=marching_cubes_pitch,
                name=scene_name,
            )
        )
    return WorldConfig(mesh=meshes)


def create_curobo_world_from_pointcloud(
    point_cloud: np.ndarray,
    object_mask: np.ndarray,
    *,
    marching_cubes_pitch: float = 0.04,
    scene_name: str = "scene",
    object_name: str = "object",
    max_points_for_marching_cubes: int = 25000,
):
    """
    Create a CuRobo WorldConfig from a point cloud and a per-point object mask.

    Splits the point cloud into (A) points where object_mask is True and (B) the rest
    (scene), then converts each part to a mesh and returns a WorldConfig.

    :param point_cloud: Points (N, 3) in the desired world or camera frame.
    :param object_mask: Boolean or 0/1 array of shape (N,). True/1 = object; False/0 = scene.
    :param marching_cubes_pitch: Voxel size for marching cubes.
    :param scene_name: Name for the scene obstacle.
    :param object_name: Name for the object obstacle.
    :return: WorldConfig with mesh obstacles for scene and object (each only if non-empty).
    """
    points = np.asarray(point_cloud, dtype=np.float64)
    mask = np.asarray(object_mask, dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"point_cloud must be (N, 3), got {points.shape}")
    if mask.shape[0] != points.shape[0]:
        raise ValueError(
            f"object_mask length must match point cloud size, got {mask.shape[0]} vs {points.shape[0]}"
        )

    scene_points = points[~mask]
    object_points = points[mask]

    rng = np.random.default_rng(42)
    if scene_points.shape[0] > max_points_for_marching_cubes:
        idx = rng.choice(scene_points.shape[0], max_points_for_marching_cubes, replace=False)
        scene_points = scene_points[idx]
    if object_points.shape[0] > max_points_for_marching_cubes:
        idx = rng.choice(object_points.shape[0], max_points_for_marching_cubes, replace=False)
        object_points = object_points[idx]

    meshes: list[Mesh] = []
    if scene_points.shape[0] > 0:
        meshes.append(
            Mesh.from_pointcloud(
                scene_points,
                pitch=marching_cubes_pitch,
                name=scene_name,
            )
        )
    if object_points.shape[0] > 0:
        meshes.append(
            Mesh.from_pointcloud(
                object_points,
                pitch=marching_cubes_pitch,
                name=object_name,
            )
        )

    return WorldConfig(mesh=meshes)


def plan_to_grasp_poses(
    world_config,
    start_joint_position: np.ndarray,
    grasp_poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    max_attempts: int = 8,
    use_cuda_graph: bool = False,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.05,
    position_threshold_z: float | None = 0.01,
    grasp_pose_is_fingertip: bool = True,
    num_ik_seeds: int = 128,
    relax_orientation: bool = False,
    use_grasp_approach: bool = False,
    grasp_approach_offset: float = 0.03,
    grasp_approach_linear_axis: int = 2,
    grasp_approach_tstep_fraction: float = 0.7,
    use_world_collision: bool = True,
    robot_collision_sphere_buffer: float | None = None,
    collision_activation_distance: float | None = 0.01,
    ignore_obstacle_names: list[str] | None = None,
) -> tuple[bool, np.ndarray | None, int | None]:
    """
    Plan a collision-free trajectory from the current joint configuration to one of
    the given grasp poses (end-effector position + quaternion wxyz) in the given
    CuRobo world.

    Frame convention: grasp_poses must be in the robot base frame (CuRobo uses the
    robot base as world origin). Camera extrinsics and any saved grasp poses must
    be expressed in this same frame.

    :param world_config: CuRobo WorldConfig (e.g. from create_curobo_world_from_depth). Ignored if use_world_collision=False.
    :param start_joint_position: Current joint positions (7,) or (8,); only first 7 (arm) are used.
    :param grasp_poses: List of (position (3,), quaternion_wxyz (4,)) in robot base (world) frame.
    :param robot_file: Robot config filename under curobo robot configs (default franka.yml).
    :param tensor_args: Device/dtype; default TensorDeviceType().
    :param max_attempts: Max planning attempts per grasp.
    :param use_cuda_graph: Whether to use CUDA graph (disable for varying world/start).
    :param position_threshold: Success threshold for position (m); larger allows more variance (default 0.02).
    :param rotation_threshold: Success threshold for orientation; larger allows more variance (default 0.12).
    :param position_threshold_z: If set, use max(position_threshold, position_threshold_z) so final reached
        position can have more variance (e.g. on z). Default 0.05 when collision checking. Set None to use only position_threshold.
    :param grasp_pose_is_fingertip: If True (default), treat grasp position as gripper fingertip center and
        convert to panda_hand (EE) frame using FRANKA_HAND_TO_FINGERTIP_Z_M before planning.
    :param num_ik_seeds: Number of IK seeds (default 128). Higher can fix IK_FAIL at cost of time.
    :param relax_orientation: If True (default), only require reaching goal position; orientation is relaxed
        to improve IK success. Set False to require full pose. Ignored if use_grasp_approach=True.
    :param use_grasp_approach: If True, use PoseCostMetric.create_grasp_approach_metric to bias the
        trajectory towards a two-phase motion: move towards an offset (pre-grasp) and then linearly
        approach the final grasp along a single axis (no hard stop at the offset; blended path).
    :param grasp_approach_offset: Offset (m) along the linear axis for grasp-approach cost.
    :param grasp_approach_linear_axis: Linear axis index for grasp-approach cost (0=x, 1=y, 2=z).
    :param grasp_approach_tstep_fraction: Timestep fraction in [0,1] at which to start activating the
        grasp-approach constraint (later part of the trajectory).
    :param use_world_collision: If True (default), use world_config for collision checking. If False, pass
        world_model=None to MotionGen (no obstacle collision checking).
    :param robot_collision_sphere_buffer: If set, override the robot's collision_sphere_buffer (m). Negative
        values shrink the robot's collision spheres and can reduce IK_FAIL when the world mesh is close.
        E.g. -0.01 or -0.02. Default None uses the value from the robot config file.
    :param collision_activation_distance: Distance (m) to activate collision cost; smaller is less
        conservative and can reduce IK_FAIL with dense meshes. Default 0.01. Set None to use CuRobo default.
    :param ignore_obstacle_names: If provided, these world obstacles are disabled for collision during
        planning (e.g. the object to grasp so the robot can approach it). Re-enabled before return.
    :return: (success, trajectory, goalset_index). trajectory is (T, 7) joint positions or None.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    device = tensor_args.device
    dtype = tensor_args.dtype

    robot_path = join_path(get_robot_configs_path(), robot_file)
    robot_dict = load_yaml(robot_path)
    if robot_collision_sphere_buffer is not None:
        inner = robot_dict.get("robot_cfg", robot_dict)
        if "kinematics" in inner:
            inner["kinematics"] = {**inner["kinematics"], "collision_sphere_buffer": robot_collision_sphere_buffer}
            print(f"[plan_to_grasp_poses] Overriding robot collision_sphere_buffer={robot_collision_sphere_buffer} m.")
    robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)
    world_model = world_config if use_world_collision else None
    if not use_world_collision:
        print("[plan_to_grasp_poses] use_world_collision=False: not providing world model to MotionGen (no obstacle collision checking).")
    effective_position_threshold = position_threshold
    if position_threshold_z is not None:
        effective_position_threshold = max(position_threshold, position_threshold_z)
        if effective_position_threshold > position_threshold:
            print(f"[plan_to_grasp_poses] Using position_threshold={effective_position_threshold:.3f} m (z allowance: position_threshold_z={position_threshold_z}).")
    # Debug: print thresholds and collision buffer settings
    print(
        "[plan_to_grasp_poses] thresholds: "
        f"position_threshold={position_threshold}, "
        f"rotation_threshold={rotation_threshold}, "
        f"position_threshold_z={position_threshold_z}, "
        f"effective_position_threshold={effective_position_threshold}"
    )
    print(
        f"[plan_to_grasp_poses] robot_collision_sphere_buffer="
        f"{robot_collision_sphere_buffer if robot_collision_sphere_buffer is not None else 'config_default'}"
    )
    if collision_activation_distance is not None:
        print(f"[plan_to_grasp_poses] collision_activation_distance={collision_activation_distance} m.")
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=world_model,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
        position_threshold=effective_position_threshold,
        rotation_threshold=rotation_threshold,
        num_ik_seeds=num_ik_seeds,
        collision_activation_distance=collision_activation_distance,
        # maximum_trajectory_dt=
    )
    motion_gen = MotionGen(motion_gen_cfg)

    q_start = np.asarray(start_joint_position, dtype=np.float64)
    q_start = q_start[:7].flatten()
    print(f"[plan_to_grasp_poses] start_joint_position (7): {q_start.tolist()}")
    print(f"[plan_to_grasp_poses] start valid: no_nan={not np.any(np.isnan(q_start))}, no_inf={not np.any(np.isinf(q_start))}, in_range=[{q_start.min():.3f}, {q_start.max():.3f}]")
    start_state = JointState.from_position(
        tensor_args.to_device(torch.from_numpy(q_start).unsqueeze(0).float())
    )

    n = len(grasp_poses)
    if n == 0:
        print("[plan_to_grasp_poses] No grasp poses; returning False, None, None.")
        return False, None, None

    if use_grasp_approach:
        pose_metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=grasp_approach_offset,
            linear_axis=grasp_approach_linear_axis,
            tstep_fraction=grasp_approach_tstep_fraction,
            tensor_args=tensor_args,
        )
        pose_metric.reach_partial_pose=True
        pose_metric.reach_vec_weight=tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 0.2])
        pose_metric.hold_vec_weight=tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 0.2])

        print(
            "[plan_to_grasp_poses] Using grasp-approach metric: "
            f"offset={grasp_approach_offset}, axis={grasp_approach_linear_axis}, "
            f"tstep_fraction={grasp_approach_tstep_fraction}."
        )
    elif relax_orientation:
        pose_metric = PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 0.2]),
        )
        print("[plan_to_grasp_poses] Using position-only reach (relax_orientation=True) to improve IK success.")
    else:
        pose_metric = PoseCostMetric.reset_metric()
    plan_cfg = MotionGenPlanConfig(
        pose_cost_metric=pose_metric,
        max_attempts=max_attempts,
        enable_graph_attempt=False,
        # time_dilation_factor=,        # try 1.25–2.0
        # finetune_js_dt_scale=,        # try 1.25–2.0
    )

    # Disable collision with specified obstacles (e.g. object to grasp) so robot can approach
    if ignore_obstacle_names:
        for name in ignore_obstacle_names:
            try:
                motion_gen.world_coll_checker.enable_obstacle(enable=False, name=name)
            except Exception as e:
                print(f"[plan_to_grasp_poses] Warning: could not disable obstacle '{name}': {e}")
    try:
        for idx in range(n):
            pos = np.asarray(grasp_poses[idx][0], dtype=np.float64)
            # pos[2] += 0.05
            quat = np.asarray(grasp_poses[idx][1], dtype=np.float64)
            if grasp_pose_is_fingertip:
                pos = _grasp_pose_fingertip_to_hand(pos, quat)
                print(f"[plan_to_grasp_poses] grasp {idx}/{n}: position (panda_hand after offset): {pos.tolist()}, quat_wxyz={quat.tolist()}")
            else:
                print(f"[plan_to_grasp_poses] grasp {idx}/{n}: position={pos.tolist()}, quat_wxyz={quat.tolist()}")

            goal_pose = Pose(
                position=tensor_args.to_device(torch.from_numpy(pos).unsqueeze(0).float()),
                quaternion=tensor_args.to_device(torch.from_numpy(quat).unsqueeze(0).float()),
            )

            # First solve IK to find the closest joint configuration, then plan in joint space
            # When ignoring the object (e.g. to grasp it), IK should not check collision with it
            ik_world = None if ignore_obstacle_names else world_model
            ik_cfg = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                world_model=ik_world,
                tensor_args=tensor_args,
                num_seeds=num_ik_seeds,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                self_collision_check=True,
            )
            ik_solver = IKSolver(ik_cfg)

            retract_cfg = tensor_args.to_device(torch.from_numpy(q_start).unsqueeze(0).float())
            ik_result = ik_solver.solve_single(goal_pose, retract_config=retract_cfg)

            if bool(ik_result.success.item()):
                # Extract IK solutions and pick the nearest to start state
                q_sols = ik_result.solution.detach().cpu().numpy()
                q_sols = np.atleast_2d(q_sols)[:, :7]  # (K, 7)
                q_goal = _pick_nearest_solution(q_sols, q_start)

                # CuRobo versions differ in where joint_names live:
                # - Some expose robot_cfg.kinematics.joint_names (kinematics model)
                # - Others expose robot_cfg.cspace.joint_names (config)
                joint_names = None
                try:
                    joint_names = getattr(robot_cfg.kinematics, "joint_names")
                except Exception:
                    joint_names = None
                if joint_names is None:
                    try:
                        joint_names = getattr(robot_cfg, "cspace").joint_names
                    except Exception:
                        joint_names = None
                if joint_names is None:
                    try:
                        joint_names = getattr(robot_cfg.kinematics, "cspace").joint_names
                    except Exception:
                        joint_names = None
                if joint_names is None:
                    raise AttributeError(
                        "Could not determine robot joint_names from RobotConfig. "
                        "Tried robot_cfg.kinematics.joint_names and robot_cfg.cspace.joint_names."
                    )

                goal_state = JointState(
                    position=tensor_args.to_device(torch.from_numpy(q_goal).unsqueeze(0).float()),
                    joint_names=joint_names,
                )

                # Plan in joint space to the selected IK solution
                result = motion_gen.plan_single_js(start_state, goal_state, plan_cfg)
                print(f"[plan_to_grasp_poses] grasp {idx}: IK succeeded, planning in joint space to nearest solution.")
            else:
                # Fallback to pose planning if IK fails
                print(f"[plan_to_grasp_poses] grasp {idx}: IK failed, falling back to pose planning.")
                result = motion_gen.plan_single(start_state, goal_pose, plan_cfg)

            success = bool(result.success.item() if result.success is not None else False)
            status_str = getattr(result.status, "name", str(result.status)) if hasattr(result, "status") and result.status is not None else "N/A"
            traj_shape = None
            if result.interpolated_plan is not None:
                try:
                    traj_js = result.get_interpolated_plan()
                    if traj_js is not None and traj_js.position is not None:
                        traj_shape = traj_js.position.shape
                except Exception as e:  # pragma: no cover - debug/log only
                    print(f"[plan_to_grasp_poses] Warning: failed to get interpolated plan shape for grasp {idx}: {e}")

            print(f"[plan_to_grasp_poses] grasp {idx}: success={success}, status={status_str}, trajectory shape={traj_shape}")

            if not success:
                # Debug helper: solve pure IK (no world collision) to reach goal, then save
                # world + robot collision spheres at that IK solution to inspect intersections.
                q_ik_debug = None
                try:
                    ik_cfg = IKSolverConfig.load_from_robot_config(
                        robot_cfg,
                        world_model=None,  # no world collision checking
                        tensor_args=tensor_args,
                        num_seeds=num_ik_seeds,
                        position_threshold=position_threshold,
                        rotation_threshold=rotation_threshold,
                        self_collision_check=False,
                        self_collision_opt=False,
                    )
                    ik_solver = IKSolver(ik_cfg)

                    retract_cfg = tensor_args.to_device(
                        torch.from_numpy(q_start).unsqueeze(0).float()
                    )
                    ik_result = ik_solver.solve_single(
                        goal_pose,
                        retract_config=retract_cfg,
                    )
                    ik_success = bool(ik_result.success.item())
                    if not ik_success:
                        try:
                            pos_err = float(ik_result.position_error.item())
                            rot_err = float(ik_result.rotation_error.item())
                            print(
                                f"[plan_to_grasp_poses] IK(no-collision) also failed for grasp {idx}: "
                                f"position_error={pos_err:.4f}, rotation_error={rot_err:.4f}"
                            )
                        except Exception:
                            print(
                                f"[plan_to_grasp_poses] IK(no-collision) also failed for grasp {idx} (could not read errors)."
                            )
                    else:
                        sol = ik_result.solution.detach().cpu().numpy()
                        q_ik_debug = np.asarray(sol, dtype=np.float64).reshape(-1)[:7]
                        print(
                            f"[plan_to_grasp_poses] IK(no-collision) succeeded for grasp {idx}; saving debug world+spheres."
                        )
                except Exception as e:  # pragma: no cover - debug/log only
                    print(
                        f"[plan_to_grasp_poses] Warning: IK(no-collision) debug failed for grasp {idx}: {e}"
                    )
                    q_ik_debug = None

                if q_ik_debug is not None:
                    try:
                        tag = f"grasp_{idx}_IK_NO_COLLISION"
                        save_world_and_robot_spheres_debug(
                            world_model,
                            q_ik_debug,
                            robot_file=robot_file,
                            tag=tag,
                        )
                    except Exception as e:  # pragma: no cover - debug/log only
                        print(
                            f"[plan_to_grasp_poses] Warning: failed to save debug world/spheres for grasp {idx}: {e}"
                        )

            if success and result.interpolated_plan is not None:
                try:
                    traj_js = result.get_interpolated_plan()
                    if traj_js is not None and traj_js.position is not None:
                        trajectory = np.squeeze(traj_js.position.detach().cpu().numpy())
                        print(f"[plan_to_grasp_poses] Returning first success at grasp index {idx}.")
                        return True, trajectory, idx
                except Exception as e:  # pragma: no cover - debug/log only
                    print(f"[plan_to_grasp_poses] Warning: failed to extract trajectory for grasp {idx} despite success: {e}")

        print("[plan_to_grasp_poses] No grasp succeeded; returning False, None, None.")
        return False, None, None
    finally:
        if ignore_obstacle_names:
            for name in ignore_obstacle_names:
                try:
                    motion_gen.world_coll_checker.enable_obstacle(enable=True, name=name)
                except Exception as e:
                    print(f"[plan_to_grasp_poses] Warning: could not re-enable obstacle '{name}': {e}")


def create_curobo_world_from_depth_with_object(
    depth_image: np.ndarray,
    object_mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray | None = None,
    *,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
    marching_cubes_pitch: float = 0.03,
    scene_name: str = "scene",
    object_name: str = "object",
    robot_joint_position: np.ndarray | None = None,
    robot_file: str = "franka.yml",
    robot_distance_threshold: float = 0.15,
    max_points_for_marching_cubes: int = 25000,
    object_pose_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> Any:
    """
    Create a CuRobo WorldConfig with object and scene as separate meshes (robot excluded).

    Similar to create_curobo_world_from_depth, but also excludes points near the robot
    (like create_curobo_world_from_depth_full) so the world doesn't include the robot volume.
    This is useful for planning with a grasped object: the object mesh can be attached to
    the robot, while the scene mesh remains as a collision obstacle.

    :param depth_image: Depth image (H, W) in meters.
    :param object_mask: Boolean or 0/1 mask (H, W). True/1 = object; False/0 = scene.
    :param intrinsics: Camera intrinsics (3, 3).
    :param camera_pose: Optional (4, 4) camera-to-world (robot base) transform.
    :param depth_clip_range: (near, far) depth range in meters.
    :param marching_cubes_pitch: Voxel size for marching cubes. Default 0.03.
    :param scene_name: Name for the scene obstacle in the world.
    :param object_name: Name for the object obstacle in the world.
    :param robot_joint_position: If provided, robot pixels are removed via RobotSegmenter.
    :param robot_file: Robot config file for RobotSegmenter (default franka.yml).
    :param robot_distance_threshold: Distance (m) threshold for segmenter. Default 0.15.
    :param max_points_for_marching_cubes: Subsample point cloud to this many points before MC.
    :param object_pose_override: Optional (position (3,), quat_wxyz (4,)). When provided (e.g. current EE
        pose when object is grasped), the object point cloud is placed at this pose so the object mesh
        in the world matches the grasped location; avoids wrong attached-object position in planning/debug.
    :return: WorldConfig with mesh obstacles for scene and object (each only added if non-empty).
    """
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    mask = np.asarray(object_mask, dtype=bool)
    if mask.ndim != 2 or depth.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "depth_image and object_mask must be 2D with same shape, got "
            f"depth {depth.shape}, mask {mask.shape}"
        )
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {intrinsics.shape}")

    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    w, h = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    w_flat = w.reshape(-1)
    h_flat = h.reshape(-1)
    z_flat = depth.reshape(-1)

    near, far = depth_clip_range
    valid = (
        ~np.isnan(z_flat)
        & ~np.isinf(z_flat)
        & (z_flat >= near)
        & (z_flat <= far)
        & (z_flat > 0)
    )

    x_cam = (w_flat - cx) * z_flat / fx
    y_cam = (h_flat - cy) * z_flat / fy
    points_cam = np.stack([x_cam, y_cam, z_flat], axis=-1)
    valid_indices = np.where(valid)[0]
    points = points_cam[valid]
    mask_flat = mask.reshape(-1)[valid_indices]

    # Exclude robot points from scene only (keep object points so grasped object stays in world)
    if robot_joint_position is not None and camera_pose is not None:
        robot_mask = _get_robot_mask_from_depth(
            depth,
            intrinsics,
            camera_pose,
            robot_joint_position,
            robot_file=robot_file,
            distance_threshold=robot_distance_threshold,
        )
        robot_mask_flat = robot_mask.reshape(-1)
        robot_at_valid = robot_mask_flat[valid_indices]
        is_object = mask_flat
        # Keep all object points; drop only scene points that are near the robot
        keep = is_object | (~robot_at_valid)
        points = points[keep]
        mask_flat = mask_flat[keep]

    # Transform to world frame if camera_pose provided
    if camera_pose is not None:
        cam = np.asarray(camera_pose, dtype=np.float64)
        if cam.shape != (4, 4):
            raise ValueError(f"camera_pose must be (4, 4), got {cam.shape}")
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        points_hom = np.hstack([points, ones])
        points = (cam @ points_hom.T).T[:, :3]

    # Split into object and scene
    scene_points = points[~mask_flat]
    object_points = points[mask_flat]

    # Optionally place object at a given pose (e.g. EE when grasped) so world matches reality
    if object_pose_override is not None and object_points.shape[0] > 0:
        pos_ov = np.asarray(object_pose_override[0], dtype=np.float64).reshape(3)
        quat_ov = np.asarray(object_pose_override[1], dtype=np.float64).reshape(4)
        tensor_args = TensorDeviceType()
        override_pose = Pose(
            position=tensor_args.to_device(torch.from_numpy(pos_ov).unsqueeze(0).float()),
            quaternion=tensor_args.to_device(torch.from_numpy(quat_ov).unsqueeze(0).float()),
        )
        obj_center = object_points.mean(axis=0)
        object_points_centered = object_points - obj_center
        pts_t = tensor_args.to_device(torch.from_numpy(object_points_centered.astype(np.float32)).float())
        object_points = override_pose.transform_points(pts_t).detach().cpu().numpy().astype(np.float64)

    # Subsample to avoid marching cubes on 100k+ points
    rng = np.random.default_rng(42)
    if scene_points.shape[0] > max_points_for_marching_cubes:
        idx = rng.choice(scene_points.shape[0], max_points_for_marching_cubes, replace=False)
        scene_points = scene_points[idx]
    if object_points.shape[0] > max_points_for_marching_cubes:
        idx = rng.choice(object_points.shape[0], max_points_for_marching_cubes, replace=False)
        object_points = object_points[idx]

    meshes: list[Mesh] = []
    if scene_points.shape[0] > 0:
        meshes.append(
            Mesh.from_pointcloud(
                scene_points,
                pitch=marching_cubes_pitch,
                name=scene_name,
            )
        )
    if object_points.shape[0] > 0:
        meshes.append(
            Mesh.from_pointcloud(
                object_points,
                pitch=marching_cubes_pitch,
                name=object_name,
            )
        )

    return WorldConfig(mesh=meshes)


def plan_with_grasped_object(
    world_config: Any,
    start_joint_position: np.ndarray,
    target_pose: tuple[np.ndarray, np.ndarray],
    object_name: str,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    max_attempts: int = 8,
    use_cuda_graph: bool = False,
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.1,
    position_threshold_z: float | None = 0.05,
    num_ik_seeds: int = 128,
    use_world_collision: bool = True,
    robot_collision_sphere_buffer: float | None = None,
    collision_activation_distance: float | None = 0.01,
    surface_sphere_radius: float = 0.001,
    link_name: str = "attached_object",
    remove_obstacles_from_world: bool = False,
    debug_out_dir: str | pathlib.Path | None = "curobo_debug2",
) -> tuple[bool, np.ndarray | None]:
    """
    Plan a collision-free trajectory to move a grasped object to a target pose.

    This function:
    1. Attaches the object (by name) from world_config to the robot at the current joint state
    2. Computes IK for the goal pose, selecting the solution closest to the start joint state
    3. Plans a trajectory in joint space from start to the IK solution
    4. Returns the joint trajectory

    The object must exist in world_config with the given object_name. After attaching,
    the object moves with the robot and is checked for collisions with the remaining
    scene obstacles.

    If planning fails, debug world meshes are saved to debug_out_dir (default: "curobo_debug2"):
    - start_joint_state_{tag}.obj: World + robot at start configuration
    - end_joint_state_{tag}.obj: World + robot at IK goal configuration (if IK succeeded)

    :param world_config: CuRobo WorldConfig containing the object (by object_name) and scene obstacles.
    :param start_joint_position: Current joint positions (7,) or (8,); object is assumed grasped at this state.
    :param target_pose: (position (3,), quaternion_wxyz (4,)) target pose in robot base (world) frame.
    :param object_name: Name of the object in world_config to attach to the robot.
    :param robot_file: Robot config filename under curobo robot configs (default franka.yml).
    :param tensor_args: Device/dtype; default TensorDeviceType().
    :param max_attempts: Max planning attempts.
    :param use_cuda_graph: Whether to use CUDA graph (disable for varying world/start).
    :param position_threshold: Success threshold for position (m).
    :param rotation_threshold: Success threshold for orientation.
    :param position_threshold_z: If set, use max(position_threshold, position_threshold_z).
    :param num_ik_seeds: Number of IK seeds (default 128).
    :param use_world_collision: If True, use world_config for collision checking.
    :param robot_collision_sphere_buffer: Override robot collision_sphere_buffer (m).
    :param collision_activation_distance: Distance (m) to activate collision cost.
    :param surface_sphere_radius: Radius (m) for surface spheres when attaching object.
    :param link_name: Link name to attach object to (default "attached_object").
    :param remove_obstacles_from_world: Remove attached object from world after attaching.
    :param debug_out_dir: Directory to save debug meshes on planning failure (default "curobo_debug2"). Set to None to disable.
    :return: (success, joint_trajectory). joint_trajectory is (T, 7) or None.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    device = tensor_args.device
    dtype = tensor_args.dtype

    robot_path = join_path(get_robot_configs_path(), robot_file)
    robot_dict = load_yaml(robot_path)
    if robot_collision_sphere_buffer is not None:
        inner = robot_dict.get("robot_cfg", robot_dict)
        if "kinematics" in inner:
            inner["kinematics"] = {**inner["kinematics"], "collision_sphere_buffer": robot_collision_sphere_buffer}
            print(f"[plan_with_grasped_object] Overriding robot collision_sphere_buffer={robot_collision_sphere_buffer} m.")
    robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)
    world_model = world_config if use_world_collision else None
    if not use_world_collision:
        print("[plan_with_grasped_object] use_world_collision=False: not providing world model to MotionGen (no obstacle collision checking).")

    # For pose planning with z variance: use position_threshold for x, y and a large value for z
    # Set position_threshold_z to a large value to allow any z position
    if position_threshold_z is None:
        position_threshold_z = 10.0  # Allow any z by default
    effective_position_threshold = max(position_threshold, position_threshold_z)
    if effective_position_threshold > position_threshold:
        print(f"[plan_with_grasped_object] Using position_threshold={effective_position_threshold:.3f} m (z allowance: position_threshold_z={position_threshold_z:.3f}).")

    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=world_model,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
        position_threshold=effective_position_threshold,
        rotation_threshold=rotation_threshold,
        num_ik_seeds=num_ik_seeds,
        collision_activation_distance=collision_activation_distance,
    )
    motion_gen = MotionGen(motion_gen_cfg)

    q_start = np.asarray(start_joint_position, dtype=np.float64)
    q_start = q_start[:7].flatten()
    print(f"[plan_with_grasped_object] start_joint_position (7): {q_start.tolist()}")
    start_state = JointState.from_position(
        tensor_args.to_device(torch.from_numpy(q_start).unsqueeze(0).float())
    )

    # Capture attached object reference before attach (so we can add it to debug meshes at EE pose)
    attached_obstacle = world_config.get_obstacle(object_name) if world_config else None

    # Attach object to robot
    print(f"[plan_with_grasped_object] Attaching object '{object_name}' to robot at link '{link_name}'...")
    attach_success = motion_gen.attach_objects_to_robot(
        start_state,
        [object_name],
        surface_sphere_radius=surface_sphere_radius,
        link_name=link_name,
        sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        remove_obstacles_from_world_config=remove_obstacles_from_world,
    )
    if not attach_success:
        print(f"[plan_with_grasped_object] Failed to attach object '{object_name}'. Check that it exists in world_config.")
        return False, None
    print(f"[plan_with_grasped_object] Successfully attached object '{object_name}' to robot.")

    # Convert target pose to CuRobo Pose
    pos = np.asarray(target_pose[0], dtype=np.float64)
    quat = np.asarray(target_pose[1], dtype=np.float64)
    print(f"[plan_with_grasped_object] Target pose: position={pos.tolist()}, quat_wxyz={quat.tolist()}")
    print(f"[plan_with_grasped_object] Allowing z variance (position_threshold_z={position_threshold_z:.3f} m), matching x, y, orientation")

    goal_pose = Pose(
        position=tensor_args.to_device(torch.from_numpy(pos).unsqueeze(0).float()),
        quaternion=tensor_args.to_device(torch.from_numpy(quat).unsqueeze(0).float()),
    )

    # Plan directly to pose with z variance enabled
    # The effective_position_threshold allows z variance while still matching x, y, orientation
    plan_cfg = MotionGenPlanConfig(
        pose_cost_metric=PoseCostMetric.reset_metric(),
        max_attempts=max_attempts,
        enable_graph_attempt=False,
    )

    print(f"[plan_with_grasped_object] Planning to pose with z variance (position_threshold={effective_position_threshold:.3f} m)...")
    result = motion_gen.plan_single(start_state, goal_pose, plan_cfg)

    success = bool(result.success.item() if result.success is not None else False)
    status_str = getattr(result.status, "name", str(result.status)) if hasattr(result, "status") and result.status is not None else "N/A"
    print(f"[plan_with_grasped_object] Planning result: success={success}, status={status_str}")

    if success and result.interpolated_plan is not None:
        try:
            traj_js = result.get_interpolated_plan()
            if traj_js is not None and traj_js.position is not None:
                trajectory = np.squeeze(traj_js.position.detach().cpu().numpy())
                print(f"[plan_with_grasped_object] Planning succeeded; trajectory shape: {trajectory.shape}")
                return True, trajectory
        except Exception as e:
            print(f"[plan_with_grasped_object] Warning: failed to extract trajectory despite success: {e}")

    # Planning failed - save debug meshes if requested
    if debug_out_dir is not None:
        tag = f"plan_fail_{int(time.time())}"
        print(f"[plan_with_grasped_object] Planning failed; saving debug meshes to {debug_out_dir}...")
        
        # Save start joint state (include attached object at EE pose)
        try:
            ee_pos, ee_quat = robot_joint_position_to_ee_pose(
                q_start, robot_file=robot_file, tensor_args=tensor_args
            )
            save_world_and_robot_spheres_debug(
                world_config=world_config,
                joint_position=q_start,
                robot_file=robot_file,
                out_dir=debug_out_dir,
                tag=f"{tag}_start",
                attached_obstacle=attached_obstacle,
                attached_obstacle_pose=(ee_pos, ee_quat),
                exclude_obstacle_names=[object_name],
            )
        except Exception as e:
            print(f"[plan_with_grasped_object] Warning: failed to save start debug mesh: {e}")
        
        # Try to compute IK for goal pose to save end joint state debug mesh
        try:
            ik_world = world_model
            ik_cfg = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                world_model=ik_world,
                tensor_args=tensor_args,
                num_seeds=num_ik_seeds,
                position_threshold=effective_position_threshold,  # Use same threshold as planning
                rotation_threshold=rotation_threshold,
                self_collision_check=True,
            )
            ik_solver = IKSolver(ik_cfg)
            retract_cfg = tensor_args.to_device(torch.from_numpy(q_start).unsqueeze(0).float())
            ik_result = ik_solver.solve_single(goal_pose, retract_config=retract_cfg)
            
            if bool(ik_result.success.item() if ik_result.success is not None else False):
                q_sols = ik_result.solution.detach().cpu().numpy()
                q_sols = np.atleast_2d(q_sols)[:, :7]
                q_goal_debug = _pick_nearest_solution(q_sols, q_start)
                ee_pos_end, ee_quat_end = robot_joint_position_to_ee_pose(
                    q_goal_debug, robot_file=robot_file, tensor_args=tensor_args
                )
                save_world_and_robot_spheres_debug(
                    world_config=world_config,
                    joint_position=q_goal_debug,
                    robot_file=robot_file,
                    out_dir=debug_out_dir,
                    tag=f"{tag}_end_ik",
                    attached_obstacle=attached_obstacle,
                    attached_obstacle_pose=(ee_pos_end, ee_quat_end),
                    exclude_obstacle_names=[object_name],
                )
            else:
                print(f"[plan_with_grasped_object] IK failed for debug mesh; skipping end state debug mesh.")
        except Exception as e:
            print(f"[plan_with_grasped_object] Warning: failed to compute IK and save end debug mesh: {e}")

    print("[plan_with_grasped_object] Planning failed; returning False, None.")
    return False, None


if __name__ == "__main__":
    print("curobo_api is a library.")
    print("To test CuRobo on saved data, run: uv run python scripts/curobo_test.py")
    raise SystemExit(0)

