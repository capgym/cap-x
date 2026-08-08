from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.integrations.g1.sdk import dex3_grasp_joints, normalize_g1_arm_side


DEFAULT_G1_WITH_HAND_URDF = Path(__file__).resolve().parents[3] / "env_configs/g1/g1_29dof_with_hand.urdf"
DEFAULT_PINCH_PREGRASP_DISTANCE_M = 0.20
LEGACY_GRASP_LOCAL_Z_OFFSET_M = 0.12
# Fixed front-table palm orientation for G1 Dex3 grasps. This maps the palm local
# +X axis to world +X so the fingers point forward, local +Y to world +Y so the
# thumb side stays on the robot left, and local +Z to world +Z. The palm XZ
# plane is therefore vertical to the world XY table plane.
FRONT_POLICY_QUATERNION_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

_DISTAL_TIP_LINKS = ("thumb_2", "index_1", "middle_1")
_DISTAL_AXIS_BY_LINK_KEY = {
    "thumb_2": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "index_1": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "middle_1": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
}
_DISTAL_FALLBACK_LENGTH_M = 0.0458
_TIP_MESH_CAP_M = 0.004

def closed_pinch_center_offset(
    *,
    hand: str = "right",
    link_frame: str = "right_hand_palm_link",
    urdf_path: str | Path = DEFAULT_G1_WITH_HAND_URDF,
) -> np.ndarray:
    """Return the closed Dex3 pinch center offset in a G1 hand link frame.

    Args:
        hand: "left" or "right", selecting the Dex3 geometry used for the pinch center.
        link_frame: The local frame for the returned offset. It must match the PyRoKi target link
            selected for this arm.
        urdf_path: G1 URDF with Dex3 hand joints and links.

    Returns:
        (3,) XYZ offset from link_frame to the center of the closed thumb, index,
        and middle fingertips, in meters. This is the point that should land on a
        visual grasp center when the hand closes.
    """

    hand_key = _normalized_g1_hand(hand)
    return np.asarray(
        _cached_closed_pinch_center_offset(hand_key, str(link_frame), str(Path(urdf_path))),
        dtype=np.float64,
    ).copy()


@lru_cache(maxsize=16)
def _cached_closed_pinch_center_offset(hand: str, link_frame: str, urdf_path: str) -> tuple[float, float, float]:
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    transforms = _link_transforms_from(
        root,
        base_link=link_frame,
        joint_values=_closed_joint_values(hand),
    )
    tip_points = []
    for key in _DISTAL_TIP_LINKS:
        link_name = f"{hand}_hand_{key}_link"
        if link_name not in transforms:
            raise ValueError(f"Link {link_name!r} is not reachable from {link_frame!r} in {path}.")
        tip_local = _distal_tip_local(root, path, link_name, key)
        tip_h = np.concatenate([tip_local, [1.0]])
        tip_points.append((transforms[link_name] @ tip_h)[:3])
    center = np.mean(np.stack(tip_points, axis=0), axis=0)
    return tuple(float(item) for item in center)

def palm_pose_from_pinch_center_pose(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray | None = None,
    *,
    pinch_center_offset: np.ndarray | None = None,
    urdf_path: str | Path = DEFAULT_G1_WITH_HAND_URDF,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a desired closed pinch-center pose into a palm-link pose.

    Args:
        position: Desired closed Dex3 pinch center XYZ in the robot/world frame.
        quaternion_wxyz: Desired pinch-center orientation in WXYZ order. If None,
            identity orientation is used.
        pinch_center_offset: Optional (3,) palm-link-local offset from
            palm to closed pinch center. Tests can inject this; real code uses URDF.
        urdf_path: URDF used to compute the closed pinch-center offset when the
            offset is not provided.

    Returns:
        palm_position, palm_quaternion_wxyz: pose to send to PyRoKi for
        palm-link frame so the closed pinch center lands at position.
    """

    offset = (
        closed_pinch_center_offset(urdf_path=urdf_path)
        if pinch_center_offset is None
        else np.asarray(pinch_center_offset, dtype=np.float64).reshape(3)
    )
    t_world_pinch = pose_to_matrix(position, _identity_quat_if_none(quaternion_wxyz))
    t_palm_pinch = np.eye(4, dtype=np.float64)
    t_palm_pinch[:3, 3] = offset
    t_world_palm = t_world_pinch @ invert_transform(t_palm_pinch)
    return matrix_to_pose(t_world_palm)

def front_policy_pinch_target_pose(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray | None = None,
    *,
    visual_points: np.ndarray | None = None,
    contact_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the G1 front-grasp policy to a visual/contact grasp target.

    Args:
        position: Fallback grasp center XYZ in the robot/world frame.
        quaternion_wxyz: Ignored by this policy. The returned orientation is a
            fixed vertical front-palm pose for table-top grasps.
        visual_points: Optional object mask point cloud in the robot/world frame.
            Its Z bounding-box center is used as the target pinch height.
        contact_points: Optional ContactGraspNet/AnyGrasp contact points in the
            robot/world frame. Their XY centroid is used as the pinch target XY.

    Returns:
        position, quaternion_wxyz: front-policy closed pinch-center target. XY
        comes from contact points when available, otherwise visual centroid or the
        fallback position. Z comes from visual bbox center when available,
        otherwise contact centroid or the fallback position. Quaternion is fixed
        so the fingers point forward and the thumb side stays left.
    """

    del quaternion_wxyz
    target = np.asarray(position, dtype=np.float64).reshape(3).copy()
    contact_center = _finite_centroid(contact_points)
    visual = _finite_points(visual_points)
    visual_center = None if visual is None else np.mean(visual, axis=0)

    xy_source = contact_center if contact_center is not None else visual_center
    if xy_source is not None:
        target[:2] = xy_source[:2]

    if visual is not None:
        target[2] = float((np.min(visual[:, 2]) + np.max(visual[:, 2])) / 2.0)
    elif contact_center is not None:
        target[2] = float(contact_center[2])

    return target, FRONT_POLICY_QUATERNION_WXYZ.copy()

def front_pregrasp_pinch_center_pose(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray | None = None,
    *,
    pregrasp_distance: float = DEFAULT_PINCH_PREGRASP_DISTANCE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the front pregrasp pinch-center pose used by the G1 bottle policy.

    Args:
        position: Final visual grasp center XYZ in the robot/world frame.
        quaternion_wxyz: Final grasp orientation in WXYZ order. If None, the
            fixed vertical front-palm orientation is used.
        pregrasp_distance: Distance in meters to retreat along world -X before the
            final grasp motion. The G1 bottle default is 0.20 m. This mirrors the front-table strategy from the
            IsaacLab G1 manipulation code.

    Returns:
        pregrasp_position, pregrasp_quaternion_wxyz for the closed pinch center.
    """

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    quat = (
        FRONT_POLICY_QUATERNION_WXYZ.copy()
        if quaternion_wxyz is None
        else _identity_quat_if_none(quaternion_wxyz)
    )
    pregrasp = pos - np.asarray([float(pregrasp_distance), 0.0, 0.0], dtype=np.float64)
    return pregrasp, quat

def remove_legacy_grasp_local_z_offset(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    *,
    distance: float = LEGACY_GRASP_LOCAL_Z_OFFSET_M,
) -> np.ndarray:
    """Undo the inherited Franka sample_grasp_pose local +Z palm offset.

    Args:
        position: Offset pose XYZ returned by the legacy sample_grasp_pose.
        quaternion_wxyz: Pose quaternion in WXYZ order.
        distance: Local +Z distance to subtract. The inherited implementation uses
            0.12 m before transforming to world.

    Returns:
        Raw visual grasp center XYZ before the legacy local +Z offset.
    """

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    rot = rotation_from_wxyz(quaternion_wxyz)
    return pos - rot.apply(np.asarray([0.0, 0.0, float(distance)], dtype=np.float64))

def _finite_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0 or arr.shape[-1] != 3:
        return None
    arr = arr.reshape((-1, 3))
    finite = arr[np.isfinite(arr).all(axis=1)]
    if finite.size == 0:
        return None
    return finite

def _finite_centroid(points: np.ndarray | None) -> np.ndarray | None:
    finite = _finite_points(points)
    if finite is None:
        return None
    return np.mean(finite, axis=0)

def pose_to_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    rot = rotation_from_wxyz(quaternion_wxyz)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = pos
    return mat

def matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    quat_xyzw = SciRotation.from_matrix(mat[:3, :3]).as_quat()
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return mat[:3, 3].astype(np.float64).copy(), quat_wxyz

def invert_transform(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = mat[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ mat[:3, 3]
    return inv

def rotation_from_wxyz(quaternion_wxyz: np.ndarray) -> SciRotation:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be non-zero.")
    quat = quat / norm
    return SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]])

def _identity_quat_if_none(quaternion_wxyz: np.ndarray | None) -> np.ndarray:
    if quaternion_wxyz is None:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be non-zero.")
    return quat / norm

def _normalized_g1_hand(hand: str) -> str:
    return normalize_g1_arm_side(hand)

def _closed_joint_values(hand: str) -> dict[str, float]:
    values = dex3_grasp_joints(trigger=1.0, squeeze=1.0, hand_side=hand)
    return {
        f"{hand}_hand_thumb_0_joint": float(values[0]),
        f"{hand}_hand_thumb_1_joint": float(values[1]),
        f"{hand}_hand_thumb_2_joint": float(values[2]),
        f"{hand}_hand_index_0_joint": float(values[3]),
        f"{hand}_hand_index_1_joint": float(values[4]),
        f"{hand}_hand_middle_0_joint": float(values[5]),
        f"{hand}_hand_middle_1_joint": float(values[6]),
    }

def _link_transforms_from(root: ET.Element, *, base_link: str, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
    child_joints: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is None or "link" not in parent.attrib:
            continue
        child_joints.setdefault(parent.attrib["link"], []).append(joint)

    transforms = {base_link: np.eye(4, dtype=np.float64)}
    stack = [base_link]
    while stack:
        parent_link = stack.pop()
        parent_transform = transforms[parent_link]
        for joint in child_joints.get(parent_link, []):
            child = joint.find("child")
            if child is None or "link" not in child.attrib:
                continue
            child_link = child.attrib["link"]
            transforms[child_link] = parent_transform @ _joint_transform(joint, joint_values)
            stack.append(child_link)
    return transforms

def _joint_transform(joint: ET.Element, joint_values: dict[str, float]) -> np.ndarray:
    transform = _origin_transform(joint.find("origin"))
    if joint.attrib.get("type") == "revolute":
        axis_element = joint.find("axis")
        axis = _float_triplet(
            None if axis_element is None else axis_element.attrib.get("xyz"),
            default=[1.0, 0.0, 0.0],
        )
        transform = transform @ _axis_angle_transform(axis, joint_values.get(joint.attrib["name"], 0.0))
    return transform

def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    xyz = _float_triplet(None if origin is None else origin.attrib.get("xyz"), default=[0.0, 0.0, 0.0])
    rpy = _float_triplet(None if origin is None else origin.attrib.get("rpy"), default=[0.0, 0.0, 0.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform

def _axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_arr))
    if norm <= 1e-12:
        raise ValueError("URDF revolute joint axis cannot be zero.")
    axis_arr = axis_arr / norm
    x, y, z = axis_arr
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    rotation = np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform

def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(item) for item in np.asarray(rpy, dtype=np.float64).reshape(3)]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx

def _float_triplet(value: str | None, *, default: list[float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    parts = [float(item) for item in value.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected three floats, got {value!r}.")
    return np.asarray(parts, dtype=np.float64)

def _distal_tip_local(root: ET.Element, urdf_path: Path, link_name: str, link_key: str) -> np.ndarray:
    axis = _DISTAL_AXIS_BY_LINK_KEY[link_key]
    mesh_tip = _mesh_tip_local(root, urdf_path, link_name, axis)
    if mesh_tip is not None:
        return mesh_tip
    return axis * _DISTAL_FALLBACK_LENGTH_M

def _mesh_tip_local(root: ET.Element, urdf_path: Path, link_name: str, axis: np.ndarray) -> np.ndarray | None:
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        return None
    vertices = []
    for mesh in link.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_path = _resolve_mesh_path(filename, urdf_path)
        if mesh_path is None:
            continue
        mesh_vertices = _stl_vertices(mesh_path)
        if mesh_vertices is not None and mesh_vertices.size:
            vertices.append(mesh_vertices)
    if not vertices:
        return None
    verts = np.concatenate(vertices, axis=0)
    axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
    projection = verts @ axis_arr
    tip_projection = float(np.max(projection))
    cap = verts[projection >= tip_projection - _TIP_MESH_CAP_M]
    if cap.size == 0:
        cap = verts[np.argmax(projection)].reshape(1, 3)
    return np.mean(cap, axis=0)

def _resolve_mesh_path(filename: str, urdf_path: Path) -> Path | None:
    clean = filename
    if clean.startswith("package://"):
        clean = clean.split("/", 3)[-1]
    raw_path = Path(clean)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    candidates: list[Path] = []
    for parent in (urdf_path.parent, *urdf_path.parents):
        candidates.append(parent / raw_path)
        candidates.append(parent / "capx" / "envs" / "assets" / "g1_description" / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def _stl_vertices(path: Path) -> np.ndarray | None:
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected = 84 + tri_count * 50
        if expected == len(data):
            vertices = np.empty((tri_count * 3, 3), dtype=np.float64)
            offset = 84
            for tri_index in range(tri_count):
                values = struct.unpack("<12f", data[offset : offset + 48])
                vertices[tri_index * 3 : tri_index * 3 + 3] = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)
                offset += 50
            return vertices
    try:
        text = data.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        return None
    points = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("vertex "):
            continue
        points.append([float(item) for item in stripped.split()[1:4]])
    if not points:
        return None
    return np.asarray(points, dtype=np.float64).reshape((-1, 3))
