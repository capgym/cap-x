"""CaP-X TRON2 RGB-D tool bindings for candidate-free visual evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

from capx.integrations.base_api import register_api
from capx.integrations.tron2.grasp_feasibility import plan_left_tcp_motion
from capx.integrations.tron2.left_motion import Tron2LeftMotionApi, _quat_wxyz_to_rotmat
from capx.integrations.vision.sam3 import init_sam3


CAMERAS = ("robot0_robotview", "chest_d435")
# Measured displacement from the official TCP origin to the center of the
# opposing broad pad faces at 50 mm opening, expressed in TCP coordinates.
_PAD_CENTER_TCP_M = np.array([0.00214, 0.0, -0.01509])


def _masked_points_base(view: dict, mask: np.ndarray) -> np.ndarray:
    """Use the RAX projection of observed depth samples into base_Link."""
    depth = np.asarray(view["images"]["depth"], dtype=float)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    k = np.asarray(view["intrinsics"], dtype=float)
    pose = np.asarray(view["pose_mat"], dtype=float)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"SAM3 mask {mask.shape} and depth image {depth.shape} must align")
    if (k.shape != (3, 3) or pose.shape != (4, 4) or min(k[0, 0], k[1, 1]) <= 0
            or not np.isfinite(k).all() or not np.isfinite(pose).all()):
        raise ValueError("Invalid RGB-D camera calibration")
    valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    v, u = np.where(valid)
    z = depth[v, u]
    points = np.column_stack(((u + 0.5 - k[0, 2]) * z / k[0, 0], (v + 0.5 - k[1, 2]) * z / k[1, 1], z))
    return points @ pose[:3, :3].T + pose[:3, 3]


def _visible_bounds(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 10:
        raise ValueError(f"Only {len(points)} valid selected 3D points, need at least 10")
    lower, upper = np.percentile(points, [2, 98], axis=0)
    return {
        "num_points": len(points),
        "bounds_percentiles": [2, 98],
        "bounds_min_m": lower.tolist(),
        "bounds_max_m": upper.tolist(),
        "bounds_midpoint_m": ((lower + upper) * 0.5).tolist(),
        "visible_surface_median_m": np.median(points, axis=0).tolist(),
        "visible_extent_m": (upper - lower).tolist(),
    }


def _xyz(name: str, value: list[float] | np.ndarray) -> np.ndarray:
    try:
        xyz = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return xyz


def _unit_direction(name: str, value: list[float] | np.ndarray) -> np.ndarray:
    direction = _xyz(name, value)
    scale = np.max(np.abs(direction))
    if scale == 0:
        raise ValueError(f"{name} must be nonzero")
    scaled = direction / scale
    return scaled / np.linalg.norm(scaled)


def _left_pad_aperture_m(env) -> float:
    """Measure the inner gap projected onto the gripper's closing axis."""
    model = env.model
    snapshot = env._observation_data()
    base_id = model.body("grasper_base_L_Link").id
    axis = snapshot.xmat[base_id].reshape(3, 3)[:, 1]
    origin = snapshot.xpos[base_id]
    spans = []
    for side in ("left", "right"):
        body = model.body(f"grasper_L_jaw_{side}_Link").id
        geoms = np.flatnonzero(
            (model.geom_bodyid == body) & (model.geom_contype != 0)
            & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        )
        if not len(geoms):
            raise ValueError(f"No collision mesh found for the left gripper's {side} jaw")
        projections = []
        for geom in geoms:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            rotation = snapshot.geom_xmat[geom].reshape(3, 3)
            projections.append(vertices @ (rotation.T @ axis) + (snapshot.geom_xpos[geom] - origin) @ axis)
        spans.append((min(points.min() for points in projections), max(points.max() for points in projections)))
    first, second = sorted(spans, key=sum)
    return max(0.0, float(second[0] - first[1]))


class Tron2RgbdNoCgnHighApi(Tron2LeftMotionApi):
    """Observe RGB-D geometry and check or execute explicit left TCP targets."""

    _motion_log_name = "TRON2 RGB-D"

    def __init__(self, env) -> None:
        super().__init__(env)
        self._segment_text = init_sam3()

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "observe_object": self.observe_object,
            "make_grasp_pose": self.make_grasp_pose,
            "get_left_ee_pose": self.get_left_ee_pose,
            "check_pose_left": self.check_pose_left,
            "goto_pose_left": self.goto_pose_left,
            "open_gripper_left": self.open_gripper_left,
            "close_gripper_left": self.close_gripper_left,
        }

    def get_left_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the left tool origin (TCP) position and WXYZ orientation in base_Link.

        The quaternion rotates TCP-local vectors into base_Link. TCP local -Z
        points toward the fingertips, and local +Y is the signed jaw axis.
        """
        return super().get_left_ee_pose()

    def observe_object(self, object_name: str, cameras: tuple[str, ...] = CAMERAS) -> dict[str, Any]:
        """Segment an object on synchronized metric RGB-D views and estimate visible 3D bounds.

        Args:
            object_name: Text prompt for SAM3 segmentation.
            cameras: Distinct names from robot0_robotview and chest_d435.

        Returns:
            A dictionary with ok, frame (base_Link), and views. If ok is True,
            bounds_min_m, bounds_max_m, bounds_midpoint_m, visible_surface_median_m,
            and visible_extent_m are top-level keys with XYZ lists in meters.
            These are axis-aligned estimates from the visible surfaces of the
            cameras that succeeded. At least one valid view is enough for ok=True.
            nearby_background_z_m is the median Z of nearby unmasked depth points
            or None. It may include other geometry and does not estimate a table
            plane. The views list contains per-camera diagnostics.
        """
        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("object_name must be a nonempty SAM3 text prompt")
        if isinstance(cameras, str):
            cameras = (cameras,)
        if not cameras or any(camera not in CAMERAS for camera in cameras) or len(set(cameras)) != len(cameras):
            raise ValueError(f"Choose distinct RGB-D cameras from {CAMERAS}")

        reports = []
        result = {"frame": "base_Link", "source": "sam3_mask_rgbd", "views": reports}
        try:
            snapshot = self._env._observation_data()
            views = {camera: self._env._get_camera_obs(camera, data=snapshot) for camera in cameras}
        except (OSError, RuntimeError, ValueError) as exc:
            return {**result, "ok": False, "reason": f"RGB-D capture failed: {exc}"}

        clouds = []
        background_z = []
        for camera, view in views.items():
            report = {"camera": camera}
            reports.append(report)
            try:
                detections = [item for item in self._segment_text(view["images"]["rgb"], text_prompt=object_name)
                              if np.isfinite(float(item["score"]))]
                if not detections:
                    raise ValueError("SAM3 found no mask with a finite score")
                best = max(detections, key=lambda item: float(item["score"]))
                mask = np.asarray(best["mask"], dtype=bool).squeeze()
                points = _masked_points_base(view, mask)
                stats = _visible_bounds(points)
                background = _masked_points_base(view, binary_dilation(mask, iterations=10) & ~mask)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                report.update(ok=False, reason=str(exc))
                continue
            clouds.append(points)
            report.update(ok=True, sam3_score=float(best["score"]), mask_pixels=int(mask.sum()), **stats)
            report["nearby_background_z_m"] = float(np.median(background[:, 2])) if len(background) else None
            if len(background):
                background_z.append(background[:, 2])
        if not clouds:
            return {**result, "ok": False, "reason": "No camera supplied enough segmented metric depth"}
        return {
            **result,
            "ok": True,
            **_visible_bounds(np.concatenate(clouds)),
            "nearby_background_z_m": float(np.median(np.concatenate(background_z))) if background_z else None,
        }

    @staticmethod
    def make_grasp_pose(
        contact_center_m: list[float] | np.ndarray,
        finger_direction: list[float] | np.ndarray,
        jaw_direction: list[float] | np.ndarray,
    ) -> tuple[list[float], list[float]]:
        """Convert a chosen pad midpoint and tool directions into a TCP pose.

        Args:
            contact_center_m: Desired XYZ midpoint between the opposing finger
                pads in base_Link, in meters.
            finger_direction: Direction from the palm toward the fingertips in
                base_Link. This becomes TCP -Z.
            jaw_direction: Signed axis across the jaws in base_Link. Its
                projection perpendicular to finger_direction becomes TCP +Y.

        Returns:
            Tuple (tcp_xyz, quat_wxyz), both Python lists. The TCP position
            already accounts for the nominal pad midpoint calibrated at a
            50 mm opening. Keep the returned position and orientation paired
            when targeting that pad midpoint.

        Raises:
            ValueError: Inputs must be finite XYZ vectors with nonzero,
                nonparallel directions.

        Example:
            tcp_xyz, quat_wxyz = make_grasp_pose(contact_center_m, finger_direction, jaw_direction)
            reached = goto_pose_left(tcp_xyz, quat_wxyz, stage="grasp")
        """
        center = _xyz("contact_center_m", contact_center_m)
        z = -_unit_direction("finger_direction", finger_direction)
        jaw = _unit_direction("jaw_direction", jaw_direction)
        x = np.cross(jaw, z)
        length = np.linalg.norm(x)
        if length < 1e-6:
            raise ValueError("jaw_direction must not be parallel to finger_direction")
        x = x / length
        y = np.cross(z, x)
        rotation = np.column_stack((x, y, z))
        tcp_xyz = center - rotation @ _PAD_CENTER_TCP_M
        quat_xyzw = Rotation.from_matrix(rotation).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return tcp_xyz.tolist(), quat_wxyz.tolist()

    def check_pose_left(
        self, pos: list[float] | np.ndarray, quat_wxyz: list[float] | np.ndarray | None = None,
    ) -> dict[str, bool | str | None]:
        """Check one straight Cartesian TCP path without moving the left arm.

        Every call starts from the robot's current measured state. A previous
        check has no effect on the start of the next check.

        Args:
            pos: Target XYZ in meters in base_Link.
            quat_wxyz: Target WXYZ orientation. Omit or pass None to fix the
                target orientation to the one measured at the start of this call.

        Returns:
            A dictionary with plan_ok (bool) and reason (None on success, a
            string on failure). A successful check covers joint limits,
            continuity, and robot self-collision at the planned waypoints.
            Table and object clearance and motion tracking are separate.
        """
        target = np.asarray(pos, dtype=float).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError("The requested TCP position must be finite")
        if quat_wxyz is None:
            quat_wxyz = self.get_left_ee_pose()[1]
        quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
        scale = np.max(np.abs(quat))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("The requested orientation must be a finite, nonzero WXYZ quaternion")
        quat = quat / scale
        quat /= np.linalg.norm(quat)
        path, reason = plan_left_tcp_motion(self._env, target, _quat_wxyz_to_rotmat(quat))
        return {"plan_ok": path is not None, "reason": reason}

    def goto_pose_left(
        self, pos: list[float] | np.ndarray, quat_wxyz: list[float] | np.ndarray | None = None,
        duration_sec: float = 2.0, *, stage: str = "free-motion",
    ) -> bool:
        """Move to one explicit base_Link TCP target through checked ServoJ.

        Args:
            pos: Target XYZ position in meters.
            quat_wxyz: Target WXYZ orientation. Omit or pass None to fix the
                target orientation to the one measured at the start of this call.
                Local -Z points along the fingertips and local Y spans the jaws.
            duration_sec: Requested motion duration in seconds.
            stage: Descriptive motion label for the log.

        Returns:
            True if the measured endpoint reaches the target. Planning or controller
            rejections and missed endpoints return False with motion feedback.
            A failed motion can leave the arm at a different pose.
        """
        return super().goto_pose_left(pos, quat_wxyz, duration_sec=duration_sec, stage=stage)

    def open_gripper_left(self) -> None:
        """Open the left parallel gripper fully."""
        with self._env.motion_feedback("open"):
            self._env.set_gripper("left", 1.0)

    def close_gripper_left(self) -> None:
        """Command closure and log the measured inner jaw gap in millimeters.

        Returns None. The gap measures jaw separation. Verify object motion
        separately to confirm a grasp.
        """
        with self._env.motion_feedback("close"):
            self._env.set_gripper("left", 0.0)
        aperture_mm = 1000.0 * _left_pad_aperture_m(self._env)
        print(f"[TRON2 RGB-D] LEFT_GRIPPER_APERTURE measured_aperture_mm={aperture_mm:.1f}")


register_api("Tron2RgbdNoCgnHighApi", Tron2RgbdNoCgnHighApi)
