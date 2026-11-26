"""Rank unchanged Contact-GraspNet poses for the TRON2 parallel jaws."""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


# TRON2 official TCP -> centre of the opposing broad jaw faces at 50 mm
# opening, expressed in the TRON2 TCP frame. Positive X points to the
# hand-back, negative Z points along the approach.
_PAD_CENTER_TCP_M = np.array([0.00214, 0.0, -0.01509])


def candidate_tcp_positions(contact_centers, cgn_rotations_base):
    """Map CGN contact midpoints to official TCP positions for both jaw assignments."""
    contacts = np.asarray(contact_centers, dtype=float).reshape(-1, 3)
    rotations = np.asarray(cgn_rotations_base, dtype=float).reshape(-1, 3, 3)
    if len(contacts) != len(rotations):
        raise ValueError("Each CGN contact center needs one CGN rotation")
    # CGN +Y = TRON2 +X (hand-back), CGN +Z = TRON2 -Z (approach).
    # A jaw swap flips the hand-back while keeping the approach unchanged.
    approach_origin = contacts + rotations[:, :, 2] * _PAD_CENTER_TCP_M[2]
    back_offset = rotations[:, :, 1] * _PAD_CENTER_TCP_M[0]
    return np.stack((approach_origin - back_offset, approach_origin + back_offset), axis=1)


@lru_cache(maxsize=1)
def _gripper_envelope():
    """Collision hull across jaw travel in CGN axes (jaw, hand-back, approach)."""
    xml = Path(__file__).resolve().parents[2] / "assets/tron2/DACH_TRON2A/xml/robot_grasper.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    base = model.body("grasper_base_L_Link").id
    bodies = [base, model.body("grasper_L_jaw_left_Link").id, model.body("grasper_L_jaw_right_Link").id]
    geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, bodies) & (model.geom_contype != 0))
    # Official ee link displacement relative to the grasper base link.
    tcp = np.array([-0.0021, 0.0, -0.1313])
    sweep = []
    # Approx. 6 mm aperture increments across the four-bar mechanism's travel.
    for angle in np.linspace(0.0, 0.72, 13):
        for name, sign in (("drive", 1), ("crank_right", 1), ("tip_left", -1), ("tip_right", -1)):
            data.qpos[model.joint(f"grasper_L_{name}_Joint").qposadr[0]] = sign * angle
        mujoco.mj_forward(model, data)
        clouds = []
        half_opening = np.inf
        for geom in geom_ids:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            local = (world - data.xpos[base]) @ data.xmat[base].reshape(3, 3) - tcp
            cgn = local[:, [1, 0, 2]].copy()
            cgn[:, 2] *= -1
            clouds.append(cgn[ConvexHull(cgn).vertices])
            if model.geom_bodyid[geom] != base:
                half_opening = min(half_opening, np.min(np.abs(cgn[:, 0])))
        cloud = np.concatenate(clouds)
        sweep.append((2 * half_opening, cloud[ConvexHull(cloud).vertices]))
    return sweep


def rank_grasps(contact_centers, cgn_rotations_base, scores, object_points_base, table_z):
    """Return (N, 2) scores for the original and jaw-swapped CGN orientations.

    Positions are candidate contact midpoints in the robot base frame. Each
    rotation is the unmodified CGN basis in that frame, before TRON2 conversion.
    Malformed positions, rotations and scores receive -inf. Geometry alters
    preference only. The physical simulation determines contact at runtime.
    """
    pos = np.asarray(contact_centers, dtype=float).reshape(-1, 3)
    rot = np.asarray(cgn_rotations_base, dtype=float).reshape(-1, 3, 3)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(pos) != len(scores) or len(rot) != len(scores):
        raise ValueError("Each position needs a CGN rotation and score")
    result = np.full((len(scores), 2), -np.inf)
    valid = np.isfinite(scores) & np.isfinite(pos).all(axis=1) & np.isfinite(rot).all(axis=(1, 2))
    ids = np.flatnonzero(valid)
    # An SO(3) transform has bounded elements. This also avoids overflow when
    # checking malformed but finite matrices.
    ids = ids[np.max(np.abs(rot[ids]), axis=(1, 2)) <= 1.003]
    r = rot[ids]
    ids = ids[(np.abs(np.linalg.det(r) - 1) < 0.003)
              & (np.max(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3)), axis=(1, 2)) < 0.003)]
    pos, rot, scores = pos[ids], rot[ids], scores[ids]
    result[ids] = scores[:, None]
    points = np.asarray(object_points_base, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(ids) or len(points) < 10:
        return result

    # Fit the upright cube's yaw from the segmented cloud. PCA axes are
    # ambiguous for a square, so minimize the area of a trimmed footprint.
    angles = np.deg2rad(np.arange(0, 90, 3))
    x_axes = np.column_stack((np.cos(angles), np.sin(angles)))
    y_axes = np.column_stack((-np.sin(angles), np.cos(angles)))
    x_bounds = np.percentile(points[:, :2] @ x_axes.T, [2, 98], axis=0)
    y_bounds = np.percentile(points[:, :2] @ y_axes.T, [2, 98], axis=0)
    best = np.argmin(np.ptp(x_bounds, axis=0) * np.ptp(y_bounds, axis=0))
    xy_axes = np.stack((x_axes[best], y_axes[best]))
    xy_bounds = np.column_stack((x_bounds[:, best], y_bounds[:, best]))
    z_bounds = np.percentile(points[:, 2], [2, 98])
    center = np.r_[xy_bounds.mean(axis=0) @ xy_axes, z_bounds.mean()]
    extent = np.r_[np.ptp(xy_bounds, axis=0), np.ptp(z_bounds)]
    if np.min(extent) <= 0.0:
        return result

    jaw, back, approach = (rot[:, :, i] for i in range(3))
    delta = center - pos
    jaw_offset = np.abs(np.sum(delta * jaw, axis=1))
    back_offset = np.abs(np.sum(delta * back, axis=1))
    insertion = np.sum(delta * approach, axis=1)
    jaw_axes = np.abs(jaw[:, :2] @ xy_axes.T)
    width = jaw_axes @ extent[:2] + np.abs(jaw[:, 2]) * extent[2]
    sweep = _gripper_envelope()
    opening_margin = sweep[-1][0] - width - 2 * jaw_offset
    # Test both equivalent assignments of the two jaws. The physical pads
    # have a small asymmetry about the hand-back axis.
    vertical = rot[:, 2, :]
    bottom = np.full((len(ids), 2), np.inf)
    # An object trapped between the jaws stops their travel. Include the
    # next narrower sampled aperture to allow for compliance and sampling.
    for i, (opening, hull) in enumerate(sweep):
        next_opening = sweep[min(i + 1, len(sweep) - 1)][0]
        active = (next_opening >= width - 0.003) | (i == len(sweep) - 1)
        if np.any(active):
            z = vertical[active]
            lower = np.column_stack((np.min(hull @ z.T, axis=0),
                                     np.min(hull @ (z * [-1, -1, 1]).T, axis=0)))
            bottom[active] = np.minimum(bottom[active], lower)
    tcp = candidate_tcp_positions(pos, rot)
    face_alignment = np.max(jaw_axes, axis=1)
    # At 50 mm opening the broad flat pad covers roughly +/- 28 mm around
    # its own center along the approach, with a 30 mm hand-back span.
    opening_penalty = np.maximum(-opening_margin, 0) / 0.012
    insertion_penalty = np.maximum(np.abs(insertion) - 0.029, 0) / 0.012
    # The central pad is about 30 mm across the back axis. Prefer opposed
    # face contacts with balanced closure over clipping one corner of a cube.
    quality = (scores + face_alignment - (back_offset / 0.016) ** 2 - (jaw_offset / 0.012) ** 2
               - opening_penalty - insertion_penalty)
    if np.isfinite(table_z):
        clearance = tcp[:, :, 2] + bottom - table_z
        result[ids] = quality[:, None] + np.minimum(clearance, 0.008) / 0.032
    else:
        result[ids] = quality[:, None]
    return result
