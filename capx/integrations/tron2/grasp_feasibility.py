"""Check left-arm CGN candidates against the TRON2 model and controller limits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .official_client import PROTOCOL_JOINT_NAMES

if TYPE_CHECKING:
    from capx.envs.simulators.tron2_cube_lift import Tron2CubeLiftLowLevel


# Fixed L_ee_Joint relative to wrist_roll_L_Link in robot_zeromass_graser.urdf.
_WRIST_TO_TCP_M = np.array([-0.0338, 0.0, -0.2125])
_JAW_SYMMETRY = np.diag([-1.0, -1.0, 1.0])
_MAX_POSITION_ERROR_M = 0.004
_MAX_ORIENTATION_ERROR_RAD = np.deg2rad(3)
_JOINT_LIMIT_MARGIN_RAD = 0.006


@dataclass
class FeasibleGrasp:
    index: int
    position: np.ndarray
    rotation: np.ndarray
    joint_paths: tuple[np.ndarray, ...]
    max_position_error_m: float
    max_orientation_error_rad: float

    @property
    def waypoint_joints(self) -> np.ndarray:
        return np.stack([segment[-1] for segment in self.joint_paths])


class _LeftArmIK:
    def __init__(self, env: Tron2CubeLiftLowLevel):
        m = self.model = env.model
        d = self.data = mujoco.MjData(m)
        base = m.body("base_Link").id
        left_shoulder = m.body("proximal_pitch_L_Link").id
        table = m.body("table").id
        self.wrist = m.body("wrist_roll_L_Link").id
        # Copy only the robot configuration. The workpiece pose is never used
        # to generate, rank or check a candidate.
        robot = np.zeros(m.nbody, dtype=bool)
        left_arm = np.zeros_like(robot)
        for body in range(1, m.nbody):
            parent = m.body_parentid[body]
            robot[body] = body == base or robot[parent]
            left_arm[body] = body == left_shoulder or left_arm[parent]
        qpos_width = np.diff(np.append(m.jnt_qposadr, m.nq))
        robot_qpos = np.repeat(robot[m.jnt_bodyid], qpos_width)
        d.qpos[robot_qpos] = env._observation_data().qpos[robot_qpos]
        self.arm_geoms = left_arm[m.geom_bodyid]
        self.obstacle_geoms = robot[m.geom_bodyid] | (m.geom_bodyid == table)

        joints = [m.joint(name).id for name in PROTOCOL_JOINT_NAMES[:7]]
        self.qpos_ids = m.jnt_qposadr[joints]
        self.dof_ids = m.jnt_dofadr[joints]
        self.lower, self.upper = m.jnt_range[joints].copy().T
        # Intersect the simulation/URDF bounds with the MoveJ command bounds
        # in the arm test stack (dual_arm_protocol_tests.robot_model).
        self.lower[1] = max(self.lower[1], 0.0)
        self.upper[1] = min(self.upper[1], 2.96706)
        self.lower[-2:] = np.maximum(self.lower[-2:], -0.7505)
        self.upper[-2:] = np.minimum(self.upper[-2:], 0.7505)
        # The controller can acknowledge a target on its soft stop without
        # moving to it. Keep every commanded joint angle inside the range.
        self.lower += _JOINT_LIMIT_MARGIN_RAD
        self.upper -= _JOINT_LIMIT_MARGIN_RAD
        self.start = d.qpos[self.qpos_ids].copy()
        mujoco.mj_kinematics(m, d)
        self.world_from_base = d.xmat[base].reshape(3, 3).copy()
        self.base_origin = d.xpos[base].copy()
        self.jac_pos = np.zeros((3, m.nv))
        self.jac_rot = np.zeros_like(self.jac_pos)

    def pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        d.qpos[self.qpos_ids] = q
        mujoco.mj_kinematics(self.model, d)
        wrist_rot = d.xmat[self.wrist].reshape(3, 3)
        tcp_world = d.xpos[self.wrist] + wrist_rot @ _WRIST_TO_TCP_M
        return (
            self.world_from_base.T @ (tcp_world - self.base_origin),
            self.world_from_base.T @ wrist_rot,
        )

    def has_collision(self) -> bool:
        mujoco.mj_fwdPosition(self.model, self.data)
        for contact in self.data.contact:
            a, b = contact.geom1, contact.geom2
            if contact.dist < -0.0001 and (
                (self.arm_geoms[a] and self.obstacle_geoms[b])
                or (self.arm_geoms[b] and self.obstacle_geoms[a])
            ):
                return True
        return False

    def solve(self, position: np.ndarray, rotation: Rotation, seed: np.ndarray):
        rotation_inv = rotation.inv()
        orientation_weight = 0.18  # meters per radian for the least-squares residual

        def residual(q: np.ndarray, jacobian: bool = False) -> np.ndarray:
            pos, rot = self.pose(q)
            error = (Rotation.from_matrix(rot) * rotation_inv).as_rotvec()
            if not jacobian:
                return np.r_[pos - position, orientation_weight * error, 0.0001 * (q - seed)]
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jac(self.model, self.data, self.jac_pos, self.jac_rot,
                          self.world_from_base @ pos + self.base_origin, self.wrist)
            # Inverse left Jacobian of the SO(3) logarithm, for the spatial
            # angular-velocity Jacobian returned by MuJoCo.
            angle = np.linalg.norm(error)
            derivative = np.eye(3)
            if angle > 1e-6:
                x, y, z = error
                skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                derivative += -0.5 * skew + (1 - 0.5 * angle / np.tan(angle / 2)) / angle**2 * (skew @ skew)
            return np.vstack((
                self.world_from_base.T @ self.jac_pos[:, self.dof_ids],
                orientation_weight * derivative @ self.world_from_base.T @ self.jac_rot[:, self.dof_ids],
                0.0001 * np.eye(7),
            ))

        fit = least_squares(
            residual, np.clip(seed, self.lower, self.upper),
            jac=lambda q: residual(q, True), bounds=(self.lower, self.upper),
            max_nfev=45, ftol=1e-8, gtol=1e-8,
        )
        pos, rot = self.pose(fit.x)
        return fit.x, float(np.linalg.norm(pos - position)), float((Rotation.from_matrix(rot) * rotation_inv).magnitude())

    def _verify_joint_paths(
        self, rotation: np.ndarray, waypoints: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], float, float] | None:
        q = self.start.copy()
        start_pos, start_mat = self.pose(q)
        start_rot = Rotation.from_matrix(start_mat)
        goal_rot = Rotation.from_matrix(rotation)
        segments = []
        max_position_error = max_rotation_error = 0.0
        for position in waypoints:
            delta_rot = (start_rot.inv() * goal_rot).as_rotvec()
            with np.errstate(over="ignore"):
                subdivisions = max(
                    np.linalg.norm(position - start_pos) / 0.024,
                    np.linalg.norm(delta_rot) / np.deg2rad(7.0),
                )
            if not np.isfinite(subdivisions):
                return None
            count = max(1, int(np.ceil(subdivisions)))
            segment = [q.copy()]
            for step in range(1, count + 1):
                fraction = step / count
                p = start_pos + fraction * (position - start_pos)
                r = start_rot * Rotation.from_rotvec(fraction * delta_rot)
                q_next, pos_error, rot_error = self.solve(p, r, q)
                if (pos_error > _MAX_POSITION_ERROR_M or rot_error > _MAX_ORIENTATION_ERROR_RAD
                        or np.max(np.abs(q_next - q)) > 0.35 or self.has_collision()):
                    return None
                max_position_error = max(max_position_error, pos_error)
                max_rotation_error = max(max_rotation_error, rot_error)
                segment.append(q_next)
                q = q_next
            segments.append(np.stack(segment))
            start_pos, start_rot = position, goal_rot
        return tuple(segments), max_position_error, max_rotation_error

    def verify_path(self, index: int, rotation: np.ndarray, waypoints: np.ndarray) -> FeasibleGrasp | None:
        checked = self._verify_joint_paths(rotation, waypoints)
        if checked is None:
            return None
        paths, position_error, orientation_error = checked
        return FeasibleGrasp(index, waypoints[2].copy(), rotation.copy(), paths, position_error, orientation_error)

    def verify_lift(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray | None:
        start_pos, _ = self.pose(self.start)
        if position[2] <= start_pos[2]:
            return None
        checked = self._verify_joint_paths(rotation, position[None, :])
        return checked[0][0] if checked is not None else None


def plan_checked_left_lift(
    env: Tron2CubeLiftLowLevel, position: np.ndarray, rotation: np.ndarray,
) -> np.ndarray | None:
    """Check an upward lift from the measured left-arm state and current gripper geometry."""
    return _LeftArmIK(env).verify_lift(position, rotation)


def cgn_waypoints(positions, rotations, approach_offset, lift_height):
    """Back off along each tool axis and clear the workpiece during entry."""
    pre = positions + approach_offset * rotations[..., :, 2]
    above = pre.copy()
    # A steep CGN approach already places the pre-grasp above the object.
    above[..., 2] = np.maximum(pre[..., 2], positions[..., 2] + 0.10)
    return np.stack((above, pre, positions, positions + [0., 0., lift_height]), axis=-2)


def iter_feasible_grasps(
    env: Tron2CubeLiftLowLevel,
    positions_base: np.ndarray,
    rotations_base: np.ndarray,
    scores: np.ndarray,
    *,
    approach_offset: float = 0.11,
    lift_height: float = 0.16,
):
    """Yield ranked CGN candidates with continuous paths in base_Link coordinates.

    The segments run from the current configuration to above the pre-grasp,
    then to the pre-grasp, contact pose, and lifted pose. Each segment includes
    both endpoints, in the seven-joint protocol order. Rotations are already
    converted from the CGN frame to the TRON2 TCP frame by the caller.
    Each proposal is yielded once, even if both jaw assignments have a path.
    """
    positions = np.asarray(positions_base, dtype=float)
    rotations = np.asarray(rotations_base, dtype=float).reshape(-1, 3, 3)
    scores = np.asarray(scores, dtype=float)
    if positions.shape != (len(rotations), 2, 3) or scores.shape != (len(rotations), 2):
        raise ValueError("Expected two calibrated TCP positions and two symmetry scores per candidate")
    if not np.isfinite([approach_offset, lift_height]).all() or approach_offset <= 0 or lift_height <= 0:
        raise ValueError("Pre-grasp distance and lift height must be positive")
    valid_branches = np.isfinite(positions).all(axis=2) & np.isfinite(scores)
    valid = np.isfinite(rotations).all(axis=(1, 2)) & valid_branches.any(axis=1)
    # Invalid transforms cannot be repaired without changing the proposed pose.
    ids = np.flatnonzero(valid)
    r = rotations[ids]
    ids = ids[(np.abs(np.linalg.det(r) - 1) < 0.003)
              & (np.max(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3)), axis=(1, 2)) < 0.003)]
    if not len(ids):
        return
    ik = _LeftArmIK(env)
    # A deterministic FK sample provides seeds for contact screening and a
    # tie-breaker between the two equivalent jaw orientations. Keep the
    # caller's grasp-quality order across different proposals.
    seeds = np.random.default_rng(0).uniform(ik.lower, ik.upper, (4096, 7))
    seeds[0] = np.clip(ik.start, ik.lower, ik.upper)
    cloud = np.empty((len(seeds), 12))
    for k, q in enumerate(seeds):
        p, rot = ik.pose(q)
        cloud[k] = np.r_[p, 0.12 * rot.ravel()]
    tree = cKDTree(cloud)
    row, branch = np.where(valid_branches[ids])
    source = ids[row]
    orientations = rotations[source].copy()
    orientations[branch == 1] = orientations[branch == 1] @ _JAW_SYMMETRY
    contact = positions[source, branch]
    stages = cgn_waypoints(contact, orientations, approach_offset, lift_height)
    query = np.empty((len(source), 4, 12))
    query[:, :, :3] = stages
    query[:, :, 3:] = 0.12 * orientations.reshape(-1, 1, 9)
    distances, nearest = tree.query(query)
    order = np.lexsort((np.max(distances, axis=1), -scores[source, branch]))
    feasible_indices: set[int] = set()
    for k in order:
        index = int(source[k])
        if index in feasible_indices:
            continue
        r = Rotation.from_matrix(orientations[k])
        # Check contact first. A grasp pose with no local fit to its closest
        # workspace sample gets a second attempt from the current joint state.
        for seed in (seeds[nearest[k, 2]], ik.start):
            _, ep, er = ik.solve(contact[k], r, seed)
            if ep <= _MAX_POSITION_ERROR_M and er <= _MAX_ORIENTATION_ERROR_RAD and not ik.has_collision():
                break
        else:
            continue
        result = ik.verify_path(index, orientations[k], stages[k])
        if result is not None:
            feasible_indices.add(index)
            yield result


def select_feasible_grasp(
    env: Tron2CubeLiftLowLevel,
    positions_base: np.ndarray,
    rotations_base: np.ndarray,
    scores: np.ndarray,
    *,
    approach_offset: float = 0.11,
    lift_height: float = 0.16,
    candidate_rank: int = 0,
) -> FeasibleGrasp | None:
    """Return the requested zero-based feasible CGN candidate from one batch."""
    plans = iter_feasible_grasps(env, positions_base, rotations_base, scores,
                                approach_offset=approach_offset, lift_height=lift_height)
    return next(islice(plans, candidate_rank, None), None)
