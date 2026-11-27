"""Plan individual left-arm TCP stages against the TRON2 joint limits."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from .official_client import LEFT_WRIST_TO_TCP_M as _WRIST_TO_TCP_M
from .official_client import PROTOCOL_JOINT_NAMES

if TYPE_CHECKING:
    from capx.envs.simulators.tron2_cube_lift import Tron2CubeLiftLowLevel


_MAX_POSITION_ERROR_M = 0.004
_MAX_ORIENTATION_ERROR_RAD = np.deg2rad(3)
_JOINT_LIMIT_MARGIN_RAD = 0.006
_MAX_JOINT_STEP_RAD = 0.35
_MAX_SUBDIVISION_DEPTH = 4
_ENDPOINT_RANDOM_SEEDS = 8


def _within_pose_tolerance(position_error: float, orientation_error: float) -> bool:
    return position_error <= _MAX_POSITION_ERROR_M and orientation_error <= _MAX_ORIENTATION_ERROR_RAD


class _LeftArmIK:
    def __init__(self, env: Tron2CubeLiftLowLevel):
        m = self.model = env.model
        d = self.data = mujoco.MjData(m)
        base = m.body("base_Link").id
        left_shoulder = m.body("proximal_pitch_L_Link").id
        self.wrist = m.body("wrist_roll_L_Link").id
        # Use the measured robot configuration for IK and self-collision checks.
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
        self.robot_geoms = robot[m.geom_bodyid]

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

    def self_collision(self) -> str | None:
        m, d = self.model, self.data
        mujoco.mj_fwdPosition(m, d)
        for contact in d.contact:
            a, b = contact.geom1, contact.geom2
            if contact.dist < -0.0001 and self.robot_geoms[a] and self.robot_geoms[b] and (
                self.arm_geoms[a] or self.arm_geoms[b]
            ):
                first = m.body(m.geom_bodyid[a]).name
                second = m.body(m.geom_bodyid[b]).name
                return f"self-collision between {first} and {second} ({-1000 * contact.dist:.2f} mm penetration)"
        return None

    def _tcp_world_jacobians(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jac(self.model, self.data, self.jac_pos, self.jac_rot,
                      self.world_from_base @ position + self.base_origin, self.wrist)
        return self.jac_pos[:, self.dof_ids], self.jac_rot[:, self.dof_ids]

    def solve(self, position: np.ndarray, rotation: Rotation, seed: np.ndarray):
        rotation_inv = rotation.inv()
        orientation_weight = 0.18  # meters per radian for the least-squares residual

        def residual(q: np.ndarray, jacobian: bool = False) -> np.ndarray:
            pos, rot = self.pose(q)
            error = (Rotation.from_matrix(rot) * rotation_inv).as_rotvec()
            if not jacobian:
                return np.r_[pos - position, orientation_weight * error, 0.0001 * (q - seed)]
            jac_pos, jac_rot = self._tcp_world_jacobians(pos)
            # Inverse left Jacobian of the SO(3) logarithm, for the spatial
            # angular-velocity Jacobian returned by MuJoCo.
            angle = np.linalg.norm(error)
            derivative = np.eye(3)
            if angle > 1e-6:
                x, y, z = error
                skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                derivative += -0.5 * skew + (1 - 0.5 * angle / np.tan(angle / 2)) / angle**2 * (skew @ skew)
            return np.vstack((
                self.world_from_base.T @ jac_pos,
                orientation_weight * derivative @ self.world_from_base.T @ jac_rot,
                0.0001 * np.eye(7),
            ))

        fit = least_squares(
            residual, np.clip(seed, self.lower, self.upper),
            jac=lambda q: residual(q, True), bounds=(self.lower, self.upper),
            max_nfev=45, ftol=1e-8, gtol=1e-8,
        )
        pos, rot = self.pose(fit.x)
        return fit.x, float(np.linalg.norm(pos - position)), float((Rotation.from_matrix(rot) * rotation_inv).magnitude())

    def _endpoint_issue(
        self, position: np.ndarray, rotation: Rotation, previous: np.ndarray, candidate: np.ndarray,
    ) -> str | None:
        """Look for a collision-free endpoint, independently of the path and its IK branch."""
        inverse = rotation.inv()
        tolerances = np.array([_MAX_POSITION_ERROR_M, _MAX_ORIENTATION_ERROR_RAD])

        def errors(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            pos, rot = self.pose(q)
            return pos, pos - position, (Rotation.from_matrix(rot) * inverse).as_rotvec()

        def ratios(q: np.ndarray) -> np.ndarray:
            _, dp, dr = errors(q)
            return np.array([np.linalg.norm(dp), np.linalg.norm(dr)]) / tolerances

        # Minimize the maximum of the two normalized errors. Each tolerance
        # must hold independently, even when the tracking objective trades them off.
        def constraints(x: np.ndarray) -> np.ndarray:
            return x[-1] - ratios(x[:-1])

        def constraint_jacobian(x: np.ndarray) -> np.ndarray:
            pos, dp, dr = errors(x[:-1])
            jac_pos, jac_rot = self._tcp_world_jacobians(pos)
            # The gradient of the rotation angle uses the unit rotation axis
            # directly with the spatial angular Jacobian.
            position_grad = (dp / max(np.linalg.norm(dp), 1e-12)) @ self.world_from_base.T @ jac_pos
            rotation_grad = (dr / max(np.linalg.norm(dr), 1e-12)) @ self.world_from_base.T @ jac_rot
            gradients = np.stack([position_grad, rotation_grad]) / tolerances[:, None]
            return np.column_stack((-gradients, np.ones(2)))

        rng = np.random.default_rng(0)
        sampled = rng.uniform(self.lower, self.upper, size=(_ENDPOINT_RANDOM_SEEDS, len(self.start)))
        seeds = (candidate, previous, self.start, (self.lower + self.upper) / 2, *sampled)
        bounds = [*zip(self.lower, self.upper, strict=True), (0, None)]
        gradient = np.r_[np.zeros(len(self.start)), 1.0]
        closest = np.full(2, np.inf)
        endpoint_collision = None
        for seed in seeds:
            seed = np.clip(seed, self.lower, self.upper)
            initial = ratios(seed)
            if initial.max() < closest.max():
                closest = initial
            if initial.max() <= 1:
                endpoint_collision = self.self_collision()
                if endpoint_collision is None:
                    return None
            with warnings.catch_warnings():
                # Older SciPy releases warn when they clip an internal step.
                warnings.filterwarnings("ignore", "Values in x were outside bounds during a minimize step",
                                        RuntimeWarning, module=r"scipy\.optimize\._slsqp_py")
                fit = minimize(lambda x: x[-1], np.r_[seed, initial.max()], jac=lambda x: gradient,
                               method="SLSQP", bounds=bounds,
                               constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
                               options={"maxiter": 80, "ftol": 1e-7})
            actual = ratios(np.clip(fit.x[:-1], self.lower, self.upper))
            if actual.max() < closest.max():
                closest = actual
            if actual.max() <= 1:
                endpoint_collision = self.self_collision()
                if endpoint_collision is None:
                    return None

        if endpoint_collision is not None:
            return (f"Endpoint IK found no collision-free solution ({endpoint_collision}). "
                    "Change the target orientation or height.")
        pos_error, rot_error = closest * tolerances
        return ("Endpoint IK found no solution within the current joint limits and "
                f"{tolerances[0] * 1000:g} mm / {np.degrees(tolerances[1]):g} deg tolerances "
                f"(best sampled result: {pos_error * 1000:.2f} mm / {np.degrees(rot_error):.2f} deg). "
                "Change the target orientation or height instead of subdividing this move.")

    def plan(self, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        q = self.start.copy()
        start_pos, start_mat = self.pose(q)
        start_rot = Rotation.from_matrix(start_mat)
        target_rot = Rotation.from_matrix(rotation)
        delta_rot = (start_rot.inv() * target_rot).as_rotvec()
        with np.errstate(over="ignore"):
            displacement = position - start_pos
            subdivisions = max(np.linalg.norm(displacement) / 0.024,
                               np.linalg.norm(delta_rot) / np.deg2rad(7.0))
        if not np.isfinite(subdivisions):
            return None, "TCP target is out of range"
        count = max(1, int(np.ceil(subdivisions)))
        path = [q]
        endpoint_found = False

        def advance(left: float, fraction: float, step: int, depth: int = 0) -> str | None:
            nonlocal endpoint_found
            q = path[-1]
            p = start_pos + fraction * displacement
            r = start_rot * Rotation.from_rotvec(fraction * delta_rot)
            q_next, pos_error, rot_error = self.solve(p, r, q)
            pose_ok = _within_pose_tolerance(pos_error, rot_error)
            changes = np.abs(q_next - q)
            joint = int(np.argmax(changes))
            jump = changes[joint] > _MAX_JOINT_STEP_RAD
            if (not pose_ok or jump) and not endpoint_found:
                issue = self._endpoint_issue(position, target_rot, q, q_next)
                if issue is not None:
                    return issue
                endpoint_found = True
            if not pose_ok:
                limited = np.flatnonzero(np.minimum(q_next - self.lower, self.upper - q_next) < 1e-4)
                bounds = ", ".join(PROTOCOL_JOINT_NAMES[i] for i in limited)
                detail = f", at planning bounds: {bounds}" if bounds else ""
                return (f"Endpoint IK found a solution, but straight-path IK failed at waypoint {step}/{count} "
                        f"({pos_error * 1000:.2f} mm / {np.degrees(rot_error):.2f} deg{detail}). "
                        "Change the approach path.")
            if jump:
                if depth == _MAX_SUBDIVISION_DEPTH:
                    return ("Endpoint IK found a solution, but the straight path still exceeds the joint step limit "
                            f"after local refinement near waypoint {step}/{count} "
                            f"({PROTOCOL_JOINT_NAMES[joint]} changes {changes[joint]:.3f} rad, "
                            f"limit {_MAX_JOINT_STEP_RAD:.2f} rad). Change the approach path.")
                mid = (left + fraction) / 2
                return advance(left, mid, step, depth + 1) or advance(mid, fraction, step, depth + 1)
            collision = self.self_collision()
            if collision is not None:
                return f"{collision} at waypoint {step}/{count}"
            path.append(q_next)
            return None

        for step in range(1, count + 1):
            issue = advance((step - 1) / count, step / count, step)
            if issue is not None:
                return None, issue
        return np.stack(path), None


def plan_left_tcp_motion(
    env: Tron2CubeLiftLowLevel, position: np.ndarray, rotation: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Plan this motion from the measured state. Table contacts are simulated during dispatch."""
    return _LeftArmIK(env).plan(position, rotation)


def cgn_approach_waypoints(position: np.ndarray, rotation: np.ndarray, offset: float):
    """Back off along the selected CGN tool axis and clear the object on entry."""
    pre_grasp = position + offset * rotation[:, 2]
    above = pre_grasp.copy()
    above[2] = max(pre_grasp[2], position[2] + 0.10)
    return above, pre_grasp, position
