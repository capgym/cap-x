"""
Solves the basic IK problem.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax import Array
from jaxls import Cost, Var, VarValues


@Cost.create_factory
def limit_velocity_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[Array],
    prev_cfg: Array,
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint velocity limit violations."""
    joint_vel = (vals[joint_var] - prev_cfg) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


def solve_ik_with_multiple_targets(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    target_wxyzs: onp.ndarray,
    target_positions: onp.ndarray,
    prev_cfg: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_names: Sequence[str]. List of link names to be controlled.
        target_wxyzs: onp.ndarray. Shape: (num_targets, 4). Target orientations.
        target_positions: onp.ndarray. Shape: (num_targets, 3). Target positions.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_wxyzs),
        jnp.array(target_positions),
        jnp.array(target_link_indices),
        jnp.array(prev_cfg),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    target_joint_indices: jax.Array,
    prev_cfg: jax.Array,
) -> jax.Array:
    JointVar = robot.joint_var_cls

    # Get the batch axes for the variable through the target pose.
    # Batch axes for the variables and cost terms (e.g., target pose) should be broadcastable!
    target_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position)
    batch_axes = target_pose.get_batch_axes()

    factors = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            target_pose,
            target_joint_indices,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=JointVar.default_factory(),
            weight=2.0,
        ),
        pk.costs.limit_cost(
            robot,
            JointVar(0),
            jnp.array([100.0] * robot.joints.num_joints),
        ),
        limit_velocity_cost(
            robot,
            JointVar(0),
            prev_cfg,
            0.01,  # dt
            10.0,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]
