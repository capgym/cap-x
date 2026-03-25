from __future__ import annotations

import types

import numpy as np
import pytest

from capx.integrations.motion.pyroki import init_pyroki
from capx.tools import tool_api


class _DefaultFactory:
    def __call__(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float64)


class _JointVarCls:
    def __init__(self) -> None:
        self.default_factory = _DefaultFactory()


class FakeRobot:
    def __init__(self) -> None:
        self.joint_var_cls = _JointVarCls()


class FakeCollision:
    @staticmethod
    def from_urdf(_urdf: object) -> object:
        return object()


def test_init_pyroki_registers_ik_and_plan(monkeypatch: object) -> None:
    import capx.integrations.motion.pyroki as integ

    # Monkeypatch pk and pks
    class PKModule(types.SimpleNamespace):
        pass

    pk = PKModule()

    class CollisionNS(types.SimpleNamespace):
        pass

    pk.Robot = types.SimpleNamespace(from_urdf=lambda urdf: FakeRobot())
    pk.collision = CollisionNS()
    pk.collision.RobotCollision = FakeCollision
    pk.collision.HalfSpace = types.SimpleNamespace(
        from_point_and_normal=lambda p, n: (tuple(p), tuple(n))
    )
    pk.collision.Sphere = types.SimpleNamespace(
        from_center_and_radius=lambda c, r: (tuple(c), float(r[0]))
    )
    pk.collision.Capsule = types.SimpleNamespace(
        from_radius_height=lambda position, radius, height: (
            tuple(np.ravel(position).tolist()),
            float(radius[0]),
            float(height[0]),
        )
    )

    monkeypatch.setattr(integ, "pk", pk, raising=True)  # type: ignore[arg-type]

    class PksModule(types.SimpleNamespace):
        pass

    pks = PksModule()

    def solve_ik(
        robot: FakeRobot,
        target_link_name: str,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
    ) -> np.ndarray:
        return np.arange(7, dtype=np.float64)

    def solve_online_planning(
        robot: FakeRobot,
        robot_coll: object,
        world_coll: list[object],
        target_link_name: str,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        timesteps: int,
        dt: float,
        start_cfg: np.ndarray,
        prev_sols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        traj = np.tile(np.arange(7, dtype=np.float64)[None, :], (timesteps, 1))
        return traj, target_position, target_wxyz

    pks.solve_ik = solve_ik  # type: ignore[attr-defined]
    pks.solve_online_planning = solve_online_planning  # type: ignore[attr-defined]
    monkeypatch.setattr(integ, "pks", pks, raising=True)  # type: ignore[arg-type]

    # Monkeypatch URDF loader
    monkeypatch.setattr(integ, "load_robot_description", lambda name: object(), raising=True)

    # Initialize
    init_pyroki("panda_description", target_link_name="panda_hand")

    # Test IK
    T = np.eye(4, dtype=np.float64)
    q = tool_api.ik_solve(T)
    assert q.shape == (7,)
    assert np.allclose(q, np.arange(7, dtype=np.float32))

    # Test plan
    obstacles = [
        {"type": "halfspace", "point": [0, 0, 0], "normal": [0, 0, 1]},
        {
            "type": "goal",
            "position": [0.5, 0.0, 0.5],
            "wxyz": [0, 0, 1, 0],
            "timesteps": 5,
            "dt": 0.02,
        },
    ]
    out = tool_api.plan_motion(np.zeros(7), np.zeros(7), obstacles)
    assert "waypoints" in out and "dt" in out
    assert len(out["waypoints"]) == 5
    assert out["dt"] == 0.02


@pytest.mark.integration
def test_pyroki_real_franka() -> None:
    # Requires robotics extra (pyroki + robot_descriptions)
    init_pyroki("panda_description", target_link_name="panda_hand")

    T = np.eye(4)
    q = tool_api.ik_solve(T)
    assert q.shape[0] >= 6
