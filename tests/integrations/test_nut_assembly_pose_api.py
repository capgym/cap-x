import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.franka.nut_assembly_privileged import (
    FrankaControlNutAssemblyPrivilegedApi,
)


def _wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return xyzw[[3, 0, 1, 2]]


def test_relative_pose_and_compose_pose_round_trip() -> None:
    api = object.__new__(FrankaControlNutAssemblyPrivilegedApi)
    parent_position = np.array([0.4, -0.2, 0.3])
    parent_quaternion = _wxyz(Rotation.from_euler("xyz", [0.2, -0.1, 0.7]))
    child_position = np.array([0.55, 0.05, 0.42])
    child_quaternion = _wxyz(Rotation.from_euler("xyz", [-0.3, 0.4, 1.1]))

    relative_position, relative_quaternion = api.relative_pose(
        parent_position,
        parent_quaternion,
        child_position,
        child_quaternion,
    )
    recovered_position, recovered_quaternion = api.compose_pose(
        parent_position,
        parent_quaternion,
        relative_position,
        relative_quaternion,
    )

    np.testing.assert_allclose(recovered_position, child_position, atol=1e-8)
    np.testing.assert_allclose(
        Rotation.from_quat(recovered_quaternion[[1, 2, 3, 0]]).as_matrix(),
        Rotation.from_quat(child_quaternion[[1, 2, 3, 0]]).as_matrix(),
        atol=1e-8,
    )


def test_object_queries_cover_round_and_random_single_variants() -> None:
    class FakeEnv:
        def active_nut_types(self) -> tuple[str, ...]:
            return ("round",)

        def get_observation(self) -> dict:
            return {
                "nut_poses": {
                    "round_nut": np.array([1, 2, 3, 1, 0, 0, 0], dtype=float),
                    "round_nut_handle": np.array([4, 5, 6, 1, 0, 0, 0], dtype=float),
                    "round_peg": np.array([7, 8, 9, 1, 0, 0, 0], dtype=float),
                }
            }

    api = object.__new__(FrankaControlNutAssemblyPrivilegedApi)
    api._env = FakeEnv()

    assert api.get_active_nut_types() == ("round",)
    np.testing.assert_array_equal(api.get_object_pose("active nut")[0], [1, 2, 3])
    np.testing.assert_array_equal(api.sample_grasp_pose("active nut handle")[0], [4, 5, 6])
    np.testing.assert_array_equal(api.get_object_pose("matching peg")[0], [7, 8, 9])
