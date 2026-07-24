import numpy as np

from capx.integrations.franka.two_arm_peg_in_hole_privileged import (
    FrankaTwoArmPegInHolePrivilegedApi,
)


def test_two_arm_peg_in_hole_pose_aliases() -> None:
    poses = {
        name: np.array([index, 0, 0, 1, 0, 0, 0], dtype=float)
        for index, name in enumerate(
            ("peg", "hole", "hole_target", "arm0_eef", "arm1_eef")
        )
    }

    class FakeEnv:
        def get_observation(self) -> dict:
            return {"peg_in_hole_poses": poses}

    api = object.__new__(FrankaTwoArmPegInHolePrivilegedApi)
    api._env = FakeEnv()

    np.testing.assert_array_equal(api.get_object_pose("peg")[0], [0, 0, 0])
    np.testing.assert_array_equal(
        api.get_object_pose("hole insertion target")[0], [2, 0, 0]
    )
    np.testing.assert_array_equal(api.get_object_pose("arm0 end effector")[0], [3, 0, 0])
    np.testing.assert_array_equal(api.get_object_pose("robot1 eef")[0], [4, 0, 0])
