import numpy as np

from capx.integrations.franka.tool_hang_privileged import (
    FrankaControlToolHangPrivilegedApi,
)


def test_tool_hang_pose_aliases_and_grasps() -> None:
    poses = {
        name: np.array([index, 0, 0, 1, 0, 0, 0], dtype=float)
        for index, name in enumerate(
            (
                "stand",
                "stand_mount",
                "frame",
                "frame_grip",
                "frame_tip",
                "frame_hang",
                "tool",
                "tool_grip",
                "tool_hole",
            )
        )
    }

    class FakeEnv:
        def get_observation(self) -> dict:
            return {"tool_hang_poses": poses}

    api = object.__new__(FrankaControlToolHangPrivilegedApi)
    api._env = FakeEnv()

    np.testing.assert_array_equal(api.get_object_pose("stand mount")[0], [1, 0, 0])
    np.testing.assert_array_equal(api.get_object_pose("frame tip")[0], [4, 0, 0])
    np.testing.assert_array_equal(api.get_object_pose("wrench hole")[0], [8, 0, 0])
    np.testing.assert_array_equal(api.sample_grasp_pose("frame")[0], [3, 0, 0])
    np.testing.assert_array_equal(api.sample_grasp_pose("wrench")[0], [7, 0, 0])
