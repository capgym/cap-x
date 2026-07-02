from __future__ import annotations

from pathlib import Path


def test_init_pyroki_server_accepts_local_urdf_path() -> None:
    import capx.serving.launch_pyroki_server as server

    server._ROBOT = None
    server._ROBOT_COLL = None
    server._TARGET_LINK = None

    urdf_path = Path("capx/envs/assets/g1_description/g1_dual_arm.urdf")

    server.init_pyroki_server(
        robot_urdf_name=str(urdf_path),
        target_link_name="left_rubber_hand",
    )

    assert server._ROBOT is not None
    assert server._ROBOT_COLL is not None
    assert server._TARGET_LINK == "left_rubber_hand"
