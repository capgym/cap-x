from __future__ import annotations

import unittest

from capx.serving import launch_servers


class LaunchServersSam3CheckpointTest(unittest.TestCase):
    def test_default_profile_passes_local_sam3_checkpoint(self) -> None:
        args = launch_servers.LaunchServersArgs(profile="default")

        servers = launch_servers._resolve_servers(args)
        sam3_server = next(s for s in servers if s["server"] == "sam3")
        sam3_server["gpu_index"] = 0

        cmd = launch_servers._build_cmd(sam3_server, workers=1)

        self.assertIn("--checkpoint-path", cmd)
        checkpoint_arg_index = cmd.index("--checkpoint-path") + 1
        self.assertEqual(
            cmd[checkpoint_arg_index],
            "capx/model_weights/sam3/sam3.pt",
        )


if __name__ == "__main__":
    unittest.main()
