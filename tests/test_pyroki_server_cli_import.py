from __future__ import annotations

import subprocess
import sys


def test_pyroki_server_cli_imports_without_capx_registration_cycle() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "capx.serving.launch_pyroki_server", "--help"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--robot" in result.stdout
