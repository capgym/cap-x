"""CaP-X evaluation entry point.

Usage::

    uv run --no-sync --active capx/envs/launch.py \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml

Execution flow::

    main()
      ├─ _run_web_ui()            (interactive browser mode)
      └─ _run_headless_trials()   (CLI batch mode)  [in capx.envs.runner]
           ├─ _run_trial_batch()  (sequential)
           └─ run_parallel_*()    (multi-worker)
                └─ _run_single_trial()  [in capx.envs.trial]
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from capx.utils.launch_utils import _load_config

os.environ.setdefault("MUJOCO_GL", "egl")


# ---------------------------------------------------------------------------
# CLI argument dataclass
# ---------------------------------------------------------------------------

@dataclass
class LaunchArgs:
    """Command-line arguments for CaP-X evaluation.

    Defines configuration options for model querying, execution, and output.
    """

    # YAML config path (required)
    config_path: str
    """Path to the YAML configuration file defining the environment and task."""

    # Model server configuration
    server_url: str = "http://127.0.0.1:8110/chat/completions"
    """URL of the vLLM server's chat completions endpoint."""

    model: str = "google/gemini-3.1-pro-preview"
    """Name of the model to query on the vLLM server."""

    temperature: float = 1.0
    """Sampling temperature for code generation (higher = more random)."""

    max_tokens: int = 2048 * 10
    """Maximum number of tokens to generate in the model response."""

    reasoning_effort: str = "medium"
    """Effort level for reasoning models (if applicable). Options: minimal, low, medium, high."""

    api_key: str | None = None
    """Optional API key for authentication with the model server."""

    # Execution configuration (can override YAML values)
    use_visual_feedback: bool | None = None
    """Whether to provide visual feedback (images) to the model during generation."""

    use_img_differencing: bool | None = None
    """Whether to provide image differencing to the model during generation."""

    use_video_differencing: bool | None = None
    """Use video-based VDM: pass a video of each turn's execution to the differencing model."""

    use_wrist_camera: bool | None = None
    """Also record and pass wrist camera video to the VDM alongside the main camera."""

    use_legacy_multi_turn_decision_prompt: bool | None = None
    """Whether to use the legacy multi-turn decision prompt."""

    visual_differencing_model: str | None = "google/gemini-3.1-pro-preview"
    """Model to use for image differencing."""

    visual_differencing_model_server_url: str | None = (
        "http://127.0.0.1:8110/chat/completions"
    )
    """Server URL of the image differencing model."""

    visual_differencing_model_api_key: str | None = None
    """API key for authentication with the image differencing model."""

    total_trials: int | None = None
    """Total number of trials to run. Overrides the value in the YAML config."""

    num_workers: int | None = None
    """Number of parallel worker processes to use. Overrides the value in the YAML config."""

    record_video: bool | None = None
    """Whether to record and save videos of the environment execution."""

    output_dir: str | None = None
    """Directory to save trial outputs (code, logs, videos)."""

    debug: bool | None = False
    """Enable debug logging (prints full model responses)."""

    use_oracle_code: bool | None = None
    """If True, uses pre-defined oracle code instead of querying the model."""

    use_parallel_ensemble: bool | None = None
    """Whether to use parallel ensemble for the coding agent."""

    use_multimodel: bool | None = None
    """Whether to use multimodel for parallel ensembling."""

    # Web UI configuration
    web_ui: bool | None = None
    """Launch the interactive web UI instead of running trials in headless CLI mode."""

    web_ui_port: int | None = None
    """Port for the web UI server (default 8200)."""


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

def _ensure_frontend_built() -> None:
    """Auto-build the web-ui frontend if ``dist/`` is missing or stale."""
    import shutil
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent.parent
    webui_dir = project_root / "web-ui"
    dist_dir = webui_dir / "dist"

    if not webui_dir.exists():
        print("[web-ui] web-ui/ directory not found — skipping frontend build")
        return

    needs_build = not dist_dir.exists()
    if not needs_build:
        dist_mtime = (dist_dir / "index.html").stat().st_mtime if (dist_dir / "index.html").exists() else 0
        for src_file in (webui_dir / "src").rglob("*"):
            if src_file.is_file() and src_file.stat().st_mtime > dist_mtime:
                needs_build = True
                break
        pkg_json = webui_dir / "package.json"
        if pkg_json.exists() and pkg_json.stat().st_mtime > dist_mtime:
            needs_build = True

    if not needs_build:
        return

    print("[web-ui] Building frontend...")
    nodeenv_dir = Path.home() / ".hyrl_nodeenv"
    npm_bin = nodeenv_dir / "bin" / "npm"
    node_bin = nodeenv_dir / "bin" / "node"

    if not node_bin.exists():
        nodeenv_bin = shutil.which("nodeenv")
        if nodeenv_bin is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-q", "nodeenv"])
            nodeenv_bin = shutil.which("nodeenv")
            if nodeenv_bin is None:
                raise RuntimeError("Could not install nodeenv. Install Node.js manually.")
        subprocess.check_call([nodeenv_bin, "--prebuilt", "--node=20.18.1", str(nodeenv_dir)])

    env = {**os.environ, "PATH": f"{nodeenv_dir / 'bin'}:{os.environ.get('PATH', '')}"}
    node_modules = webui_dir / "node_modules"
    pkg_json = webui_dir / "package.json"
    if not node_modules.exists() or (pkg_json.exists() and pkg_json.stat().st_mtime > node_modules.stat().st_mtime):
        subprocess.check_call([str(npm_bin), "install"], cwd=webui_dir, env=env)
    subprocess.check_call([str(npm_bin), "run", "build"], cwd=webui_dir, env=env)
    print("[web-ui] Frontend build complete")


def _run_web_ui(args: LaunchArgs, config: dict[str, Any]) -> None:
    """Start the interactive web UI server."""
    import uvicorn
    from capx.web.server import create_app

    _ensure_frontend_built()
    port = int(config.get("web_ui_port", 8200))
    app = create_app()
    app.state.default_config_path = args.config_path
    print(f"\n  CaP-X Interactive Web UI: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: LaunchArgs) -> None:
    """Load config and dispatch to web UI or headless trial execution."""
    from capx.envs.runner import _run_headless_trials, _start_api_servers, _stop_api_servers

    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    server_procs = _start_api_servers(api_servers)

    try:
        if config.get("web_ui", False):
            _run_web_ui(args, config)
        else:
            _run_headless_trials(args, env_factory, config, start_time)
    finally:
        try:
            _stop_api_servers(server_procs)
        except KeyboardInterrupt:
            # Force exit if user interrupts during cleanup
            import sys
            sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(LaunchArgs))
