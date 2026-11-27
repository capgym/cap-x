"""Run the paired, no-CGN TRON2 evaluation through the stock launcher."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import runtime_overrides

runtime_overrides.install()
from scripts import run_tron2_model


def evaluate_with_dashboard(cli, model_url, vdm_url):
    from capx.envs.simulators.tron2_cube_lift import Tron2CubeLiftLowLevel
    from dashboard_instrumentation import install_dashboard_tracing

    args = SimpleNamespace(
        server_url=model_url,
        visual_differencing_model_server_url=vdm_url,
        api_key=os.getenv("TRON2_MODEL_API_KEY"),
        visual_differencing_model_api_key=os.getenv("TRON2_VDM_API_KEY"),
    )
    stream = open(os.environ["TRON2_DASHBOARD_EVENTS"], "a", encoding="utf-8", buffering=1)
    # The JPEG worker is a daemon and can still finish a preview during shutdown.
    install_dashboard_tracing(args, stream, env_class=Tron2CubeLiftLowLevel, include_graspnet=False, stream_qwen=True)
    return stock_evaluate(cli, model_url, vdm_url)


stock_evaluate = run_tron2_model._evaluate


def main() -> int:
    if "--serve-relay" in sys.argv[1:]:
        return run_tron2_model.main()

    import yaml
    import rgbd_no_cgn_api  # Registers the same RGB-D API used by v1.
    from rgbd_tasks import Tron2CubeLiftRgbdCodeEnv
    from trial_telemetry import record_trial

    if any(arg.split("=")[0] == "--candidate-file" for arg in sys.argv[1:]):
        raise ValueError("The paired RGB-D evaluation accepts no grasp candidates")
    variant = os.environ["CAPX_PROMPT_VARIANT"]
    if variant not in {"v1", "v2"}:
        raise ValueError(f"Unknown prompt variant {variant!r}")
    root = Path(__file__).resolve().parent
    if variant == "v2":
        Tron2CubeLiftRgbdCodeEnv.prompt += (root / "v2_spec.txt").read_text(encoding="utf-8")
    profiles = {}
    for tier in ("M1", "M2", "M3"):
        path = root / "profiles" / f"{tier}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        servers = [item["_target_"] for item in config["api_servers"]]
        task = config["env"]["cfg"]
        if servers != ["capx.serving.launch_sam3_server.main"]:
            raise ValueError(f"Expected only SAM3 in {path}")
        if task.get("privileged") or task["apis"] != ["Tron2RgbdNoCgnHighApi"]:
            raise ValueError(f"Unexpected policy API in {path}")
        profiles[tier] = os.path.relpath(path, Path.cwd() / "env_configs/tron2")
    run_tron2_model._PROFILES = profiles
    run_tron2_model._run_recorded_trial = record_trial(run_tron2_model._run_recorded_trial)
    if "TRON2_DASHBOARD_EVENTS" in os.environ:
        run_tron2_model._evaluate = evaluate_with_dashboard
    return run_tron2_model.main()


if __name__ == "__main__":
    raise SystemExit(main())
