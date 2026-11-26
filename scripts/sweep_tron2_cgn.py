"""Run one recorded CaP-X oracle trial per ranked CGN candidate on TRON2."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE_DONE = "[CGN sweep] oracle action sequence completed"
SELECTION_FIELDS = re.compile(r"\b(camera|zero_based_index|jaw_branch|score)=([^\s]+)")
MOTION_WARNING = re.compile(
    r"\[CGN-only\] WARNING:|\bjoint limit:|joint\[\d+\].*out of|CONTROLLER_JOINT_LIMIT|WAYPOINT_NOT_REACHED|"
    r"WAYPOINT_SETTLE_WARNING", re.I,
)


def _write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_rank(env, launch_args, config: dict, rank: int, output_dir: Path,
              trial_name: str, candidate_file: Path) -> dict:
    from imageio_ffmpeg import count_frames_and_secs

    from capx.envs.runner import TRIAL_TIMEOUT_SECONDS, _run_single_trial_with_timeout
    from capx.envs.tasks.base import Tee
    from capx.utils.video_utils import _write_video

    trial_dir = output_dir / trial_name
    trial_dir.mkdir()
    log_path = trial_dir / "run.log"
    config = {**config, "output_dir": str(trial_dir)}
    os.environ["CAPX_TRON2_CGN_RANK"] = str(rank)
    os.environ["CAPX_TRON2_CGN_CANDIDATES"] = str(candidate_file)
    env.enable_video_capture(False, clear=True)
    summary = None
    error = None
    video = None
    frames = None

    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(
        Tee(sys.stdout, log)
    ), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        print(f"[CGN sweep] starting {trial_name}, candidate rank {rank}")
        try:
            summary = _run_single_trial_with_timeout(
                env, 1, launch_args, config, None, TRIAL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            error = str(exc)
            traceback.print_exc()

        video = next(trial_dir.glob("trial_*/video_combined.mp4"), None)
        if video is None and env.get_video_frame_count():
            try:
                _write_video(env.get_video_frames(clear=True), str(trial_dir), suffix="partial")
                video = trial_dir / "video_partial.mp4"
            except Exception:
                traceback.print_exc()
        if video is not None:
            try:
                frames, _ = count_frames_and_secs(str(video))
                if not frames:
                    raise ValueError("MP4 contains no decodable frames")
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[CGN sweep] video verification failed: {exc}", file=sys.stderr)
                frames = None

    lines = log_path.read_text(encoding="utf-8").splitlines()
    selected = next((line for line in lines if "[CGN-only] selected " in line), "")
    candidate = dict(SELECTION_FIELDS.findall(selected))
    for field in ("zero_based_index", "jaw_branch"):
        if field in candidate:
            candidate[field] = int(candidate[field])
    if "score" in candidate:
        candidate["score"] = float(candidate["score"])
    warnings = [line for line in lines if MOTION_WARNING.search(line)]
    tcp_checks = [line for line in lines if line.startswith("[CGN-only]") and "measured TCP error" in line]
    finished = ORACLE_DONE in lines
    earlier_lines = lines[:lines.index(ORACLE_DONE)] if finished else []
    result = {
        "trial": trial_name,
        "candidate_rank": rank,
        "candidate_file": os.path.relpath(candidate_file, output_dir),
        "candidate": candidate or None,
        "candidate_selection_log": selected or None,
        "sandbox_rc": summary.sandbox_rc if summary else None,
        "execution_completed": finished and summary is not None and summary.success,
        "motion_warnings": warnings,
        "all_motion_targets_reached": len(tcp_checks) == 4 and all(line.endswith("reached=True") for line in tcp_checks),
        "continued_after_motion_warning": any(MOTION_WARNING.search(line) for line in earlier_lines),
        "reward": summary.reward if summary else None,
        "task_completed": summary.task_completed if summary else None,
        "video": str(video.relative_to(output_dir)) if video else None,
        "video_frames": frames,
        "log": str(log_path.relative_to(output_dir)),
        "summary": str(Path(summary.code_path).with_name("summary.txt").relative_to(output_dir))
        if summary and summary.code_path else None,
        "error": error,
    }
    _write_json(trial_dir / "result.json", result)
    print(f"[CGN sweep] rank {rank}: {json.dumps(result)}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for test artifacts.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ranks", nargs="+", type=int, help="Zero-based ranks from the same proposal batch (default: 0–4).")
    selection.add_argument("--fresh-trials", type=int, help="Sample a new CGN batch and select rank 0 for each trial.")
    parser.add_argument("--candidate-file", type=Path, help="Resume using a saved CGN proposal batch.")
    parser.add_argument("--jaw-branch", type=int, choices=(0, 1), help="Replay one specific jaw assignment.")
    parser.add_argument("--config-path", type=Path, default=ROOT / "env_configs/tron2/tron2_cube_lift_visual_cgn_only.yaml")
    parser.add_argument("--route2-startup-dir", type=Path,
                        help="Coordinate motor and controller startup with the isolated Route 2 launcher.")
    cli = parser.parse_args()
    ranks = cli.ranks or list(range(5))
    if min(ranks) < 0 or len(ranks) != len(set(ranks)):
        parser.error("--ranks must be distinct nonnegative integers")
    if cli.fresh_trials is not None and (cli.fresh_trials < 1 or cli.candidate_file is not None):
        parser.error("--fresh-trials requires a positive count and samples its own candidates")
    if cli.candidate_file is not None and not cli.candidate_file.is_file():
        parser.error("--candidate-file must point to a saved CGN proposal batch")
    os.environ.pop("CAPX_TRON2_CGN_JAW_BRANCH", None)
    if cli.jaw_branch is not None:
        os.environ["CAPX_TRON2_CGN_JAW_BRANCH"] = str(cli.jaw_branch)

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.launch import LaunchArgs
    from capx.envs.runner import _start_api_servers, _stop_api_servers
    from capx.utils.launch_utils import _load_config

    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True)
    candidate_file = cli.candidate_file.resolve() if cli.candidate_file else output_dir / "candidates.npz"
    if cli.fresh_trials is None:
        cases = [(f"rank_{rank:03d}", rank, candidate_file) for rank in ranks]
    else:
        cases = [(f"fresh_{i:03d}", 0, output_dir / f"fresh_{i:03d}_candidates.npz")
                 for i in range(1, cli.fresh_trials + 1)]
    launch_args = LaunchArgs(
        config_path=str(cli.config_path), model="oracle", use_oracle_code=True,
        total_trials=1, num_workers=1, record_video=True,
    )
    env_factory, config, api_servers = _load_config(launch_args)
    if env_factory["cfg"].get("apis") != ["Tron2ControlCgnOnlyApi"]:
        parser.error("--config-path must select only Tron2ControlCgnOnlyApi")
    if cli.route2_startup_dir and env_factory["cfg"].get("low_level") != "tron2_cube_lift_low_level":
        parser.error("--route2-startup-dir requires the motor low-level environment")

    server_procs = _start_api_servers(api_servers)
    env = None
    results = []
    try:
        env = instantiate(env_factory)
        if cli.route2_startup_dir:
            env.low_level_env.start_motor_loop()
            (cli.route2_startup_dir / "motor.ready").touch()
            controller_ready = cli.route2_startup_dir / "controller.ready"
            deadline = time.monotonic() + 75.0
            while not controller_ready.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("The isolated TRON2 controller did not signal ready")
                time.sleep(0.05)
        env.oracle_code = env.oracle_code.rstrip() + f"\nprint({ORACLE_DONE!r})\n"
        for name, rank, batch in cases:
            results.append(_run_rank(env, launch_args, config, rank, output_dir, name, batch))
            _write_json(output_dir / "results.json", {
                "selection": "fresh CGN per trial" if cli.fresh_trials else "ranked within one CGN batch",
                "trials": results,
            })
    finally:
        try:
            if env is not None:
                env.low_level_env.close()
        finally:
            _stop_api_servers(server_procs)
    return int(any(not (result["execution_completed"] and result["video_frames"]) for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
