from __future__ import annotations

import argparse
import json
from pathlib import Path

from capx.security import validate_generated_program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-trials", type=int, default=100)
    parser.add_argument("--minimum-success-rate", type=float, default=0.8)
    parser.add_argument("--require-video", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict:
    summary_path = args.result_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["total_trials"] != args.expected_trials:
        raise ValueError(
            f"expected {args.expected_trials} trials, found {summary['total_trials']}"
        )
    if summary["use_oracle_code"]:
        raise ValueError("oracle-code evaluation is not an accepted policy result")
    if summary["git_dirty"]:
        raise ValueError("formal evaluation must use a clean code commit")
    if summary["task_completion_rate"] <= args.minimum_success_rate:
        raise ValueError(
            f"task completion {summary['task_completion_rate']:.3f} must be greater than "
            f"{args.minimum_success_rate:.3f}"
        )

    code_paths = sorted(args.result_dir.glob("trial_*/code.py"))
    if len(code_paths) != args.expected_trials:
        raise ValueError(f"expected {args.expected_trials} generated programs, found {len(code_paths)}")
    for code_path in code_paths:
        result = validate_generated_program(code_path.read_text())
        if not result.allowed:
            raise ValueError(f"{code_path} failed policy guard: {result.reason}")

    videos = sorted(args.result_dir.glob("trial_*/*.mp4"))
    if args.require_video and not videos:
        raise ValueError("no MP4 demo found")
    return {
        "result_dir": str(args.result_dir.resolve()),
        "total_trials": summary["total_trials"],
        "task_completed": summary["task_completed"],
        "task_completion_rate": summary["task_completion_rate"],
        "generated_programs_checked": len(code_paths),
        "videos": [str(path.resolve()) for path in videos],
        "git_commit": summary["git_commit"],
    }


def main() -> None:
    print(json.dumps(validate(parse_args()), indent=2))


if __name__ == "__main__":
    main()
