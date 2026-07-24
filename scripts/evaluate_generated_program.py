from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from capx.security import validate_generated_program
from verl_agent_reward.capx_franka_reward import compute_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one guarded generated robot program across environment seeds."
    )
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    code = args.code_path.read_text()
    guard = validate_generated_program(code)
    if not guard.allowed:
        raise ValueError(f"generated program failed policy guard: {guard.reason}")
    results = [
        compute_score(args.data_source, code, {}, {"seed": seed})
        for seed in range(args.seeds)
    ]
    receipt = {
        "data_source": args.data_source,
        "source": str(args.code_path.resolve()),
        "program_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "results": results,
        "successes": sum(bool(result["won"]) for result in results),
        "total": len(results),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
