from __future__ import annotations

import argparse
import json
from pathlib import Path

from capx.utils.nut_assembly_validation import validate_nut_assembly_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-trials", type=int, default=100)
    parser.add_argument("--minimum-success-rate", type=float, default=0.8)
    parser.add_argument("--require-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(validate_nut_assembly_results(parse_args()), indent=2))


if __name__ == "__main__":
    main()
