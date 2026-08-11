"""Persistent Cap-X grasp/hold/place runner for the manual+SLAM G1 demo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader


DEFAULT_CONFIG_PATH = "env_configs/g1/g1_grasp_bottle.yaml"
DEFAULT_OBJECT_NAME = "plastic water bottle"


class BottleCarryDemoRunner:
    """Keep one Cap-X environment alive across grasp, navigation hold, and place."""

    REQUIRED_FUNCTIONS = (
        "sample_grasp_center_pose",
        "grasp_at_pinch_center",
        "move_pinch_center_to_pose",
        "move_to_pregrasp_side_pose",
        "open_gripper",
    )

    def __init__(
        self,
        functions: Mapping[str, Callable[..., Any]],
        *,
        object_name: str = DEFAULT_OBJECT_NAME,
        place_hover_dz: float = 0.10,
        place_retreat_x: float = 0.20,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        missing = [name for name in self.REQUIRED_FUNCTIONS if not callable(functions.get(name))]
        if missing:
            raise ValueError(f"Cap-X G1 API is missing required functions: {missing}")
        if not np.isfinite(place_hover_dz) or place_hover_dz <= 0.0:
            raise ValueError("place_hover_dz must be positive and finite")
        if not np.isfinite(place_retreat_x) or place_retreat_x <= 0.0:
            raise ValueError("place_retreat_x must be positive and finite")
        self.functions = functions
        self.object_name = str(object_name)
        self.place_hover_dz = float(place_hover_dz)
        self.place_retreat_x = float(place_retreat_x)
        self.input_fn = input_fn
        self.grasp_position: np.ndarray | None = None
        self.grasp_quaternion_wxyz: np.ndarray | None = None
        self.phase = "initialized"

    def grasp_at_table_a(self) -> dict[str, Any]:
        """Sample once, execute the verified bottle grasp, and enter side hold."""
        position, quaternion = self.functions["sample_grasp_center_pose"](self.object_name)
        self.grasp_position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.grasp_quaternion_wxyz = (
            np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
        )
        self.functions["grasp_at_pinch_center"](
            self.grasp_position,
            self.grasp_quaternion_wxyz,
            trigger=1.0,
            squeeze=1.0,
            lift_dz=0.12,
        )
        self.functions["move_to_pregrasp_side_pose"]()
        self.phase = "holding_for_navigation"
        return self.status()

    def place_at_table_b(self) -> dict[str, Any]:
        """Reuse the table-A torso-frame target for hover, place, retreat, and side lift."""
        if self.grasp_position is None or self.grasp_quaternion_wxyz is None:
            raise RuntimeError("grasp_at_table_a() must complete before placement")
        position = self.grasp_position.copy()
        quaternion = self.grasp_quaternion_wxyz.copy()
        hover = position + np.asarray([0.0, 0.0, self.place_hover_dz])
        retreat = position + np.asarray(
            [-self.place_retreat_x, 0.0, self.place_hover_dz]
        )
        self.functions["move_pinch_center_to_pose"](hover, quaternion)
        self.functions["move_pinch_center_to_pose"](position, quaternion)
        self.functions["open_gripper"]()
        self.functions["move_pinch_center_to_pose"](retreat, quaternion)
        self.functions["move_to_pregrasp_side_pose"]()
        self.phase = "placed"
        return self.status()

    def status(self) -> dict[str, Any]:
        """Return phase and the preserved table-A torso-frame grasp target."""
        return {
            "success": self.phase in {"holding_for_navigation", "placed"},
            "phase": self.phase,
            "object_name": self.object_name,
            "grasp_position": (
                None if self.grasp_position is None else self.grasp_position.tolist()
            ),
            "grasp_quaternion_wxyz": (
                None
                if self.grasp_quaternion_wxyz is None
                else self.grasp_quaternion_wxyz.tolist()
            ),
            "environment_persistent": True,
            "manual_capx_ipc_added": False,
        }

    def run(self) -> dict[str, Any]:
        """Run the two operator-confirmed manipulation stages in one process."""
        self.input_fn(
            "Confirm manual navigate_to_pose(table_A) ARRIVED and open-loop "
            "micro-adjustment is complete; press Enter to grasp: "
        )
        print(json.dumps(self.grasp_at_table_a(), indent=2))
        self.input_fn(
            "Bottle hold is active. In the manual terminal navigate to table B, "
            "wait for ARRIVED, and micro-adjust; press Enter to place: "
        )
        result = self.place_at_table_b()
        print(json.dumps(result, indent=2))
        return result


def _load_persistent_environment(config_path: str, *, dry_run: bool) -> Any:
    config = DictLoader.load(str(config_path))
    if "env" not in config:
        raise ValueError(f"Cap-X config has no env factory: {config_path}")
    env_factory = config["env"]
    if dry_run:
        env_factory["cfg"]["low_level"]["dry_run"] = True
    environment = instantiate(env_factory)
    environment.reset()
    return environment


def _find_g1_functions(environment: Any) -> Mapping[str, Callable[..., Any]]:
    for api in environment._apis.values():
        functions = api.functions()
        if all(name in functions for name in BottleCarryDemoRunner.REQUIRED_FUNCTIONS):
            return functions
    raise RuntimeError("configured Cap-X environment has no complete G1 bottle API")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep one Cap-X G1 environment alive across bottle grasp, hold, and place."
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--place-hover-dz", type=float, default=0.10)
    parser.add_argument("--place-retreat-x", type=float, default=0.20)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Force G1 low-level dry-run; vision/IK services and observation "
            "input are still required."
        ),
    )
    parser.add_argument(
        "--enable-real",
        action="store_true",
        help=(
            "Explicitly acknowledge live arm_action and Dex3 output through "
            "the configured Gateway route."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run and args.enable_real:
        raise SystemExit("--dry-run and --enable-real are mutually exclusive")
    if not args.dry_run and not args.enable_real:
        raise SystemExit("real execution requires --enable-real")
    if not args.dry_run and os.environ.get("CAPX_G1_DRY_RUN", "").lower() != "false":
        raise SystemExit("real execution requires CAPX_G1_DRY_RUN=false")
    if not Path(args.config_path).is_file():
        raise SystemExit(f"missing config: {args.config_path}")

    environment = None
    try:
        environment = _load_persistent_environment(args.config_path, dry_run=args.dry_run)
        runner = BottleCarryDemoRunner(
            _find_g1_functions(environment),
            object_name=args.object_name,
            place_hover_dz=args.place_hover_dz,
            place_retreat_x=args.place_retreat_x,
        )
        runner.run()
        return 0
    except KeyboardInterrupt:
        print("\nCap-X demo interrupted; closing the persistent environment.")
        return 130
    finally:
        if environment is not None:
            close = getattr(environment.low_level_env, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
