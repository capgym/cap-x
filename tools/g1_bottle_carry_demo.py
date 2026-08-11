"""Persistent Cap-X grasp/hold/place runner for the manual+SLAM G1 demo."""

from __future__ import annotations

import argparse
from enum import Enum
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader


DEFAULT_CONFIG_PATH = "env_configs/g1/g1_grasp_bottle.yaml"
DEFAULT_OBJECT_NAME = "plastic water bottle"


class BottleCarryPhase(str, Enum):
    """Fail-closed phases for the persistent two-table manipulation demo."""

    INITIALIZED = "initialized"
    HOLDING_FOR_NAVIGATION = "holding_for_navigation"
    PLACING = "placing"
    PLACED = "placed"
    FAILED = "failed"


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
        hold_status_fn: Callable[[], Mapping[str, Any]],
        object_name: str = DEFAULT_OBJECT_NAME,
        place_hover_dz: float = 0.10,
        place_retreat_x: float = 0.20,
        hold_check_period_s: float = 0.10,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        missing = [
            name
            for name in self.REQUIRED_FUNCTIONS
            if not callable(functions.get(name))
        ]
        if missing:
            raise ValueError(f"Cap-X G1 API is missing required functions: {missing}")
        if not callable(hold_status_fn):
            raise ValueError("hold_status_fn must be callable")
        if not np.isfinite(place_hover_dz) or place_hover_dz <= 0.0:
            raise ValueError("place_hover_dz must be positive and finite")
        if not np.isfinite(place_retreat_x) or place_retreat_x <= 0.0:
            raise ValueError("place_retreat_x must be positive and finite")
        if not np.isfinite(hold_check_period_s) or hold_check_period_s <= 0.0:
            raise ValueError("hold_check_period_s must be positive and finite")
        self.functions = functions
        self.hold_status_fn = hold_status_fn
        self.object_name = str(object_name)
        self.place_hover_dz = float(place_hover_dz)
        self.place_retreat_x = float(place_retreat_x)
        self.hold_check_period_s = float(hold_check_period_s)
        self.input_fn = input_fn
        self.grasp_position: np.ndarray | None = None
        self.grasp_quaternion_wxyz: np.ndarray | None = None
        self.phase = BottleCarryPhase.INITIALIZED
        self.last_error: str | None = None
        self._last_hold_status: dict[str, Any] | None = None
        self._state_lock = threading.Lock()
        self._hold_monitor_stop = threading.Event()
        self._hold_monitor_thread: threading.Thread | None = None

    def grasp_at_table_a(self) -> dict[str, Any]:
        """Complete grasp and side hold before committing a reusable pose."""
        self._require_phase(BottleCarryPhase.INITIALIZED)
        try:
            position, quaternion = self.functions["sample_grasp_center_pose"](
                self.object_name
            )
            grasp_position = np.asarray(position, dtype=np.float64).reshape(3).copy()
            grasp_quaternion = (
                np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
            )
            if not np.all(np.isfinite(grasp_position)) or not np.all(
                np.isfinite(grasp_quaternion)
            ):
                raise ValueError("sampled grasp pose must be finite")
            self.functions["grasp_at_pinch_center"](
                grasp_position,
                grasp_quaternion,
                trigger=1.0,
                squeeze=1.0,
                lift_dz=0.12,
            )
            self.functions["move_to_pregrasp_side_pose"]()
            self._assert_hold_healthy()
            with self._state_lock:
                self.grasp_position = grasp_position
                self.grasp_quaternion_wxyz = grasp_quaternion
                self.phase = BottleCarryPhase.HOLDING_FOR_NAVIGATION
                self.last_error = None
            return self.status()
        except Exception as error:
            self._record_failure(f"table_a_grasp_failed:{type(error).__name__}:{error}")
            raise

    def place_at_table_b(self) -> dict[str, Any]:
        """Reuse the verified table-A torso-frame target for table-B placement."""
        self._require_phase(BottleCarryPhase.HOLDING_FOR_NAVIGATION)
        self._assert_hold_healthy()
        self._stop_hold_monitor()
        self._require_phase(BottleCarryPhase.HOLDING_FOR_NAVIGATION)
        with self._state_lock:
            assert self.grasp_position is not None
            assert self.grasp_quaternion_wxyz is not None
            position = self.grasp_position.copy()
            quaternion = self.grasp_quaternion_wxyz.copy()
            self.phase = BottleCarryPhase.PLACING
        hover = position + np.asarray([0.0, 0.0, self.place_hover_dz])
        retreat = position + np.asarray(
            [-self.place_retreat_x, 0.0, self.place_hover_dz]
        )
        try:
            self.functions["move_pinch_center_to_pose"](hover, quaternion)
            self.functions["move_pinch_center_to_pose"](position, quaternion)
            self.functions["open_gripper"]()
            self.functions["move_pinch_center_to_pose"](retreat, quaternion)
            self.functions["move_to_pregrasp_side_pose"]()
            with self._state_lock:
                self.phase = BottleCarryPhase.PLACED
                self.last_error = None
            return self.status()
        except Exception as error:
            self._record_failure(f"table_b_place_failed:{type(error).__name__}:{error}")
            raise

    def status(self) -> dict[str, Any]:
        """Return phase, hold health, and the committed table-A grasp target."""
        hold_status = self._read_hold_status()
        with self._state_lock:
            phase = self.phase
            last_error = self.last_error
            grasp_position = (
                None if self.grasp_position is None else self.grasp_position.tolist()
            )
            grasp_quaternion = (
                None
                if self.grasp_quaternion_wxyz is None
                else self.grasp_quaternion_wxyz.tolist()
            )
        hold_healthy = bool(hold_status.get("healthy", False))
        return {
            "success": (
                phase
                in {
                    BottleCarryPhase.HOLDING_FOR_NAVIGATION,
                    BottleCarryPhase.PLACED,
                }
                and hold_healthy
            ),
            "phase": phase.value,
            "object_name": self.object_name,
            "grasp_position": grasp_position,
            "grasp_quaternion_wxyz": grasp_quaternion,
            "arm_hold_monitored": True,
            "arm_hold_healthy": hold_healthy,
            "arm_hold_status": hold_status,
            "last_error": last_error,
            "environment_persistent": True,
            "manual_capx_ipc_added": False,
        }

    def run(self) -> dict[str, Any]:
        """Run the two operator-confirmed manipulation stages in one process."""
        try:
            self.input_fn(
                "Confirm manual navigate_to_pose(table_A) ARRIVED and open-loop "
                "micro-adjustment is complete; press Enter to grasp: "
            )
            print(json.dumps(self.grasp_at_table_a(), indent=2))
            self._start_hold_monitor()
            self.input_fn(
                "Bottle hold is active. In the manual terminal navigate to table B, "
                "wait for ARRIVED, and micro-adjust; press Enter to place: "
            )
            self._assert_hold_healthy()
            self._stop_hold_monitor()
            result = self.place_at_table_b()
            print(json.dumps(result, indent=2))
            return result
        finally:
            self._stop_hold_monitor()

    def _require_phase(self, required: BottleCarryPhase) -> None:
        with self._state_lock:
            phase = self.phase
            pose_ready = (
                self.grasp_position is not None
                and self.grasp_quaternion_wxyz is not None
            )
        if (
            required == BottleCarryPhase.HOLDING_FOR_NAVIGATION
            and phase == BottleCarryPhase.INITIALIZED
            and not pose_ready
        ):
            raise RuntimeError("grasp_at_table_a() must complete before placement")
        if phase != required:
            raise RuntimeError(
                f"{required.value} phase required; current phase is {phase.value}"
            )
        if required == BottleCarryPhase.HOLDING_FOR_NAVIGATION and not pose_ready:
            raise RuntimeError("grasp_at_table_a() must complete before placement")

    def _read_hold_status(self) -> dict[str, Any]:
        try:
            raw_status = self.hold_status_fn()
            if not isinstance(raw_status, Mapping):
                raise TypeError("hold status must be a mapping")
            status = dict(raw_status)
            status["healthy"] = bool(status.get("healthy", False))
        except Exception as error:
            status = {
                "success": False,
                "healthy": False,
                "last_error": f"{type(error).__name__}:{error}",
            }
        with self._state_lock:
            self._last_hold_status = status
        return status

    def _assert_hold_healthy(self) -> dict[str, Any]:
        status = self._read_hold_status()
        if bool(status.get("healthy")):
            return status
        detail = status.get("last_error") or "persistent arm hold is not healthy"
        message = f"arm_hold_unhealthy:{detail}"
        self._record_failure(message)
        raise RuntimeError(message)

    def _record_failure(self, message: str) -> None:
        with self._state_lock:
            self.phase = BottleCarryPhase.FAILED
            self.last_error = str(message)

    def _start_hold_monitor(self) -> None:
        self._require_phase(BottleCarryPhase.HOLDING_FOR_NAVIGATION)
        self._assert_hold_healthy()
        with self._state_lock:
            thread = self._hold_monitor_thread
            if thread is not None and thread.is_alive():
                return
            self._hold_monitor_stop.clear()
            thread = threading.Thread(
                target=self._hold_monitor_loop,
                daemon=True,
                name="capx-bottle-arm-hold-monitor",
            )
            self._hold_monitor_thread = thread
            thread.start()

    def _hold_monitor_loop(self) -> None:
        while not self._hold_monitor_stop.wait(self.hold_check_period_s):
            try:
                self._assert_hold_healthy()
            except RuntimeError as error:
                print(f"[capx-bottle-demo] {error}")
                return

    def _stop_hold_monitor(self) -> None:
        self._hold_monitor_stop.set()
        with self._state_lock:
            thread = self._hold_monitor_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(1.0, 2.0 * self.hold_check_period_s))


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


def _find_arm_hold_status(
    environment: Any,
) -> Callable[[], Mapping[str, Any]]:
    hold_status = getattr(environment.low_level_env, "arm_hold_status", None)
    if not callable(hold_status):
        raise RuntimeError(
            "configured Cap-X G1 low-level environment has no arm_hold_status()"
        )
    return hold_status


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
            hold_status_fn=_find_arm_hold_status(environment),
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
