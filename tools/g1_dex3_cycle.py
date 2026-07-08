from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _load_g1_sdk_module():
    module_path = Path(__file__).resolve().parents[1] / "capx" / "integrations" / "g1" / "sdk.py"
    spec = importlib.util.spec_from_file_location("_capx_g1_sdk", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load G1 SDK module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_G1_SDK = _load_g1_sdk_module()
DEFAULT_G1_DEX3_RIGHT_CMD_TOPIC = _G1_SDK.DEFAULT_G1_DEX3_RIGHT_CMD_TOPIC
DEFAULT_G1_DEX3_RIGHT_STATE_TOPIC = _G1_SDK.DEFAULT_G1_DEX3_RIGHT_STATE_TOPIC
DEFAULT_G1_NETWORK_INTERFACE = _G1_SDK.DEFAULT_G1_NETWORK_INTERFACE
G1Dex3HandBridge = _G1_SDK.G1Dex3HandBridge
dex3_grasp_joints = _G1_SDK.dex3_grasp_joints
dex3_semantic_to_dds_joints = _G1_SDK.dex3_semantic_to_dds_joints


@dataclass(frozen=True)
class Dex3Action:
    name: str
    joints: np.ndarray


def build_action_sequence() -> list[Dex3Action]:
    open_joints = dex3_grasp_joints(trigger=0.0, squeeze=0.0)
    index_pinch_joints = dex3_grasp_joints(trigger=1.0, squeeze=0.0)
    three_finger_joints = dex3_grasp_joints(trigger=1.0, squeeze=1.0)
    return [
        Dex3Action("open_gripper", open_joints),
        Dex3Action("close_index_pinch", index_pinch_joints),
        Dex3Action("open_gripper", open_joints),
        Dex3Action("close_gripper", three_finger_joints),
        Dex3Action("open_gripper", open_joints),
    ]


def _format_joints(joints: np.ndarray) -> str:
    return np.array2string(np.asarray(joints, dtype=np.float64), precision=4, suppress_small=True)


def _publish_action(bridge: Any, action: Dex3Action, *, dry_run: bool) -> None:
    semantic = np.asarray(action.joints, dtype=np.float64)
    dds = dex3_semantic_to_dds_joints(semantic)
    print(
        f"[g1-dex3-cycle] action={action.name} dry_run={dry_run} "
        f"semantic={_format_joints(semantic)} dds={_format_joints(dds)}"
    )
    ok = bridge.publish_hand_joints(semantic)
    if ok is False:
        raise RuntimeError(f"Dex3 publish failed while sending {action.name}.")


def run(
    args: argparse.Namespace,
    *,
    bridge_cls: Callable[..., Any] = G1Dex3HandBridge,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1; choose an explicit finite count for real-robot testing.")
    if args.hold_s < 0.0:
        raise ValueError("--hold-s must be >= 0.")

    dry_run = not bool(args.execute)
    bridge = bridge_cls(
        network_interface=args.interface,
        domain_id=args.domain_id,
        dry_run=dry_run,
        publisher_topic=args.cmd_topic,
        state_topic=args.state_topic,
        write_timeout=args.write_timeout,
    )
    sequence = build_action_sequence()
    open_action = sequence[0]

    try:
        if not dry_run:
            print(
                "[g1-dex3-cycle] EXECUTE enabled; publishing to "
                f"{args.cmd_topic} via interface {args.interface}."
            )
            bridge.connect()
            if args.wait_for_state and hasattr(bridge, "wait_for_hand_state"):
                if not bridge.wait_for_hand_state(timeout_s=3.0):
                    raise RuntimeError(
                        "No Dex3 hand state received on "
                        f"{args.state_topic}; use --no-wait-for-state only if you intentionally want command-only testing."
                    )
        else:
            print("[g1-dex3-cycle] dry-run only. Add --execute to publish to the real Dex3 hand.")

        for cycle_idx in range(args.cycles):
            print(f"[g1-dex3-cycle] cycle {cycle_idx + 1}/{args.cycles}")
            for action in sequence:
                _publish_action(bridge, action, dry_run=dry_run)
                sleep_fn(args.hold_s)
    finally:
        if args.open_on_exit:
            _publish_action(bridge, open_action, dry_run=dry_run)
        close = getattr(bridge, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cycle Unitree G1 right Dex3 hand open, thumb-index pinch, and three-finger close commands."
    )
    parser.add_argument("--interface", default=DEFAULT_G1_NETWORK_INTERFACE)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--cmd-topic", default=DEFAULT_G1_DEX3_RIGHT_CMD_TOPIC)
    parser.add_argument("--state-topic", default=DEFAULT_G1_DEX3_RIGHT_STATE_TOPIC)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--write-timeout", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true", help="Actually publish to the real robot. Default is dry-run.")
    parser.add_argument(
        "--no-wait-for-state",
        dest="wait_for_state",
        action="store_false",
        help="Do not require a Dex3 hand state message before publishing real commands.",
    )
    parser.add_argument(
        "--no-open-on-exit",
        dest="open_on_exit",
        action="store_false",
        help="Do not send an extra open command in the cleanup path.",
    )
    parser.set_defaults(wait_for_state=True, open_on_exit=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
