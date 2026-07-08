from __future__ import annotations

import argparse
import socket
import time

try:
    from .protocol import connect_with_retry, frame_to_g1_lowlevel_message, recv_framed, send_framed
except ImportError:  # Allow running as `python client_to_capx.py`.
    from protocol import connect_with_retry, frame_to_g1_lowlevel_message, recv_framed, send_framed


def _connect_capx(host: str, port: int) -> socket.socket:
    return connect_with_retry(host, port, retry_s=1.0)


def run(args: argparse.Namespace) -> None:
    robot_sock: socket.socket | None = None
    capx_sock: socket.socket | None = None
    forwarded = 0

    while True:
        try:
            if robot_sock is None:
                print(
                    f"[g1-vision-client] connecting to publisher "
                    f"{args.robot_host}:{args.robot_port}"
                )
                robot_sock = connect_with_retry(args.robot_host, args.robot_port, retry_s=1.0)
                print("[g1-vision-client] publisher connected")

            if capx_sock is None:
                print(f"[g1-vision-client] connecting to CaP-X {args.capx_host}:{args.capx_port}")
                capx_sock = _connect_capx(args.capx_host, args.capx_port)
                print("[g1-vision-client] CaP-X connected")

            frame = recv_framed(robot_sock)
            observation = frame_to_g1_lowlevel_message(frame)
            send_framed(capx_sock, observation)
            _latest_action = recv_framed(capx_sock)
            forwarded += 1
            if forwarded % args.log_every == 0:
                print(
                    f"[g1-vision-client] forwarded {forwarded} frames "
                    f"from {frame.get('camera_name')} at t={frame.get('timestamp'):.3f}"
                )
        except (EOFError, OSError, ValueError) as exc:
            print(f"[g1-vision-client] bridge error, reconnecting: {exc}")
            if robot_sock is not None:
                robot_sock.close()
                robot_sock = None
            if capx_sock is not None:
                capx_sock.close()
                capx_sock = None
            time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive G1 RGB-D frames and forward them to CaP-X G1RealLowLevel."
    )
    parser.add_argument("--robot-host", required=True, help="G1/onboard publisher IP address.")
    parser.add_argument("--robot-port", type=int, default=9100)
    parser.add_argument("--capx-host", default="127.0.0.1")
    parser.add_argument("--capx-port", type=int, default=9000)
    parser.add_argument("--log-every", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
