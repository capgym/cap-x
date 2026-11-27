"""Serve CaP-X image feedback through the Codex CLI signed in on this Mac."""

import base64
import binascii
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import monotonic
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = "openai/codex-cli-vdm"
MAX_BODY_BYTES = 24 * 1024 * 1024
IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg"}


def describe_images(cli, messages, work_dir, timeout=150):
    prompt_parts = [
        "You are the visual feedback model for a robot simulation. "
        "Compare the attached observations and return only a short plain-text "
        "description of what changed, with respect to the task."
    ]
    image_paths = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            prompt_parts.append(content)
            continue
        for part in content:
            if part["type"] == "text":
                prompt_parts.append(part["text"])
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                header, payload = url.split(",", 1)
                media_type, encoding = header.removeprefix("data:").split(";", 1)
                if not header.startswith("data:") or encoding != "base64":
                    raise ValueError("Expected an inline base64-encoded image")
                suffix = IMAGE_SUFFIXES.get(media_type)
                if suffix is None:
                    raise ValueError("Expected a PNG or JPEG image")
                image_path = work_dir / f"image_{len(image_paths) + 1}{suffix}"
                image_path.write_bytes(base64.b64decode(payload, validate=True))
                image_paths.append(image_path)
                prompt_parts.append(f"[Attached image {len(image_paths)}]")
            else:
                raise ValueError("Unexpected message content type")
    if not image_paths:
        raise ValueError("At least one image is required")

    result_path = work_dir / "description.txt"
    command = [
        cli, "-a", "never", "exec", "--skip-git-repo-check", "--ephemeral",
        "--sandbox", "read-only", "--ignore-user-config", "--ignore-rules",
        "--disable", "shell_tool", "--disable", "apps", "--disable", "plugins",
        "--disable", "in_app_browser", "--output-last-message", str(result_path),
        "--image", *map(str, image_paths), "--", "-",
    ]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_API_KEY", None)
    process = subprocess.run(
        command, input="\n\n".join(prompt_parts), text=True,
        capture_output=True, cwd=work_dir, env=env, timeout=timeout,
    )
    if process.returncode:
        details = "\n".join(process.stderr.splitlines()[-8:])
        raise RuntimeError(f"Codex CLI exited with status {process.returncode}\n{details}")
    result = result_path.read_text(encoding="utf-8").strip()
    if not result:
        raise RuntimeError("Codex CLI returned an empty description")
    return result


class FeedbackHandler(BaseHTTPRequestHandler):
    timeout = 30

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        try:
            self.send_response_only(status)
            self.send_header("Server", self.version_string())
            self.send_header("Date", self.date_time_string())
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError as error:
            self.log_error("Could not write HTTP %s response to SSH tunnel: %s", status, error)
            return
        self.log_request(status, len(data))

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"model": MODEL})
        else:
            self.send_json(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        authorization = self.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(authorization, self.server.authorization):
            self.send_json(401, {"error": "Invalid bridge token"})
            return
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "Unknown endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY_BYTES:
                raise ValueError("Invalid request size")
            self.log_message("Receiving VDM request body (%d bytes)", size)
            try:
                raw_body = self.rfile.read(size)
            except TimeoutError:
                self.log_error("No request data received for %s seconds", self.timeout)
                self.send_json(408, {"error": "Timed out receiving the VDM image over SSH."})
                return
            except OSError as error:
                self.log_error("Could not receive VDM request body from SSH tunnel: %s", error)
                return
            if len(raw_body) != size:
                raise ValueError(f"Incomplete VDM request body ({len(raw_body)} of {size} bytes)")
            payload = json.loads(raw_body)
            if payload.get("model") != MODEL:
                raise ValueError(f"Expected model {MODEL}")
            self.log_message("Starting local Codex CLI")
            start = monotonic()
            with tempfile.TemporaryDirectory(prefix="capx-vdm-") as directory:
                description = describe_images(
                    self.server.cli, payload["messages"], Path(directory), self.server.cli_timeout
                )
            self.log_message("Local Codex CLI returned in %.1f seconds", monotonic() - start)
        except (ValueError, KeyError, TypeError, AttributeError, binascii.Error) as error:
            self.send_json(400, {"error": str(error)})
            return
        except subprocess.TimeoutExpired as error:
            self.log_error("Local Codex CLI timed out after %s seconds", error.timeout)
            self.send_json(424, {"error": f"Local Codex CLI timed out after {error.timeout} seconds."})
            return
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.log_error("%s", error)
            # CaP-X retries many 5xx statuses indefinitely. Surface this error immediately.
            self.send_json(424, {"error": "Local Codex call failed. Check the Mac terminal."})
            return
        self.send_json(200, {
            "model": MODEL,
            "choices": [{"message": {"role": "assistant", "content": description}}],
        })


def main():
    cli, port = sys.argv[1:]
    cli_timeout = int(os.environ.get("CAPX_CODEX_VDM_TIMEOUT", "150"))
    if not 1 <= cli_timeout <= 185:
        raise SystemExit("CAPX_CODEX_VDM_TIMEOUT must be between 1 and 185 seconds.")
    token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit("Missing bridge token on stdin")
    with HTTPServer(("127.0.0.1", int(port)), FeedbackHandler) as server:
        server.authorization = f"Bearer {token}".encode()
        server.cli = cli
        server.cli_timeout = cli_timeout
        print(f"Codex VDM listening on 127.0.0.1:{port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
