"""Serve local CaP-X run records and ingest the legacy SSH launcher stream."""

import argparse
import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, unquote, urlsplit

from dashboard_history import historical_run, local_asset, read_json, tron2_history
from tron2_paths import TRON2_REPO


HERE = Path(__file__).resolve().parent
TRON2_OUTPUTS = TRON2_REPO / "outputs"
TRON2_RUNS = TRON2_OUTPUTS / "tron2_dashboard_runs"
LEGACY_OUTPUTS = TRON2_REPO.parent / "outputs"
if not (LEGACY_OUTPUTS / "m3_dashboard_runs").is_dir():
    LEGACY_OUTPUTS = TRON2_OUTPUTS
if legacy_root := os.environ.get("CAPX_LEGACY_OUTPUTS"):
    LEGACY_OUTPUTS = Path(legacy_root).expanduser().resolve()
LEGACY_RUNS = LEGACY_OUTPUTS / "m3_dashboard_runs"
PREFIX = "CAPX_EVENT\t"
RUN_ID = re.compile(r"m3_stack_codex_mac_\d{8}_\d{6}(?:_review)?")
COMPONENT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}")
ARCHIVE_ID = re.compile(r"capx-27b-mm-paired-\d{8}")
TRON2_STATES = {"pending", "running", "complete", "interrupted", "error"}


@dataclass(frozen=True)
class Tron2Episode:
    run_id: str
    episode_id: str
    tier: str
    seed: int
    variant: str
    state: str
    attempt: Path | None
    cursor_key: str
    mode: str = "live"

    @property
    def selection(self):
        return f"tron2:{self.run_id}:{self.episode_id}"

    @property
    def name(self):
        variant = f" {self.variant}" if self.variant else ""
        return f"TRON2{variant} {self.tier} · seed {self.seed} · {self.run_id}"

    def metadata(self):
        return {"id": self.selection, "name": self.name, "mode": self.mode, "state": self.state}


def manifest_episodes(run_id):
    root = (TRON2_RUNS / run_id).resolve()
    if root.parent != TRON2_RUNS.resolve():
        return
    manifest = read_json(root / "manifest.json", {})
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        return
    rows = manifest.get("episodes")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        episode_id = row.get("id")
        tier, seed = row.get("tier"), row.get("seed")
        state = row.get("state")
        if (not isinstance(episode_id, str) or not COMPONENT.fullmatch(episode_id)
                or not isinstance(tier, str) or tier not in {"M1", "M2", "M3", "M4", "S1", "S2", "S3", "S4"}
                or type(seed) is not int or not isinstance(state, str) or state not in TRON2_STATES):
            continue
        attempt = None
        cursor_key = ""
        active = row.get("active_attempt")
        if isinstance(active, str):
            part = Path(active)
            if (len(part.parts) == 3 and part.parts[:2] == ("episodes", episode_id)
                    and COMPONENT.fullmatch(part.name)):
                target = (root / part).resolve()
                if target.is_relative_to(root / "episodes" / episode_id):
                    attempt, cursor_key = target, part.as_posix()
        variant = row.get("variant")
        yield Tron2Episode(run_id, episode_id, tier, seed, variant if isinstance(variant, str) else "",
                           state, attempt, cursor_key)


def archive_episodes(run_id):
    root = (TRON2_OUTPUTS / run_id).resolve()
    if root.parent != TRON2_OUTPUTS.resolve():
        return
    report = read_json(root / "episodes.json", [])
    if not isinstance(report, list):
        return
    for row in report:
        if not isinstance(row, dict):
            continue
        phase, variant, tier, seed = (row.get(key) for key in ("phase", "variant", "tier", "seed"))
        if phase not in {"smoke", "paired"} or variant not in {"v1", "v2"} or tier not in {"M1", "M2", "M3"}:
            continue
        if type(seed) is not int or seed < 0:
            continue
        attempt = (root / phase / variant / tier / f"seed-{seed}").resolve()
        if not attempt.is_relative_to(root) or not (attempt / "evaluation/results.json").is_file():
            continue
        episode_id = f"{phase}-{variant}-{tier}-seed-{seed:03d}"
        state = "complete" if not row.get("error") else "error"
        yield Tron2Episode(run_id, episode_id, tier, seed, variant, state, attempt, "archive", "replay")


def tron2_episode(selection):
    prefix, *parts = selection.split(":")
    if prefix != "tron2" or len(parts) != 2 or not all(COMPONENT.fullmatch(part) for part in parts):
        raise ValueError("Invalid TRON2 selection")
    run_id, episode_id = parts
    matches = manifest_episodes(run_id)
    episode = next((entry for entry in matches if entry.episode_id == episode_id), None)
    if episode is None and ARCHIVE_ID.fullmatch(run_id):
        episode = next((entry for entry in archive_episodes(run_id) if entry.episode_id == episode_id), None)
    if episode is None:
        raise FileNotFoundError(selection)
    return episode


def write_atomic(path, data):
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_bytes(data)
    temp.replace(path)


def write_json(path, data):
    write_atomic(path, json.dumps(data, ensure_ascii=False).encode())


def directory(run_id):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    root = LEGACY_OUTPUTS if run_id.endswith("_review") else LEGACY_RUNS
    return root / run_id


def record(run_id):
    root = directory(run_id)
    root.mkdir(mode=0o700, parents=True)
    (root / "events").mkdir()
    write_json(root / "state.json", {"state": "running"})
    index = 0
    for line in sys.stdin:
        if not line.startswith(PREFIX):
            print(line, end="", flush=True)
            continue
        try:
            event = json.loads(line[len(PREFIX):])
            if event["type"] == "frame":
                image = base64.b64decode(event["jpeg"], validate=True)
                if image.startswith(b"\xff\xd8") and image.endswith(b"\xff\xd9"):
                    write_atomic(root / "latest.jpg", image)
                continue
            write_json(root / "events" / f"{index:08d}.json", event)
            index += 1
        except (KeyError, TypeError, ValueError):
            print("Could not decode a dashboard event.", file=sys.stderr, flush=True)
    write_json(root / "state.json", {"state": "finalizing"})


def finish(run_id, exit_code):
    root = directory(run_id)
    if not (root / "state.json").is_file():
        return
    state = "complete" if exit_code == 0 else "failed"
    write_json(root / "state.json", {"state": state, "exit_code": exit_code})


def state_for(root):
    state_path = root / "state.json"
    if state_path.is_file():
        return json.loads(state_path.read_bytes())["state"]
    return "complete"


def run_entries():
    entries = []
    for root in (*LEGACY_OUTPUTS.glob("m3_stack_codex_mac_*_review"), *LEGACY_RUNS.glob("m3_stack_codex_mac_*")):
        if root.is_dir() and RUN_ID.fullmatch(root.name):
            replay = root.name.endswith("_review")
            entries.append({
                "id": root.name,
                "name": root.name.removesuffix("_review"),
                "mode": "replay" if replay else "live",
                "state": state_for(root),
            })
    for root in TRON2_RUNS.glob("*"):
        if COMPONENT.fullmatch(root.name):
            entries.extend(episode.metadata() for episode in manifest_episodes(root.name))
    for root in TRON2_OUTPUTS.glob("capx-27b-mm-paired-*"):
        if ARCHIVE_ID.fullmatch(root.name):
            entries.extend(episode.metadata() for episode in archive_episodes(root.name))
    return sorted(entries, key=lambda entry: entry["id"], reverse=True)


def tron2_snapshot(episode, after, cursor):
    attempt = episode.attempt
    event_paths = (
        sorted(path for path in (attempt / "events").glob("*.json")
               if path.stem.isdigit() and path.resolve().is_relative_to(attempt))
        if attempt is not None else []
    )
    history = (
        tron2_history(attempt, include_events=not event_paths)
        if attempt is not None else {"events": [], "summary": "", "metrics": {}, "video": None, "log": None}
    )
    total = len(event_paths) if event_paths else len(history["events"])
    reset = (cursor is not None and cursor != episode.cursor_key) or after > total
    start = 0 if reset else after
    events = [read_json(path, {}) for path in event_paths[start:]] if event_paths else history["events"][start:]
    frame = local_asset(attempt, "latest.jpg") if attempt is not None else None
    return history | {
        "name": episode.name,
        "state": episode.state,
        "events": events,
        "next": total,
        "cursor_key": episode.cursor_key,
        "reset_cursor": reset,
        "frame_revision": frame.stat().st_mtime_ns // 1000 if frame is not None else None,
    }


def run_snapshot(run_id, after, cursor=None):
    if run_id.startswith("tron2:"):
        return tron2_snapshot(tron2_episode(run_id), after, cursor)
    root = directory(run_id)
    if not root.is_dir():
        raise FileNotFoundError(run_id)
    if run_id.endswith("_review"):
        snapshot = historical_run(root)
        events = snapshot["events"]
        return snapshot | {
            "events": events[after:], "next": len(events), "state": "complete",
            "frame_revision": None,
        }
    state = state_for(root)
    events = sorted((root / "events").glob("*.json"))
    summary = root / "summaries.txt"
    frame = root / "latest.jpg"
    videos = sorted(root.glob("trial_*/video_combined.mp4"))
    return {
        "name": run_id,
        "summary": summary.read_text() if summary.is_file() else "",
        "events": [json.loads(path.read_bytes()) for path in events[after:]],
        "video": videos[0].relative_to(root).as_posix() if videos else None,
        "frame_revision": frame.stat().st_mtime_ns // 1000 if frame.is_file() else None,
        "next": len(events),
        "state": state,
    }


def media_root(run_id):
    if not run_id.startswith("tron2:"):
        return directory(run_id)
    episode = tron2_episode(run_id)
    if episode.attempt is None:
        raise FileNotFoundError(run_id)
    return episode.attempt


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.handle_get(include_body=True)

    def do_HEAD(self):
        self.handle_get(include_body=False)

    def handle_get(self, include_body):
        host = self.headers.get("Host", "").lower()
        port = self.server.server_port
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.send_error(403)
            return
        url = urlsplit(self.path)
        parts = unquote(url.path).strip("/").split("/")
        try:
            if parts == ["api", "runs"]:
                self.send_json(run_entries(), include_body)
            elif parts[:2] == ["api", "run"] and len(parts) == 3:
                query = parse_qs(url.query, keep_blank_values=True)
                after = max(0, int(query.get("after", ["0"])[0]))
                self.send_json(run_snapshot(parts[2], after, query.get("cursor", [None])[0]), include_body)
            elif parts[:2] == ["api", "run"] and parts[3:] == ["frame.jpg"]:
                root = media_root(parts[2]).resolve()
                frame = (root / "latest.jpg").resolve()
                if not frame.is_relative_to(root):
                    raise FileNotFoundError(frame)
                self.send_file(frame, include_body)
            elif parts[0] == "media" and len(parts) >= 3:
                root = media_root(parts[1]).resolve()
                asset = root.joinpath(*parts[2:]).resolve()
                allowed_log = parts[1].startswith("tron2:") and asset == root / "tabletop.log"
                if not asset.is_file() or not asset.is_relative_to(root) or (
                    asset.suffix not in {".mp4", ".png", ".jpg", ".json", ".txt", ".sh"} and not allowed_log
                ):
                    self.send_error(404)
                    return
                self.send_file(asset, include_body)
            else:
                frontend = {"": "index.html", "index.html": "index.html", "app.js": "app.js", "style.css": "style.css"}
                self.send_file(HERE / "dashboard_ui" / frontend["/".join(parts)], include_body)
        except (FileNotFoundError, KeyError):
            self.send_error(404)
        except ValueError:
            self.send_error(400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, data, include_body):
        data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def send_file(self, path, include_body):
        with path.open("rb") as source:
            source.seek(0, 2)
            size = source.tell()
            start, stop = 0, size - 1
            requested_range = self.headers.get("Range")
            if requested_range:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested_range)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    if last:
                        stop = min(stop, int(last))
                else:
                    start = max(0, size - int(last))
                if start > stop:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if requested_range else 200)
            content_type = "text/plain; charset=utf-8" if path.suffix == ".log" else mimetypes.guess_type(path)[0]
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            remaining = stop - start + 1
            self.send_header("Content-Length", str(remaining))
            if requested_range:
                self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
            self.end_headers()
            if include_body:
                source.seek(start)
                while remaining and (block := source.read(min(remaining, 65536))):
                    self.wfile.write(block)
                    remaining -= len(block)

    def log_message(self, *_):
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("serve").add_argument("--port", type=int, default=8120)
    for action in ("record", "finish", "run-dir"):
        command = actions.add_parser(action)
        command.add_argument("run_id")
        if action == "finish":
            command.add_argument("exit_code", type=int)
    args = parser.parse_args()
    if args.action == "serve":
        with ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler) as server:
            print(f"CaP-X dashboard at http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    elif args.action == "record":
        record(args.run_id)
    elif args.action == "finish":
        finish(args.run_id, args.exit_code)
    else:
        print(directory(args.run_id))


if __name__ == "__main__":
    main()
