"""Load evidence saved by the M3 and TRON2 trial runners."""

from pathlib import Path
import json
import re


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def read_json(path: Path, default=None):
    """Read a local artifact that might still be in transit from a worker."""
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return default


def local_asset(root: Path, relative: str) -> Path | None:
    """Resolve an existing file without following paths outside its archive."""
    if not relative or Path(relative).is_absolute():
        return None
    path = (root / relative).resolve()
    return path if path.is_relative_to(root.resolve()) and path.is_file() else None


def tron2_history(attempt: Path, *, include_events: bool) -> dict:
    """Read the completed stock report and any saved interactions for one episode."""
    evaluation = attempt / "evaluation"
    report = local_asset(attempt, "evaluation/results.json")
    results = read_json(report, {}) if report is not None else {}
    trials = results.get("trials", []) if isinstance(results, dict) else []
    trial = trials[0] if isinstance(trials, list) and trials and isinstance(trials[0], dict) else {}
    relative = trial.get("trial_artifacts")
    trial_dir = (evaluation / relative).resolve() if isinstance(relative, str) else None
    if trial_dir is not None and not trial_dir.is_relative_to(attempt.resolve()):
        trial_dir = None

    def path_in_trial(filename):
        if trial_dir is None:
            return None
        return local_asset(attempt, (trial_dir / filename).relative_to(attempt.resolve()).as_posix())

    video = trial.get("video")
    combined = local_asset(attempt, f"evaluation/{video}") if isinstance(video, str) else None
    if combined is None:
        combined = path_in_trial("video_combined.mp4")
    summary = path_in_trial("summary.txt")
    responses = path_in_trial("all_responses.json")
    metrics = {
        key: trial[key]
        for key in ("task_completed", "reward", "episode_wall_sec", "final_lift_cm", "table_peak_normal_n")
        if trial.get(key) is not None
    }
    snapshot = {
        "summary": _text(summary) if summary is not None else "",
        "metrics": metrics,
        "video": combined.relative_to(attempt.resolve()).as_posix() if combined is not None else None,
        "events": [],
        "log": "tabletop.log" if local_asset(attempt, "tabletop.log") is not None else None,
    }
    if trial.get("error"):
        snapshot["summary"] = f"{trial['error']}\n\n{snapshot['summary']}".rstrip()
    if not include_events or trial_dir is None:
        return snapshot

    def add(kind, model, title, *, path=None, **fields):
        event = {"type": kind, "model": model, "title": title, **fields}
        if path is not None:
            event["asset"] = path.relative_to(attempt.resolve()).as_posix()
        snapshot["events"].append(event)

    initial = path_in_trial("visual_feedback_00.png")
    if initial is not None:
        add("frame", "sim", "初始仿真画面", path=initial, turn=0)
    stored = read_json(responses, []) if responses is not None else []
    for entry in stored if isinstance(stored, list) else ():
        if not isinstance(entry, dict):
            continue
        turn = entry.get("block_idx", [None])[0] if entry.get("block_idx") else None
        messages = entry.get("initial_prompt") or entry.get("multi_turn_prompt")
        if messages:
            add("request", "qwen", "Qwen 保存的消息", body={"messages": messages}, turn=turn)
        if entry.get("decision") == "initial" and (raw := path_in_trial("raw_response.sh")) is not None:
            add("reply", "qwen", "Qwen 保存的回复", text=_text(raw), turn=turn, path=raw)
        else:
            decision = entry.get("decision", "")
            reasoning = entry.get("reasoning") or entry.get("incomplete_response")
            code_blocks = entry.get("code_blocks") or []
            text = "\n\n".join(filter(None, (decision, reasoning, *code_blocks)))
            if text:
                add("reply", "qwen", "保存的回复", turn=turn, text=text, status="partial",
                    decision=decision, reasoning=reasoning, code_blocks=code_blocks)
        if isinstance(turn, int):
            if (clip := path_in_trial(f"video_turn_{turn:02d}.mp4")) is not None:
                add("frame", "sim", "执行录像", turn=turn, path=clip)
            if (image := path_in_trial(f"visual_feedback_{turn + 1:02d}.png")) is not None:
                add("frame", "sim", "执行后的画面", turn=turn, path=image)
    return snapshot


def _messages_text(messages: list[dict]) -> str:
    lines = []
    image_number = 0
    for message in messages:
        lines.append(f"[{message['role']}]")
        content = message["content"]
        if isinstance(content, str):
            lines.append(content)
        else:
            for part in content:
                if part["type"] == "text":
                    lines.append(part["text"])
                elif part["type"] == "image_url":
                    image_number += 1
                    lines.append(f"[图片 {image_number}]")
        lines.append("")
    return "\n".join(lines).rstrip()


def historical_run(review_dir: Path) -> dict:
    """Return model exchanges, observations and videos using local saved evidence."""
    result = {
        "name": review_dir.name.removesuffix("_review"),
        "summary": _text(review_dir / "summaries.txt"),
        "events": [],
        "video": None,
    }
    events = result["events"]

    def add(kind, model, title, text, turn=None, *, asset=None, status=None, assets=()):
        event = {"type": kind, "model": model, "title": title, "text": text}
        if turn is not None:
            event["turn"] = turn
        if asset is not None and asset.is_file():
            event["asset"] = asset.relative_to(review_dir).as_posix()
        if status:
            event["status"] = status
        if assets:
            event["assets"] = [path.relative_to(review_dir).as_posix() for path in assets if path.is_file()]
        events.append(event)

    trial_dir = next(iter(sorted(review_dir.glob("trial_*"))), None)
    if trial_dir is None:
        add("note", "sim", "缺少已保存的试验", "找不到这次运行的试验目录。", status="missing")
        return result

    combined = trial_dir / "video_combined.mp4"
    if combined.is_file():
        result["video"] = combined.relative_to(review_dir).as_posix()

    codex_dir = review_dir / "codex_requests_reconstructed"
    cli_text = _text(codex_dir / "codex_cli_stdin_readable.txt")
    cli_sections = re.split(r"^===== 请求 (\d+) =====\s*$", cli_text, flags=re.MULTILINE)
    cli_by_turn = {
        int(turn): prompt.strip()
        for turn, prompt in zip(cli_sections[1::2], cli_sections[2::2])
    }
    qwen_messages_file = review_dir / "qwen_initial_messages.json"
    all_responses = _text(trial_dir / "all_responses.json")
    responses = json.loads(all_responses) if all_responses else []
    by_turn = {
        response["block_idx"][0]: response
        for response in responses
        if response.get("block_idx")
    }
    videos_by_turn = {
        int(path.stem.removeprefix("video_turn_")): path
        for path in trial_dir.glob("video_turn_[0-9][0-9].mp4")
    }
    codex_turns = {
        int(path.name.partition(".")[0])
        for path in codex_dir.glob("[0-9][0-9].request.json")
    }

    initial_frame = trial_dir / "visual_feedback_00.png"
    if initial_frame.is_file():
        add("frame", "sim", "仿真初始画面", "CaP-X 在执行任何机器人代码之前保存的画面。", 0, asset=initial_frame)

    for turn in sorted(by_turn.keys() | videos_by_turn.keys() | codex_turns):
        current_frame = trial_dir / f"visual_feedback_{turn:02d}.png"
        frames = [current_frame]
        if turn:
            frames.insert(0, trial_dir / f"visual_feedback_{turn - 1:02d}.png")

        codex_request = codex_dir / f"{turn:02d}.request.json"
        if codex_request.is_file():
            payload = json.loads(_text(codex_request))
            text = "根据当时保存的画面与 CaP-X 代码重建。附件 JSON 包含消息和图片。"
            if turn in cli_by_turn:
                text += f"\n\n重建的 Codex CLI 输入和图片参数\n\n{cli_by_turn[turn]}"
            else:
                text += f"\n\n{_messages_text(payload['messages'])}"
            add("request", "codex", "Codex 请求", text, turn, asset=codex_request,
                assets=frames, status="reconstructed")
        else:
            add("request", "codex", "Codex 请求缺失", "没有找到这一轮的请求或重建文件。", turn,
                status="missing")

        codex_reply = codex_dir / f"{turn:02d}.reply_content.txt"
        reply_text = _text(codex_reply)
        if reply_text:
            add("reply", "codex", "Codex 回复文本",
                "这段回复来自保存的 Qwen 初始消息。原 HTTP 响应没有保存。\n\n" + reply_text.strip(),
                turn, asset=codex_reply, status="partial")
        else:
            add("reply", "codex", "Codex 回复缺失", "这一轮返回的原始回复没有保存。", turn, status="missing")

        response = by_turn.get(turn, {})
        if turn == 0:
            saved = _text(qwen_messages_file)
            messages = json.loads(saved) if saved else response.get("initial_prompt")
        else:
            messages = response.get("multi_turn_prompt")
        if messages:
            asset = qwen_messages_file if turn == 0 else None
            add("request", "qwen", "Qwen 请求消息", _messages_text(messages), turn, asset=asset)
        else:
            add("request", "qwen", "Qwen 请求缺失", "这一轮的完整消息没有保存。", turn, status="missing")

        raw_response = trial_dir / "raw_response.sh"
        if turn == 0 and raw_response.is_file():
            add("reply", "qwen", "Qwen 原始回复文本", _text(raw_response), turn, asset=raw_response)
        elif code_blocks := response.get("code_blocks"):
            decision = response["decision"].upper()
            code = "\n\n".join(code_blocks)
            add("reply", "qwen", "Qwen 回复节选",
                f"CaP-X 保存的决定为 {decision}，下方是提取出的代码。完整原始回复没有保存。\n\n{code}",
                turn, status="partial")
        else:
            add("reply", "qwen", "Qwen 回复缺失", "这一轮的回复没有保存。", turn, status="missing")

        video = videos_by_turn.get(turn)
        if video is not None:
            add("frame", "sim", f"执行代码块 {turn:02d}", "仿真实际录制的片段。", turn, asset=video)
        next_frame = trial_dir / f"visual_feedback_{turn + 1:02d}.png"
        if next_frame.is_file():
            add("frame", "sim", "执行后的画面", "下一次视觉请求使用的当前画面。", turn, asset=next_frame)

    trial_summary = trial_dir / "summary.txt"
    summary = _text(trial_summary)
    if "executing action in terminated episode" in summary:
        add("note", "sim", "试验截断",
            "最后一段视频记录了终止前的动作。执行中出现 `executing action in terminated episode`，"
            "没有再发出视觉请求。奖励为 0.032，叠方块任务未完成。", asset=trial_summary)

    return result
