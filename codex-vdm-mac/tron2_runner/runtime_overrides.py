"""Apply eval-only time budgets to the exact stock TRON2 runner sources."""

from __future__ import annotations

from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
import os
import sys


def _seconds(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class EvalLoader(SourceFileLoader):
    def __init__(self, original: SourceFileLoader, changes: tuple) -> None:
        super().__init__(original.name, original.path)
        self.changes = changes

    def get_code(self, fullname: str):
        path = self.get_filename(fullname)
        source = self.get_data(path).decode("utf-8")
        for old, new, required_count in self.changes:
            if source.count(old) != required_count:
                raise RuntimeError(f"Unexpected runner source in {path}. Cannot apply matched trial budgets for {old!r}.")
            source = source.replace(old, new)
        return self.source_to_code(source, path)


class EvalFinder(MetaPathFinder):
    def __init__(self, changes: dict) -> None:
        self.changes = changes

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname not in self.changes:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, SourceFileLoader):
            raise ImportError(f"Cannot find Python source for {fullname}")
        spec.loader = EvalLoader(spec.loader, self.changes[fullname])
        return spec


_PAYLOAD_ANCHOR = '    headers = {"Content-Type": "application/json"}\n    if args.api_key:'
_PAYLOAD_CONTROL = '''    if args.model == os.environ.get("CAPX_THINKING_CODE_MODEL"):
        mode = os.environ["CAPX_THINKING_MODE"]
        if mode not in {"thinking_off", "thinking_on"}:
            raise ValueError(f"Invalid Qwen thinking mode: {mode}")
        payload["chat_template_kwargs"] = {"enable_thinking": mode == "thinking_on"}
        if mode == "thinking_on":
            payload["reasoning_effort"] = "medium"
            payload["thinking_token_budget"] = 4096
        images = sum(
            part.get("type") == "image_url"
            for message in payload.get("messages", [])
            for part in message.get("content", [])
            if isinstance(part, dict)
        )
        print(f"MODEL_INPUT model={args.model} images={images} max_tokens={args.max_tokens} thinking_budget={payload.get('thinking_token_budget')}", flush=True)
    headers = {"Content-Type": "application/json"}
    if args.api_key:'''
_USAGE_ANCHOR = "    body = response.json()\n    out = {}"
_USAGE_LOG = '''    body = response.json()
    if body.get("choices"):
        choice = body["choices"][0]
        message = choice.get("message", {})
        thought = message.get("reasoning") or message.get("reasoning_content") or ""
        print("MODEL_RESPONSE " + json.dumps({
            "model": args.model,
            "finish_reason": choice.get("finish_reason"),
            "reasoning_chars": len(thought),
            "answer_chars": len(message.get("content") or ""),
            "usage": body.get("usage"),
        }), flush=True)
    out = {}'''

_UNAVAILABLE_REASON = "Model response was truncated at its token limit without final code or a decision."
_QUERY_RESULT_ANCHOR = '    return out  # type: ignore[return-value]'
_QUERY_RESULT = '''    if args.model == os.environ.get("CAPX_THINKING_CODE_MODEL") and body.get("choices"):
        finish_reason = body["choices"][0].get("finish_reason")
        if finish_reason == "length" and not (out["content"] or "").strip():
            out["content"] = ""
            out["incomplete_response"] = ''' + repr(_UNAVAILABLE_REASON) + '''
    return out  # type: ignore[return-value]'''

_INITIAL_RESULT_ANCHOR = '    return out["content"], out["reasoning"], ensemble_data'
_INITIAL_RESULT = '    return out["content"], out["reasoning"], ensemble_data, out.get("incomplete_response")'
_DECISION_PARSE_ANCHOR = '    decision, new_code = _parse_multi_turn_decision(content["content"])'
_DECISION_PARSE = '''    if reason := content.get("incomplete_response"):
        decision, new_code = "response_truncated", reason
    else:
        decision, new_code = _parse_multi_turn_decision(content["content"])'''
_BLOCK_PARSE_ANCHOR = '    initial_blocks = _extract_code(raw_code)'
_BLOCK_PARSE = '    initial_blocks = [] if initial_unavailable else _extract_code(raw_code)'
_TRIAL_START_ANCHOR = '    terminated = truncated = False\n    code_block_idx = 0'
_TRIAL_START = '''    terminated = False
    truncated = bool(initial_unavailable)
    if initial_unavailable:
        print(f"MODEL_OUTPUT_UNAVAILABLE stage=initial reason={initial_unavailable}", flush=True)
        info_step.update(stderr=initial_unavailable, task_completed=False)
        if partial_artifacts is not None:
            partial_artifacts.update(truncated=True, info_step=info_step)
    code_block_idx = 0'''
_RESPONSE_TRUNCATED = '''            elif decision == "response_truncated":
                print(f"MODEL_OUTPUT_UNAVAILABLE stage=decision turn={code_block_idx} reason={new_code}",
                      flush=True)
                all_responses.append({
                    "decision": "response_truncated",
                    "block_idx": [code_block_idx],
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "incomplete_response": new_code,
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                stderr_history.append(new_code)
                truncated = True
                if partial_artifacts is not None:
                    partial_artifacts["truncated"] = True
                break

            elif decision == "finish":
'''

def install() -> None:
    request = _seconds("CAPX_THINKING_REQUEST_TIMEOUT_S", 900)
    trial = _seconds("CAPX_THINKING_TRIAL_TIMEOUT_S", 6000)
    relay = request + 60
    changes = {
        "scripts.run_tron2_model": (
            ("settimeout(220)", f"settimeout({relay})", 2),
            ("tuple(peers), (), (), 220)", f"tuple(peers), (), (), {relay})", 1),
        ),
        "capx.llm.client": (
            ("timeout=200", f"timeout={request}", 3),
            (_PAYLOAD_ANCHOR, _PAYLOAD_CONTROL, 2),
            (_USAGE_ANCHOR, _USAGE_LOG, 1),
            (_QUERY_RESULT_ANCHOR, _QUERY_RESULT, 1),
        ),
        "capx.envs.runner": (
            ("TRIAL_TIMEOUT_SECONDS = 1000", f"TRIAL_TIMEOUT_SECONDS = {trial}", 1),
        ),
        "capx.envs.trial": (
            ("signal.alarm(1000)", f"signal.alarm({trial})", 1),
            ("from PIL import Image\n", "from PIL import Image\n\nfrom model_feedback import compact_servoj_feedback\n", 1),
            ('        console_stdout=info_step["stdout"],\n        console_stderr=info_step["stderr"],',
             '        console_stdout=compact_servoj_feedback(info_step["stdout"]),\n'
             '        console_stderr=compact_servoj_feedback(info_step["stderr"]),', 1),
            ('    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])',
             '    task_description = "Lift the red cube at least 10 cm with the left arm."', 1),
            (_INITIAL_RESULT_ANCHOR, _INITIAL_RESULT, 1),
            ("    # --- 3. Initial code generation ---", "    initial_unavailable = None\n    # --- 3. Initial code generation ---", 1),
            ("raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)",
             "raw_code, reasoning, ensemble_data, initial_unavailable = _query_initial_code(args, config, obs)", 1),
            (_BLOCK_PARSE_ANCHOR, _BLOCK_PARSE, 1),
            ('        "decision": "initial",', '        "decision": "initial",\n        "incomplete_response": initial_unavailable,', 1),
            (_TRIAL_START_ANCHOR, _TRIAL_START, 1),
            (_DECISION_PARSE_ANCHOR, _DECISION_PARSE, 1),
            ('            elif decision == "finish":\n', _RESPONSE_TRUNCATED, 1),
            ("if recording_frames and turn_frame_ranges:", "if recording_frames and (turn_frame_ranges or initial_unavailable):", 1),
        ),
    }
    if loaded := changes.keys() & sys.modules.keys():
        raise RuntimeError(f"Budget overrides must install before importing {sorted(loaded)}")
    sys.meta_path.insert(0, EvalFinder(changes))
