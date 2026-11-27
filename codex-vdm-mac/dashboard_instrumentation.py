"""Emit observation events from the existing headless CaP-X trial."""

import base64
import io
import json
import sys
import threading
import time
import uuid


def _decode_body(body):
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except ValueError:
            pass
    return body


_CREDENTIAL_FIELDS = {
    "access_token", "api_key", "authorization", "client_secret", "cookie",
    "password", "proxy_authorization", "refresh_token", "set_cookie", "token",
    "x_api_key",
}


def _field_name(name):
    return str(name).lower().replace("-", "_")


def _redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {
            _redact(key, secrets): (
                "[redacted]" if _field_name(key) in _CREDENTIAL_FIELDS else _redact(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


class _QwenStreamError(Exception):
    """The provider did not deliver a complete Chat Completions stream."""


class _QwenDeltas:
    """Publish the first safe text promptly and coalesce subsequent text."""

    fields = ("reasoning", "content")

    def __init__(self, send, secrets):
        self._send = send
        self._secrets = secrets
        self._prefixes = {secret[:length] for secret in secrets for length in range(1, len(secret))}
        self._tails = {field: "" for field in self.fields}
        self._parts = {field: [] for field in self.fields}
        self._lock = threading.Lock()
        self._timer = None
        self._started = False

    def _redact_piece(self, field, text):
        value = self._tails[field] + text
        safe = []
        offset = 0
        while offset < len(value):
            # A credential can be split across provider deltas or UI batches.
            if self._prefixes and value[offset:] in self._prefixes:
                break
            match = next((secret for secret in self._secrets if value.startswith(secret, offset)), None)
            if match:
                safe.append("[redacted]")
                offset += len(match)
            else:
                safe.append(value[offset])
                offset += 1
        self._tails[field] = value[offset:]
        self._parts[field].append("".join(safe))

    def add(self, **fields):
        with self._lock:
            for field, text in fields.items():
                if text:
                    self._redact_piece(field, text)
            if not self._started:
                self._flush()
            elif self._timer is None and any(any(parts) for parts in self._parts.values()):
                self._timer = threading.Timer(0.2, self._flush_delayed)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self):
        delta = {field: "".join(parts) for field, parts in self._parts.items() if any(parts)}
        for parts in self._parts.values():
            parts.clear()
        if delta:
            self._send(**delta)
            self._started = True

    def _flush_delayed(self):
        with self._lock:
            self._timer = None
            self._flush()

    def finish(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            for field in self.fields:
                self._parts[field].append(_redact(self._tails[field], self._secrets))
                self._tails[field] = ""
            self._flush()


def _sse_events(response):
    # Requests respects HTTP chunk boundaries for the vLLM response. For
    # non-chunked streams byte reads keep short events from waiting for a buffer.
    chunk_size = 64 if "chunked" in response.headers.get("Transfer-Encoding", "").lower() else 1
    data = []
    for line in response.iter_lines(chunk_size=chunk_size):
        if line.startswith(b"data:"):
            data.append(line[5:].removeprefix(b" "))
        elif line.startswith(b"event:") and line[6:].strip() == b"error":
            raise _QwenStreamError("Qwen returned an SSE error event")
        elif not line and data:
            yield b"\n".join(data)
            data.clear()


def _collect_qwen_stream(response, on_delta):
    if response.headers.get("Content-Type", "").split(";")[0].strip().lower() != "text/event-stream":
        raise _QwenStreamError("Qwen did not return an SSE response")
    metadata = {}
    choice_info = {}
    message = {"role": "assistant"}
    parts = {field: [] for field in _QwenDeltas.fields}
    usage = None
    finish_reason = None
    for event in _sse_events(response):
        if event.strip() == b"[DONE]":
            break
        try:
            chunk = json.loads(event)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise _QwenStreamError("Qwen returned invalid SSE JSON") from error
        if not isinstance(chunk, dict) or "error" in chunk:
            raise _QwenStreamError("Qwen returned an SSE error response")
        metadata.update({key: value for key, value in chunk.items() if key not in {"choices", "usage"}})
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not choices:
            continue
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise _QwenStreamError("Expected one Qwen completion choice")
        choice = choices[0]
        delta = choice.get("delta", {})
        if choice.get("index", 0) != 0 or not isinstance(delta, dict):
            raise _QwenStreamError("Qwen returned an invalid completion delta")
        choice_info.update({key: value for key, value in choice.items() if key not in {"delta", "finish_reason"}})
        finish_reason = choice.get("finish_reason") or finish_reason
        if role := delta.get("role"):
            message["role"] = role
        reason = delta.get("reasoning") or delta.get("reasoning_content")
        visible = delta.get("content")
        updates = {"reasoning": reason, "content": visible}
        for field, text in updates.items():
            if text is not None and not isinstance(text, str):
                raise _QwenStreamError("Qwen returned a non-text completion delta")
            if text:
                parts[field].append(text)
        on_delta(**updates)
    else:
        raise _QwenStreamError("Qwen stream closed before [DONE]")
    if not isinstance(finish_reason, str) or not finish_reason or usage is None:
        raise _QwenStreamError("Qwen stream ended without a finish reason and usage")
    message["content"] = "".join(parts["content"])
    message["reasoning"] = "".join(parts["reasoning"]) or None
    choice = choice_info | {"index": 0, "message": message, "finish_reason": finish_reason}
    return metadata | {"object": "chat.completion", "choices": [choice], "usage": usage}


def install_dashboard_tracing(args, stream=None, *, env_class=None, include_graspnet=True, stream_qwen=False):
    """Observe model requests, grasp planning, and existing RGB frame captures."""
    import requests
    from PIL import Image

    if env_class is None:
        from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv

        env_class = RobosuiteBaseEnv
    graspnet_plan = None
    if include_graspnet:
        from capx.integrations.vision.graspnet import SERVICE_URL

        graspnet_plan = f"{SERVICE_URL}/plan"

    models = {
        args.server_url: "qwen",
        args.visual_differencing_model_server_url: "codex",
    }
    secrets = {
        key for key in (
            getattr(args, "api_key", None),
            getattr(args, "visual_differencing_model_api_key", None),
        ) if key
    }
    event_stream = sys.stdout if stream is None else stream
    event_lock = threading.Lock()

    def emit(**event):
        line = "CAPX_EVENT\t" + json.dumps(event, separators=(",", ":"))
        with event_lock:
            print(line, file=event_stream, flush=True)

    original_post = requests.post

    def trace_graspnet(url, positional, options):
        title = "GraspNet 抓取诊断"
        payload = options.get("json")
        if isinstance(payload, dict):
            options["json"] = {**payload, "include_diagnostics": True}
        try:
            response = original_post(url, *positional, **options)
        except requests.exceptions.RequestException:
            emit(type="note", model="sim", title=title, text="连接失败，未收到抓取诊断。",
                 status="network_error")
            raise

        if response.status_code >= 400:
            emit(type="note", model="sim", title=title, text="抓取服务请求失败。",
                 status=str(response.status_code))
            return response

        try:
            body = response.json()
        except ValueError:
            body = None
        diagnostics = body.get("diagnostics") if isinstance(body, dict) else None
        if not isinstance(diagnostics, dict):
            emit(type="note", model="sim", title=title, text="抓取服务未返回诊断字段。",
                 status="missing")
            return response

        metrics = (
            ("掩膜像素", "mask_pixels"),
            ("有效深度", "valid_depth_pixels"),
            ("深度范围内", "in_range_depth_pixels"),
            ("目标点", "segmented_points"),
            ("返回前候选", "candidates_before_response"),
            ("返回候选", "returned_candidates"),
            ("重试次数", "retries"),
        )
        summary = " · ".join(
            f"{label} {diagnostics[field]}" for label, field in metrics
            if type(diagnostics.get(field)) is int
        ) or "抓取服务已返回诊断。"
        if diagnostics.get("discarded_at_retry_limit"):
            summary += " · 达到重试上限后丢弃了候选"
        emit(type="note", model="sim", title=title, text=summary, body=_redact(diagnostics, secrets))
        return response

    def traced_post(url, *positional, **options):
        if graspnet_plan is not None and url == graspnet_plan:
            return trace_graspnet(url, positional, options)
        model = models.get(url)
        if model is None:
            return original_post(url, *positional, **options)

        request_id = uuid.uuid4().hex
        headers = options.get("headers") or {}
        request_secrets = secrets | {
            value.split(" ", 1)[-1]
            for name, value in headers.items()
            if _field_name(name) in _CREDENTIAL_FIELDS and value
        }
        request_secrets = sorted(filter(None, request_secrets), key=len, reverse=True)
        source = positional[0] if positional else options.get("data")
        payload = _decode_body(source if source is not None else options.get("json"))
        # The stock policy caller includes this marker. The same endpoint also
        # gets a one-shot JSON color probe, which must remain non-streaming.
        template = payload.get("chat_template_kwargs") if isinstance(payload, dict) else None
        streaming = (stream_qwen and model == "qwen" and isinstance(template, dict)
                     and "enable_thinking" in template and not payload.get("stream") and not options.get("stream"))
        if streaming:
            payload = payload | {
                "stream": True,
                "stream_options": (payload.get("stream_options") or {}) | {"include_usage": True},
            }
            if positional:
                positional = (json.dumps(payload), *positional[1:])
            elif source is not None:
                options["data"] = json.dumps(payload)
            else:
                options["json"] = payload
            options.update(stream=True, allow_redirects=False)
        emit(type="request", model=model, text="API request", request_id=request_id,
             body=_redact(payload, request_secrets))

        try:
            response = original_post(url, *positional, **options)
            if streaming and response.status_code >= 400:
                # Preserve the stock caller's handling of HTTP errors returned
                # before generation. Only a failed in-flight stream is terminal.
                with response:
                    _ = response.content
            elif streaming:
                delta_batch = _QwenDeltas(
                    lambda **delta: emit(type="delta", model=model, request_id=request_id, **delta),
                    request_secrets,
                )
                try:
                    with response:
                        if response.is_redirect:
                            raise _QwenStreamError("Qwen unexpectedly redirected the streaming request")
                        body = _collect_qwen_stream(response, delta_batch.add)
                finally:
                    delta_batch.finish()
                response._content = json.dumps(body, ensure_ascii=False).encode()
                response._content_consumed = True
                response.encoding = "utf-8"
                response.headers.pop("Transfer-Encoding", None)
                response.headers.pop("Content-Encoding", None)
                response.headers["Content-Type"] = "application/json; charset=utf-8"
                response.headers["Content-Length"] = str(len(response.content))
        except _QwenStreamError as error:
            emit(type="reply", model=model, text="API stream error", request_id=request_id,
                 status="stream_error", body=str(error))
            raise
        except requests.exceptions.RequestException as error:
            status = (str(error.response.status_code) if streaming and error.response is not None
                      else "network_error")
            emit(type="reply", model=model, text=type(error).__name__, request_id=request_id,
                 status=status, body=_redact(str(error), request_secrets))
            raise

        try:
            body = response.json()
        except ValueError:
            body = response.text
        emit(type="reply", model=model, text="API response", request_id=request_id,
             status=str(response.status_code), body=_redact(body, request_secrets))
        return response

    frame_ready = threading.Condition()
    pending_frame = None
    last_frame_at = float("-inf")
    preview_enabled = True
    original_record_frame = env_class._record_frame

    def publish_frames():
        nonlocal pending_frame, preview_enabled
        while True:
            with frame_ready:
                frame_ready.wait_for(lambda: pending_frame is not None)
                frame = pending_frame
                pending_frame = None
            try:
                image = Image.fromarray(frame)
                image.thumbnail((384, 384), Image.Resampling.BILINEAR)
                with io.BytesIO() as output:
                    image.save(output, format="JPEG", quality=70)
                    jpeg = base64.b64encode(output.getvalue()).decode("ascii")
                emit(type="frame", model="sim", text="Simulation camera", jpeg=jpeg)
            except Exception as error:
                preview_enabled = False
                emit(type="note", model="sim", text=f"Camera preview stopped ({type(error).__name__})")
                return

    def record_frame(self):
        nonlocal pending_frame, last_frame_at
        before = len(self._frame_buffer)
        original_record_frame(self)
        now = time.monotonic()
        if preview_enabled and len(self._frame_buffer) > before and now - last_frame_at >= 0.25:
            last_frame_at = now
            with frame_ready:
                pending_frame = self._frame_buffer[-1]
                frame_ready.notify()

    requests.post = traced_post
    env_class._record_frame = record_frame
    threading.Thread(target=publish_frames, name="capx-dashboard-camera", daemon=True).start()
