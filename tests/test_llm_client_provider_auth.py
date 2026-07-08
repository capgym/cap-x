from __future__ import annotations


class DummyResponse:
    status_code = 200
    text = "ok"

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return {"choices": [{"message": {"content": "done"}}]}


def test_query_model_uses_capx_llm_api_key_for_openai_compatible_provider(monkeypatch) -> None:
    from capx.llm.client import ModelQueryArgs, query_model

    captured = {}

    def fake_post(url, *, headers, data, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setenv("CAPX_LLM_API_KEY", "capx-key")
    monkeypatch.delenv("PARATERA_API_KEY", raising=False)
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    out = query_model(
        ModelQueryArgs(
            model="Qwen3.6-Plus",
            server_url="https://llmapi.paratera.com/v1/chat/completions",
            api_key=None,
        ),
        [{"role": "user", "content": "hello"}],
    )

    assert out["content"] == "done"
    assert captured["url"] == "https://llmapi.paratera.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer capx-key"


def test_query_model_uses_paratera_api_key_when_generic_key_is_absent(monkeypatch) -> None:
    from capx.llm.client import ModelQueryArgs, query_model

    captured = {}

    def fake_post(url, *, headers, data, timeout):
        captured["headers"] = headers
        return DummyResponse()

    monkeypatch.delenv("CAPX_LLM_API_KEY", raising=False)
    monkeypatch.setenv("PARATERA_API_KEY", "paratera-key")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    query_model(
        ModelQueryArgs(
            model="GLM-5.1",
            server_url="https://llmapi.paratera.com/v1/chat/completions",
            api_key=None,
        ),
        [{"role": "user", "content": "hello"}],
    )

    assert captured["headers"]["Authorization"] == "Bearer paratera-key"


def test_query_model_explicit_api_key_overrides_environment(monkeypatch) -> None:
    from capx.llm.client import ModelQueryArgs, query_model

    captured = {}

    def fake_post(url, *, headers, data, timeout):
        captured["headers"] = headers
        return DummyResponse()

    monkeypatch.setenv("CAPX_LLM_API_KEY", "env-key")
    monkeypatch.setenv("PARATERA_API_KEY", "paratera-key")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    query_model(
        ModelQueryArgs(
            model="DeepSeek-V4-Pro",
            server_url="https://llmapi.paratera.com/v1/chat/completions",
            api_key="explicit-key",
        ),
        [{"role": "user", "content": "hello"}],
    )

    assert captured["headers"]["Authorization"] == "Bearer explicit-key"
