from __future__ import annotations

import requests


def test_post_with_retries_includes_http_error_response_body(monkeypatch) -> None:
    from capx.utils.serve_utils import post_with_retries

    class FakeResponse:
        status_code = 500
        text = "IK solve failed: target is unreachable"

        def raise_for_status(self) -> None:
            raise requests.HTTPError("500 Server Error", response=self)

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("capx.utils.serve_utils.requests.post", fake_post)
    monkeypatch.setattr("capx.utils.serve_utils.time.sleep", lambda _seconds: None)

    try:
        post_with_retries("http://127.0.0.1:8116/ik", {"target": []}, timeout_seconds=1.0, max_retries=1)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected post_with_retries to raise after HTTP 500")

    assert "500 Server Error" in message
    assert "IK solve failed: target is unreachable" in message
