"""Exercise HTTP clients without importing unrelated robot/simulator integrations."""

import base64
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import requests
from PIL import Image

from capx.utils import serve_utils


@pytest.fixture(params=["sam3", "molmo"])
def detector(request, monkeypatch):
    path = Path(__file__).parents[1] / "capx/integrations/vision" / f"{request.param}.py"
    spec = importlib.util.spec_from_file_location(request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(serve_utils.time, "sleep", lambda seconds: None)
    image = Image.new("RGB", (32, 32))
    if request.param == "sam3":
        client = module.init_sam3()
        return request.param, lambda: client(image, "cube"), 5
    client = module.init_molmo()
    return request.param, lambda: client(image, objects=["cube"]), 3


def response(status, data):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode()
    return result


@pytest.mark.parametrize("failure", ["connection", "timeout", 400, 422, 429, 503])
def test_request_failures_remain_errors(detector, monkeypatch, failure):
    name, call, attempts = detector
    if failure == "connection":
        error = requests.ConnectionError("offline")
    elif failure == "timeout":
        error = requests.Timeout("slow service")
    else:
        error = requests.HTTPError(response=response(failure, {}))
    post = Mock(side_effect=error)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(RuntimeError if name == "sam3" else type(error)) as caught:
        call()
    actual = caught.value.__cause__ if name == "sam3" else caught.value
    assert actual is error
    assert post.call_count == attempts


@pytest.mark.parametrize("empty", [False, True])
def test_successful_detection_results(detector, monkeypatch, empty):
    name, call, _ = detector
    if name == "sam3":
        item = {
            "shape": [1, 1],
            "mask_base64": base64.b64encode(b"\x01").decode(),
            "box": [0, 0, 1, 1],
            "score": 0.9,
            "label": "cube",
        }
        data = {"results": [] if empty else [item]}
    else:
        text = "No object found." if empty else '<point x="50" y="25">cube</point>'
        data = {"choices": [{"message": {"content": text}}]}
    post = Mock(return_value=response(200, data))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests.Session, "post", post)
    result = call()
    if name == "sam3":
        assert len(result) == (0 if empty else 1)
        if not empty:
            np.testing.assert_array_equal(result[0]["mask"], [[True]])
            assert result[0]["score"] == 0.9
    else:
        assert result == {"cube": (None, None) if empty else (16, 8)}
    post.assert_called_once()


@pytest.mark.parametrize(
    "helper", [serve_utils.post_with_retries, serve_utils.post_with_queue_tolerance]
)
@pytest.mark.parametrize("status", [400, 503])
def test_retry_exhaustion_preserves_http_response(monkeypatch, helper, status):
    reply = response(status, {})
    monkeypatch.setattr(requests, "post", Mock(return_value=reply))
    monkeypatch.setattr(serve_utils.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError) as caught:
        helper("http://test", {}, max_retries=1)
    assert isinstance(caught.value.__cause__, requests.HTTPError)
    assert caught.value.__cause__.response is reply


@pytest.mark.parametrize(
    "helper", [serve_utils.post_with_retries, serve_utils.post_with_queue_tolerance]
)
def test_transient_failure_then_success(monkeypatch, helper):
    post = Mock(side_effect=[requests.ConnectionError("offline"), response(200, {"ok": True})])
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(serve_utils.time, "sleep", lambda seconds: None)
    assert helper("http://test", {}, max_retries=2) == {"ok": True}
    assert post.call_count == 2
