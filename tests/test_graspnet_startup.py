"""Checkpoint validation must finish before the GraspNet service starts."""

import pickle
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from capx.serving import launch_contact_graspnet_server as server


@pytest.fixture
def startup(monkeypatch, tmp_path):
    model = torch.nn.Linear(2, 1)
    estimator_module = ModuleType("contact_graspnet_pytorch.contact_grasp_estimator")
    estimator_module.GraspEstimator = Mock(return_value=Mock(model=model))
    monkeypatch.setitem(sys.modules, estimator_module.__name__, estimator_module)
    monkeypatch.setattr(server, "load_contact_graspnet_config", lambda path: {})
    monkeypatch.setattr(server, "_GRASP_ESTIMATOR", None)
    # Keep the pinned checkpoint loader available when testing the unfixed launcher.
    vendor = Path(server.__file__).parents[1] / "third_party/contact_graspnet_pytorch"
    monkeypatch.syspath_prepend(str(vendor))
    monkeypatch.setattr(server, "__file__", str(tmp_path / "capx/serving/server.py"))
    checkpoint_dir = (
        tmp_path
        / "capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints"
    )
    checkpoint_dir.mkdir(parents=True)
    serve = Mock()
    monkeypatch.setattr(server.uvicorn, "run", serve)
    return model, checkpoint_dir / "model.pt", serve


@pytest.mark.parametrize(
    "kind,error",
    [
        ("missing", FileNotFoundError),
        ("corrupt", pickle.UnpicklingError),
        ("missing_model", KeyError),
        ("wrong_keys", RuntimeError),
        ("wrong_shape", RuntimeError),
    ],
)
def test_invalid_checkpoint_prevents_serving(startup, kind, error):
    model, path, serve = startup
    if kind == "corrupt":
        path.write_bytes(b"not a checkpoint")
    elif kind == "missing_model":
        torch.save({"epoch": 1}, path)
    elif kind == "wrong_keys":
        torch.save({"model": {"unrelated": torch.ones(1)}}, path)
    elif kind == "wrong_shape":
        state = model.state_dict()
        state["weight"] = torch.ones(1, 3)
        torch.save({"model": state}, path)

    with pytest.raises(error):
        server.main(device="cpu")
    serve.assert_not_called()


def test_valid_checkpoint_is_loaded_before_serving(startup):
    model, path, serve = startup
    expected = {name: torch.full_like(value, 7) for name, value in model.state_dict().items()}
    torch.save({"model": expected, "epoch": 1}, path)

    def check_loaded(*args, **kwargs):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name])

    serve.side_effect = check_loaded
    server.main(device="cpu", host="127.0.0.1", port=8115)
    serve.assert_called_once_with(server.app, host="127.0.0.1", port=8115)
