from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


def _install_sam3_stubs() -> None:
    """Allow importing the SAM3 server in environments without sam3 installed."""
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.bfloat16 = object()

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def set_device(_device_idx: int) -> None:
                return None

        torch_mod.cuda = _Cuda()
        torch_mod.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
            cudnn=types.SimpleNamespace(allow_tf32=False),
        )
        sys.modules["torch"] = torch_mod

    if "sam3.model_builder" in sys.modules:
        return

    sam3_pkg = types.ModuleType("sam3")
    sam3_pkg.__path__ = []  # type: ignore[attr-defined]
    model_pkg = types.ModuleType("sam3.model")
    model_pkg.__path__ = []  # type: ignore[attr-defined]

    processor_mod = types.ModuleType("sam3.model.sam3_image_processor")

    class Sam3Processor:  # noqa: N801 - mirrors imported class name
        pass

    processor_mod.Sam3Processor = Sam3Processor

    builder_mod = types.ModuleType("sam3.model_builder")

    def build_sam3_image_model(**_kwargs):
        raise AssertionError("test should patch build_sam3_image_model")

    builder_mod.build_sam3_image_model = build_sam3_image_model

    sys.modules.setdefault("sam3", sam3_pkg)
    sys.modules.setdefault("sam3.model", model_pkg)
    sys.modules.setdefault("sam3.model.sam3_image_processor", processor_mod)
    sys.modules.setdefault("sam3.model_builder", builder_mod)


class Sam3ServerCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        _install_sam3_stubs()
        self.server = importlib.import_module("capx.serving.launch_sam3_server")

    def test_main_passes_local_checkpoint_and_disables_hf_download(self) -> None:
        calls: list[dict[str, object]] = []

        class DummyModel:
            def to(self, device: str) -> "DummyModel":
                self.device = device
                return self

        model = DummyModel()

        def fake_build(**kwargs):
            calls.append(kwargs)
            return model

        with (
            mock.patch.object(self.server, "build_sam3_image_model", side_effect=fake_build),
            mock.patch.object(self.server.Sam3Processor, "__new__", return_value="processor"),
            mock.patch.object(self.server.torch.cuda, "is_available", return_value=False),
            mock.patch.object(self.server.uvicorn, "run") as run_mock,
        ):
            self.server.main(
                device="cpu",
                port=8114,
                host="127.0.0.1",
                checkpoint_path="capx/model_weights/sam3/sam3.pt",
            )

        self.assertEqual(
            calls,
            [
                {
                    "checkpoint_path": "capx/model_weights/sam3/sam3.pt",
                    "load_from_HF": False,
                    "enable_inst_interactivity": True,
                }
            ],
        )
        run_mock.assert_called_once_with(self.server.app, host="127.0.0.1", port=8114)

    def test_main_keeps_hf_download_when_checkpoint_is_not_set(self) -> None:
        calls: list[dict[str, object]] = []

        class DummyModel:
            pass

        def fake_build(**kwargs):
            calls.append(kwargs)
            return DummyModel()

        with (
            mock.patch.object(self.server, "build_sam3_image_model", side_effect=fake_build),
            mock.patch.object(self.server.Sam3Processor, "__new__", return_value="processor"),
            mock.patch.object(self.server.torch.cuda, "is_available", return_value=False),
            mock.patch.object(self.server.uvicorn, "run"),
        ):
            self.server.main(device="cpu", port=8114, host="127.0.0.1")

        self.assertEqual(
            calls,
            [
                {
                    "checkpoint_path": None,
                    "load_from_HF": True,
                    "enable_inst_interactivity": True,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
