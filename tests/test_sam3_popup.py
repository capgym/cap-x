from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np


def test_sam3_mask_popup_blocks_until_keypress_and_closes_window(monkeypatch) -> None:
    from capx.utils.sam3_popup import Sam3MaskPopupDisplay

    events = []

    fake_cv2 = SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=16,
        COLOR_RGB2BGR=1,
        getTextSize=lambda label, font, scale, thickness: ((20, 8), 2),
        putText=lambda *args, **kwargs: events.append(("putText",)),
        cvtColor=lambda image, code: image,
        imshow=lambda name, image: events.append(("imshow", name, image.shape)),
        waitKey=lambda delay: events.append(("waitKey", delay)) or 65,
        destroyWindow=lambda name: events.append(("destroyWindow", name)),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")

    display = Sam3MaskPopupDisplay(window_name="test-sam3")

    assert display.show(np.zeros((4, 4, 3), dtype=np.uint8), title="mask", wait_ms=0) is True
    assert ("waitKey", 0) in events
    assert ("destroyWindow", "test-sam3") in events


def test_sam3_mask_popup_skips_without_linux_gui(monkeypatch, capsys) -> None:
    from capx.utils.sam3_popup import Sam3MaskPopupDisplay

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    display = Sam3MaskPopupDisplay()

    assert display.show(np.zeros((4, 4, 3), dtype=np.uint8), title="mask", wait_ms=0) is False
    assert "skipping popup" in capsys.readouterr().out
