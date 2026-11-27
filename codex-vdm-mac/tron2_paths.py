"""Resolve the TRON2 checkout for the bundled and sibling layouts."""

from pathlib import Path


TRON2_REPO = Path(__file__).resolve().parent.parent
if not (TRON2_REPO / "scripts" / "run_tron2_model.py").is_file():
    TRON2_REPO /= "tron2-sim"
