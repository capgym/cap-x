from __future__ import annotations

from typing import Any

import numpy as np


class RoboSuiteWrapper:
    """Stub wrapper for robosuite environment.

    Replace with actual robosuite task construction and success checks.
    """

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self._success = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        np.random.default_rng(seed)
        self._success = False
        obs = {
            "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
            "depth": np.zeros((256, 256), dtype=np.float32),
            "lang_goal": f"Task: {self.task_name}",
        }
        return obs, {}

    def step_sim(self) -> None:
        pass

    def success(self) -> bool:
        return self._success

    def set_success(self, value: bool) -> None:
        self._success = value
