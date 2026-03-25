from __future__ import annotations

from typing import Any

from capx.integrations.libero import LiberoHandle, load_libero_task


class LiberoWrapper:
    """Stub wrapper for LIBERO environment.

    Replace internals with real LIBERO env creation and success checks.
    """

    def __init__(self, suite_name: str, task_id: int, cam_w: int = 128, cam_h: int = 128) -> None:
        self.handle: LiberoHandle = load_libero_task(
            suite_name=suite_name, task_id=task_id, cam_w=cam_w, cam_h=cam_h
        )
        self._success = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._success = False
        obs, info = self.handle.reset(seed=seed)
        return obs, info

    def step_action(
        self, action: list[float]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        obs, reward, done, info = self.handle.step(action)
        self._success = self._success or done
        return obs, reward, done, info

    def success(self) -> bool:
        return self._success

    def set_success(self, value: bool) -> None:
        self._success = value
