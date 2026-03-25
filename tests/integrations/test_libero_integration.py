from __future__ import annotations

import sys
import types

from capx.integrations import libero as lib_mod


class FakeEnv:
    def __init__(self) -> None:
        self._t = 0

    def reset(self, seed=None, options=None):  # noqa: D401, ARG002
        self._t = 0
        return {"image": None}, {}

    def seed(self, s: int) -> None:  # noqa: D401, ARG002
        pass

    def set_init_state(self, state):  # noqa: D401, ARG002
        pass

    def step(self, action):  # noqa: D401, ARG002
        self._t += 1
        done = self._t >= 2
        return {"image": None}, 1.0 if done else 0.0, done, {"t": self._t}


def test_load_libero_task(monkeypatch: object) -> None:
    class FakeSuite:
        def get_task(self, task_id: int):  # noqa: D401
            return types.SimpleNamespace(
                problem_folder="pf", bddl_file="file.bddl", language="lang"
            )

        def get_task_init_states(self, task_id: int):  # noqa: D401
            return [None]

    def get_benchmark_dict():  # noqa: D401
        return {"libero_10": lambda: FakeSuite()}

    # Define FakeOffEnv before injection
    class FakeOffEnv(FakeEnv):
        def __init__(self, **kwargs):  # noqa: D401, ARG002
            super().__init__()

    # Create fake package/module hierarchy for libero
    libero_pkg = types.ModuleType("libero")
    libero_libero = types.ModuleType("libero.libero")
    envs_mod = types.ModuleType("libero.libero.envs")
    utils_mod = types.ModuleType("libero.libero.utils")
    # Attach attributes
    libero_libero.benchmark = types.SimpleNamespace(get_benchmark_dict=get_benchmark_dict)  # type: ignore[attr-defined]
    envs_mod.OffScreenRenderEnv = FakeOffEnv  # type: ignore[attr-defined]
    utils_mod.get_libero_path = lambda name: "/x"  # type: ignore[attr-defined]
    # Register in sys.modules
    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_mod)
    monkeypatch.setitem(sys.modules, "libero.libero.utils", utils_mod)

    # No need to patch attributes on lib_mod; load_libero_task imports from the injected modules above

    handle = lib_mod.load_libero_task("libero_10", task_id=0)
    obs, info = handle.reset(seed=0)
    assert isinstance(obs, dict)
    obs, rew, done, info = handle.step([0.0] * 7)
    assert done in (True, False)
