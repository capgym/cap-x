"""CaP-X: Code-as-action Policy eXecution for V(L)A agents in simulators.

This package provides:
- A Gymnasium environment that accepts code (string) as actions
- A local sandbox runner for executing code with a curated tool API
- Stubs for tool APIs: IK (PyRoKI adapter), vision, motion planning, BC tail
- Rollout utilities to interface with VeRL/AgentGym-RL
- Tyro CLIs and example GRPO config
"""

from __future__ import annotations

__all__ = [
    "envs",
    "serving",
    "integrations",
    "utils",
]

__version__ = "0.1.0"
