"""Evolving skill library for CaP-X.

Automatically extracts reusable functions from successful trial code,
tracks usage frequency, and promotes popular skills for injection into
future trials.
"""

from capx.skills.library import Skill, SkillLibrary

__all__ = ["Skill", "SkillLibrary"]
