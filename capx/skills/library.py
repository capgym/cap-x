"""Core evolving skill library that persists across trials.

Skills are extracted from successful trial code, tracked by frequency,
and promoted to the active library when they appear in multiple tasks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from capx.skills.extractor import extract_functions


@dataclass
class Skill:
    """A single reusable skill."""

    name: str
    code: str  # Full function source code
    docstring: str  # Extracted docstring
    occurrences: int  # How many successful trials used this
    source_tasks: list[str]  # Which tasks it was extracted from
    promoted: bool  # Whether it's been promoted to the active library


class SkillLibrary:
    """Evolving skill library that persists across trials.

    Skills are extracted from successful trial code, tracked by frequency,
    and promoted to the active library when they appear in multiple tasks.
    """

    DEFAULT_PATH = Path(".capx_skills.json")

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.skills: dict[str, Skill] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load skills from disk."""
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for name, info in data.get("skills", {}).items():
                self.skills[name] = Skill(**info)

    def save(self) -> None:
        """Persist skills to disk."""
        data = {
            "skills": {name: asdict(skill) for name, skill in self.skills.items()}
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_from_code(self, code: str, task_name: str = "") -> list[str]:
        """Extract function definitions from trial code and update library.

        Returns list of newly discovered function names.
        """
        functions = extract_functions(code)
        new_names: list[str] = []

        for func in functions:
            name = func["name"]
            if name in self.skills:
                # Update existing skill
                skill = self.skills[name]
                skill.occurrences += 1
                if task_name and task_name not in skill.source_tasks:
                    skill.source_tasks.append(task_name)
                # Update code to the latest version
                skill.code = func["code"]
                if func["docstring"]:
                    skill.docstring = func["docstring"]
            else:
                # New skill
                self.skills[name] = Skill(
                    name=name,
                    code=func["code"],
                    docstring=func["docstring"],
                    occurrences=1,
                    source_tasks=[task_name] if task_name else [],
                    promoted=False,
                )
                new_names.append(name)

        return new_names

    # ------------------------------------------------------------------
    # Promotion & querying
    # ------------------------------------------------------------------

    def get_promoted_skills(self, min_occurrences: int = 2) -> dict[str, str]:
        """Return skills that qualify for promotion (frequently occurring).

        Auto-promotes skills meeting *min_occurrences* and returns a mapping
        of ``{name: code}`` for all promoted skills.
        """
        # Auto-promote based on occurrence threshold
        for skill in self.skills.values():
            if skill.occurrences >= min_occurrences:
                skill.promoted = True

        return {
            name: skill.code
            for name, skill in self.skills.items()
            if skill.promoted
        }

    def get_skill_docs(self) -> str:
        """Return formatted documentation of promoted skills for prompts."""
        promoted = {n: s for n, s in self.skills.items() if s.promoted}
        if not promoted:
            return "No promoted skills available."

        lines = [
            "# Available Skill Library Functions",
            f"({len(promoted)} promoted skills)\n",
        ]
        for name, skill in sorted(promoted.items()):
            lines.append(f"## {name}")
            if skill.docstring:
                lines.append(f"  {skill.docstring}")
            lines.append(f"  Occurrences: {skill.occurrences}")
            lines.append(f"  Tasks: {', '.join(skill.source_tasks)}")
            lines.append(f"\n```python\n{skill.code}\n```\n")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Namespace injection
    # ------------------------------------------------------------------

    def inject_into_namespace(self, namespace: dict[str, Any]) -> None:
        """Execute promoted skills and inject them into a code execution namespace."""
        promoted = self.get_promoted_skills()
        for name, code in promoted.items():
            try:
                exec(code, namespace)  # noqa: S102
            except Exception as exc:
                print(f"[SkillLibrary] Failed to inject skill '{name}': {exc}")

    # ------------------------------------------------------------------
    # Manual management
    # ------------------------------------------------------------------

    def add_skill(
        self, name: str, code: str, docstring: str = "", source_task: str = ""
    ) -> None:
        """Manually add or update a skill."""
        if name in self.skills:
            skill = self.skills[name]
            skill.code = code
            skill.occurrences += 1
            if docstring:
                skill.docstring = docstring
            if source_task and source_task not in skill.source_tasks:
                skill.source_tasks.append(source_task)
        else:
            self.skills[name] = Skill(
                name=name,
                code=code,
                docstring=docstring,
                occurrences=1,
                source_tasks=[source_task] if source_task else [],
                promoted=False,
            )

    def remove_skill(self, name: str) -> None:
        """Remove a skill from the library."""
        self.skills.pop(name, None)

    def promote(self, name: str) -> None:
        """Promote a skill to the active library."""
        if name in self.skills:
            self.skills[name].promoted = True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the library."""
        total = len(self.skills)
        promoted = sum(1 for s in self.skills.values() if s.promoted)
        if total == 0:
            return "Skill library is empty."

        lines = [
            f"Skill Library: {total} skills ({promoted} promoted)",
            "-" * 50,
        ]
        for name in sorted(self.skills):
            skill = self.skills[name]
            status = "[promoted]" if skill.promoted else ""
            lines.append(
                f"  {name}: {skill.occurrences} occurrences, "
                f"{len(skill.source_tasks)} tasks {status}"
            )
        return "\n".join(lines)
