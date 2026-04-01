"""Core evolving skill library that persists across trials.

Skills are extracted from successful trial code, tracked by frequency,
and promoted to the active library when they appear in multiple tasks.

The library supports two modes:
  1. **Online evolution** — skills are extracted per-trial during evaluation
     when ``evolve_skill_library: true`` is set in the YAML config.
  2. **Batch curation** — ``scripts/eval_analysis/compile_skill_library.py``
     aggregates functions across experiments and uses an LLM to curate
     the final library.  See :meth:`SkillLibrary.curate_with_llm`.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from capx.skills.extractor import extract_functions

# Patterns that indicate task-specific functions (should not be promoted).
# Matches the filtering in scripts/eval_analysis/compile_skill_library.py.
TASK_SPECIFIC_PATTERNS = [
    r"cube", r"stack", r"lift", r"wipe", r"spill",
    r"place_.*_on", r"pick_.*_up", r"grab_the", r"move_to_goal",
]

# LLM prompt used by compile_skill_library.py and curate_with_llm().
# Kept here as the canonical version so both online and batch paths
# can reference it.
SKILL_LIBRARY_PROMPT = """\
You are an expert robotics software engineer curating a reusable skill library.

Below are function definitions extracted from successful robot manipulation \
code generations across multiple tasks and models.
These functions were composed by LLM coding agents to solve various \
manipulation tasks using a reduced/low-level robotics API.

Your task is to:
1. Identify the MOST USEFUL and REUSABLE functions that could form a \
general-purpose skill library
2. Group similar functions into categories (e.g., perception, motion, \
grasping, coordinate transforms)
3. For each category, select the BEST implementation(s) — prefer \
well-documented, general-purpose versions
4. Exclude task-specific or overly narrow functions
5. Note any functions that appear frequently — this indicates high utility

Output format:
- Organize by category
- For each selected function, explain WHY it's useful and reusable
- Include the full function code, along with proper Python docstring and \
type hints, return types, etc.
- Note how many times similar functions appeared (popularity)

Focus on functions that would be valuable additions to a robotics \
manipulation toolkit.

---
EXTRACTED FUNCTIONS (with occurrence counts and sources):
{functions}
"""


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

    @staticmethod
    def _is_task_specific(name: str) -> bool:
        """Check if a function name matches task-specific patterns."""
        return any(re.search(p, name, re.IGNORECASE) for p in TASK_SPECIFIC_PATTERNS)

    @staticmethod
    def _is_trivial(code: str, min_lines: int = 3) -> bool:
        """Check if a function is too short to be useful."""
        return len(code.strip().splitlines()) < min_lines

    def get_promoted_skills(self, min_occurrences: int = 2) -> dict[str, str]:
        """Return skills that qualify for promotion (frequently occurring).

        Auto-promotes skills meeting *min_occurrences* that are not
        task-specific or trivial, and returns a mapping of
        ``{name: code}`` for all promoted skills.
        """
        for skill in self.skills.values():
            if (
                skill.occurrences >= min_occurrences
                and not self._is_task_specific(skill.name)
                and not self._is_trivial(skill.code)
            ):
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

    # ------------------------------------------------------------------
    # LLM curation (batch mode)
    # ------------------------------------------------------------------

    def format_for_prompt(self) -> str:
        """Format all skills for the LLM curation prompt.

        Mirrors ``format_functions_for_prompt()`` from the original
        ``scripts/eval_analysis/compile_skill_library.py``.
        """
        lines: list[str] = []
        for name in sorted(self.skills):
            skill = self.skills[name]
            lines.append(f"### {name}")
            lines.append(f"Occurrences: {skill.occurrences}")
            lines.append(f"Source tasks: {', '.join(skill.source_tasks)}")
            lines.append(f"```python\n{skill.code}\n```")
            lines.append("")
        return "\n".join(lines)

    def curate_with_llm(
        self,
        server_url: str = "http://127.0.0.1:8110/chat/completions",
        model: str = "google/gemini-3.1-pro-preview",
    ) -> str:
        """Call an LLM to curate the skill library.

        Uses :data:`SKILL_LIBRARY_PROMPT` — the same prompt from the
        original ``compile_skill_library.py`` — to organize skills by
        category, select best implementations, and explain reusability.

        Returns the raw LLM response text (curated library).
        """
        import requests

        prompt = SKILL_LIBRARY_PROMPT.format(functions=self.format_for_prompt())
        resp = requests.post(
            server_url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0.2,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

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
