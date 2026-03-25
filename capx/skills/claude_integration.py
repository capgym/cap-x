"""Helpers for formatting the skill library for Claude Code prompts."""

from __future__ import annotations

from capx.skills.library import SkillLibrary


def format_skills_for_prompt(library: SkillLibrary) -> str:
    """Format promoted skills as API documentation suitable for inclusion in prompts.

    Returns a Markdown-formatted string describing all promoted skills with
    their signatures, docstrings, and source information.
    """
    promoted = {n: s for n, s in library.skills.items() if s.promoted}
    if not promoted:
        return ""

    lines = [
        "# Reusable Skills (from previous successful trials)\n",
        "The following helper functions have been extracted from successful trials "
        "and are available for use. You can call them directly or adapt them.\n",
    ]
    for name, skill in sorted(promoted.items()):
        lines.append(f"### `{name}`")
        if skill.docstring:
            lines.append(f"{skill.docstring}\n")
        lines.append(f"Used in {skill.occurrences} successful trial(s).")
        lines.append(f"Source tasks: {', '.join(skill.source_tasks) or 'N/A'}\n")
        lines.append(f"```python\n{skill.code}\n```\n")

    return "\n".join(lines)


def format_skills_as_python(library: SkillLibrary) -> str:
    """Format promoted skills as importable Python source code.

    Returns a string that can be written to a ``.py`` file or ``exec()``-ed
    to make all promoted skills available in a namespace.
    """
    promoted = {n: s for n, s in library.skills.items() if s.promoted}
    if not promoted:
        return "# No promoted skills available.\n"

    lines = [
        '"""Auto-generated skill library from CaP-X successful trials."""\n',
    ]
    for name, skill in sorted(promoted.items()):
        lines.append(f"# Skill: {name} (occurrences: {skill.occurrences})")
        lines.append(skill.code)
        lines.append("")

    return "\n".join(lines)
