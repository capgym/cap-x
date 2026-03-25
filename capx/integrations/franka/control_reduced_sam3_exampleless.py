"""Exampleless variant of FrankaControlApiReducedSam3.

Functionally identical to FrankaControlApiReducedSam3 but strips usage
examples from docstrings exposed via ``combined_doc()``.
"""

import inspect
from typing import Any

from capx.envs.base import BaseEnv
from capx.integrations.franka.control_reduced_sam3 import FrankaControlApiReducedSam3


def _strip_examples(doc: str | None) -> str:
    """Remove 'Example:' / 'Examples:' sections from a Google-style docstring."""
    if not doc:
        return ""
    lines = doc.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("example:") or stripped.startswith("examples:"):
            skip = True
            continue
        if skip:
            # Keep skipping indented continuation lines that belong to the example block
            if line.strip() == "" or (len(line) > 0 and line[0] in (" ", "\t") and not stripped.startswith("args:") and not stripped.startswith("returns:")):
                # Check if this looks like a new section header
                if stripped.endswith(":") and not stripped.startswith(">>>"):
                    skip = False
                    out.append(line)
                continue
            else:
                skip = False
        out.append(line)
    return "\n".join(out)


class FrankaControlApiReducedSam3Exampleless(FrankaControlApiReducedSam3):
    """FrankaControlApiReducedSam3 without usage examples in docstrings."""

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] | None = [0.0, 0.0, -0.107],
        is_spill_wipe: bool = False,
        is_peg_assembly: bool = False,
        is_handover: bool = False,
        bimanual: bool = False,
    ) -> None:
        super().__init__(
            env,
            tcp_offset=tcp_offset,
            is_spill_wipe=is_spill_wipe,
            is_peg_assembly=is_peg_assembly,
            is_handover=is_handover,
            bimanual=bimanual,
        )

    def combined_doc(self) -> str:
        """Aggregate function docs with examples stripped."""
        lines: list[str] = []
        for name, fn in self.functions().items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(...)"
            doc = _strip_examples(inspect.getdoc(fn))
            lines.append(f"{name}{sig}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
        return "\n".join(lines).strip()
