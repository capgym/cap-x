from __future__ import annotations

import ast
from dataclasses import dataclass


_FORBIDDEN_NAMES = {
    "APIS",
    "ORACLE_CODE",
    "__import__",
    "compile",
    "env",
    "eval",
    "exec",
    "expert_script",
    "getattr",
    "globals",
    "ground_truth",
    "human_oracle_code",
    "inspect",
    "locals",
    "open",
    "oracle_code",
    "play_once",
    "vars",
}

_FORBIDDEN_IMPORT_ROOTS = {
    "importlib",
    "inspect",
    "os",
    "pathlib",
    "pickle",
    "subprocess",
    "sys",
}


@dataclass(frozen=True)
class ProgramGuardResult:
    allowed: bool
    reason: str = ""


class _GuardVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violation = ""

    def _reject(self, reason: str) -> None:
        if not self.violation:
            self.violation = reason

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self._reject(f"forbidden name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _FORBIDDEN_NAMES or node.attr.startswith("_"):
            self._reject(f"forbidden attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", maxsplit=1)[0]
            if root in _FORBIDDEN_IMPORT_ROOTS:
                self._reject(f"forbidden import: {root}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", maxsplit=1)[0]
        if root in _FORBIDDEN_IMPORT_ROOTS:
            self._reject(f"forbidden import: {root}")
        self.generic_visit(node)


def validate_generated_program(source: str) -> ProgramGuardResult:
    """Reject programs that bypass the documented robot APIs or access oracle state."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ProgramGuardResult(True)

    visitor = _GuardVisitor()
    visitor.visit(tree)
    if visitor.violation:
        return ProgramGuardResult(False, visitor.violation)
    return ProgramGuardResult(True)
