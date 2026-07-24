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

_ALLOWED_IMPORT_ROOTS = {"numpy", "scipy"}

_DOCUMENTED_API_NAMES = {
    "close_gripper",
    "get_active_nut_types",
    "get_object_pose",
    "goto_home_joint_position",
    "goto_home_joint_positions",
    "goto_pose",
    "goto_pose_arm0",
    "goto_pose_arm1",
    "open_gripper",
    "compose_pose",
    "relative_pose",
    "sample_grasp_pose",
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
            if root not in _ALLOWED_IMPORT_ROOTS:
                self._reject(f"forbidden import: {root}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", maxsplit=1)[0]
        if root not in _ALLOWED_IMPORT_ROOTS:
            self._reject(f"forbidden import: {root}")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._reject("try/except is forbidden in generated robot programs")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in _DOCUMENTED_API_NAMES:
            self._reject(f"cannot redefine documented API: {node.name}")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name in _DOCUMENTED_API_NAMES:
            self._reject(f"cannot redefine documented API: {node.name}")
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
