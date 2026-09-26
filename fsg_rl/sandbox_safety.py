"""Static safety checks used before non-Docker Python execution."""

from __future__ import annotations

import ast


ALLOWED_IMPORTS = {
    "collections",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "statistics",
}
FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "quit",
    "requests",
    "setattr",
    "socket",
    "subprocess",
    "type",
    "urllib",
    "vars",
}


class SandboxSafetyError(ValueError):
    """Raised when generated Python violates the static sandbox policy."""


def validate_python_code(code: str, *, label: str = "generated code") -> str:
    code = code.strip()
    if not code:
        raise SandboxSafetyError(f"{label} is empty")
    if len(code) > 20_000:
        raise SandboxSafetyError(f"{label} is too long")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxSafetyError(f"{label} has invalid Python: {exc}") from exc
    _validate_ast(tree, label)
    return code


def validate_python_expression(expression: str, *, label: str = "test expression") -> str:
    expression = expression.strip()
    if not expression or len(expression) > 4000:
        raise SandboxSafetyError(f"{label} is empty or too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SandboxSafetyError(f"{label} is invalid Python expression: {exc}") from exc
    _validate_ast(tree, label)
    return expression


def _validate_ast(tree: ast.AST, label: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise SandboxSafetyError(
                        f"{label} imports forbidden module {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in ALLOWED_IMPORTS:
                raise SandboxSafetyError(
                    f"{label} imports forbidden module {module!r}"
                )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxSafetyError(f"{label} uses forbidden name {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxSafetyError(f"{label} uses forbidden dunder attribute")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "__" in node.value
        ):
            raise SandboxSafetyError(f"{label} contains forbidden dunder string")
