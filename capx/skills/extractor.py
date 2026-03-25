"""Function extraction utilities for the evolving skill library.

Uses the same regex patterns as scripts/eval_analysis/parse_outputs.py
to extract top-level function definitions from trial code.
"""

from __future__ import annotations

import re

# Regex to extract top-level function definitions (captures full function with body).
# Matches 'def func_name(args):' followed by an indented body.
# Only captures top-level functions (lines starting at column 0).
FUNCTION_DEF_PATTERN = re.compile(
    r'^(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*\n(?:(?:[ \t]+.+\n?)+))',
    re.MULTILINE,
)

# Pattern to extract function name and signature.
FUNCTION_SIG_PATTERN = re.compile(
    r'^def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:',
    re.MULTILINE,
)

# Pattern to extract a docstring (triple-quoted string) from the start of a function body.
DOCSTRING_PATTERN = re.compile(
    r'^\s+(?:\"\"\"([\s\S]*?)\"\"\"|\'\'\'([\s\S]*?)\'\'\')',
    re.MULTILINE,
)


def extract_docstring(func_code: str) -> str:
    """Extract the docstring from a function definition.

    Looks for a triple-quoted string immediately after the ``def`` line.
    Returns an empty string if no docstring is found.
    """
    # Skip the first line (the def line) and look for a docstring
    lines = func_code.split("\n", 1)
    if len(lines) < 2:
        return ""
    body = lines[1]
    match = DOCSTRING_PATTERN.match(body)
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    return ""


def extract_functions(code: str) -> list[dict]:
    """Extract all top-level function definitions from code.

    Returns a list of dicts, each with keys:
        - ``name``: The function name.
        - ``signature``: The full signature line (e.g. ``def foo(x, y) -> int``).
        - ``code``: The full function source code.
        - ``docstring``: The extracted docstring, or empty string.
    """
    functions: list[dict] = []

    for match in FUNCTION_DEF_PATTERN.finditer(code):
        full_def = match.group(1).rstrip()

        sig_match = FUNCTION_SIG_PATTERN.match(full_def)
        if sig_match:
            name = sig_match.group(1)
            params = sig_match.group(2).strip()
            return_type = sig_match.group(3).strip() if sig_match.group(3) else None

            signature = f"def {name}({params})"
            if return_type:
                signature += f" -> {return_type}"

            docstring = extract_docstring(full_def)

            functions.append({
                "name": name,
                "signature": signature,
                "code": full_def,
                "docstring": docstring,
            })

    return functions
