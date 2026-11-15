"""Report Python classes/functions that lack explanatory docstrings.

This lightweight quality check helps keep the repository suitable for classroom
walkthroughs as new code is added. It does not score prose quality; reviewers
still need to confirm that a docstring explains purpose, parameters, and choices.
"""
from __future__ import annotations

import ast
from pathlib import Path


def find_missing_docstrings(root: Path) -> list[str]:
    """Return ``path:line:symbol`` entries for undocumented definitions.

    Args:
        root: Repository root containing app, remote_agents, scripts, and tests.
    """
    files = []
    for folder in ["app", "remote_agents", "scripts", "tests"]:
        files.extend((root / folder).rglob("*.py"))
    missing = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node):
                missing.append(f"{path.relative_to(root)}:{node.lineno}:{node.name}")
    return missing


def main() -> None:
    """Print the audit and return a failing exit code when gaps remain."""
    root = Path(__file__).resolve().parents[1]
    missing = find_missing_docstrings(root)
    if missing:
        print("Definitions without docstrings:")
        print("\n".join(missing))
        raise SystemExit(1)
    print("Documentation audit passed: every class and function has a docstring.")


if __name__ == "__main__":
    main()
