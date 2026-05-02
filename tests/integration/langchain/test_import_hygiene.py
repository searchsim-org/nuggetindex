"""Import-hygiene invariant: LangChain adapter only uses public top-level API.

Allowing the adapter to reach into ``nuggetindex.core.models`` or
``nuggetindex.store.backends`` would couple third-party glue to internal
implementation details and defeat the point of maintaining a stable public
surface. This test AST-parses each ``.py`` under
``src/nuggetindex/integrations/langchain`` and asserts no ``ImportFrom``
hits any forbidden internal namespace.

Allowed: ``from nuggetindex import NuggetStore`` (top-level re-exports) and
``from nuggetindex.governance import GovernancePostProcessor`` (a public
subpackage the governance module explicitly declares stable).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")


FORBIDDEN_PREFIXES = (
    "nuggetindex.core.models",
    "nuggetindex.core.enums",
    "nuggetindex.core.schema",
    "nuggetindex.core.errors",
    "nuggetindex.store.backends",
    "nuggetindex.store.base",
    "nuggetindex.pipeline.constructor",
    "nuggetindex.pipeline.conflict",
    "nuggetindex.pipeline.dedup",
    "nuggetindex.pipeline.temporal",
    "nuggetindex.pipeline.canonicalize",
    "nuggetindex.extractors.base",
    "nuggetindex.extractors.llm",
    "nuggetindex.extractors.quality",
    "nuggetindex.extractors.rule_based",
    "nuggetindex.extractors.clients",
    "nuggetindex.retrieve.retriever",
    "nuggetindex.retrieve.fusion",
    "nuggetindex.audit.api",
    "nuggetindex.utils",
)


def _pkg_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent.parent
        / "src"
        / "nuggetindex"
        / "integrations"
        / "langchain"
    )


def test_pkg_dir_resolves() -> None:
    assert _pkg_dir().is_dir(), f"expected {_pkg_dir()} to exist"


def test_no_forbidden_internal_imports() -> None:
    violations: list[str] = []
    for py in sorted(_pkg_dir().glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for bad in FORBIDDEN_PREFIXES:
                    if mod == bad or mod.startswith(bad + "."):
                        violations.append(f"{py.name}: from {mod} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in FORBIDDEN_PREFIXES:
                        if alias.name == bad or alias.name.startswith(bad + "."):
                            violations.append(f"{py.name}: import {alias.name}")
    assert not violations, (
        "integrations/langchain must only import from top-level nuggetindex.* "
        "public API; got forbidden imports:\n  " + "\n  ".join(violations)
    )


def test_all_modules_present() -> None:
    """Sanity: the three feature modules exist under the package dir."""
    names = {p.name for p in _pkg_dir().glob("*.py")}
    assert {"__init__.py", "retriever.py", "postprocessor.py", "loader.py"}.issubset(names)
