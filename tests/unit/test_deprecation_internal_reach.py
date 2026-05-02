"""Ensure no module under ``src/nuggetindex`` reaches into the deprecated
``NuggetStore._backend`` attribute.

``store/base.py`` is the only legitimate location, and only for the property
definition itself — but because the property name is ``_backend`` and its
body references ``self._backend_impl`` (not ``self._backend``), we expect
zero offenders even without the allow-list. The allow-list remains so the
guard stays correct if anyone adds references inside ``base.py`` for the
deprecation shim (e.g. inside ``__getattr__`` style fallbacks).
"""
from __future__ import annotations

import ast
from pathlib import Path


def test_no_private_backend_access_from_other_modules():
    pkg_root = Path(__file__).parent.parent.parent / "src" / "nuggetindex"
    allowed = {"store/base.py"}
    offenders: list[str] = []
    for py in pkg_root.rglob("*.py"):
        rel = py.relative_to(pkg_root).as_posix()
        if rel in allowed:
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "_backend"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in ("_store", "store")
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"Forbidden ._backend reach: {offenders}"
