"""Shared helpers for the ``reproduce.rqN`` scripts.

Kept here so each RQ script can stay declarative — a table of rows, a
printout, a load attempt — without each one re-implementing the scaffolding.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any


def print_banner(rq: str, title: str, paper_ref: str) -> None:
    """Print a consistent 'about to run RQn' banner to stdout."""
    bar = "=" * 72
    print(bar)
    print(f"nuggetindex reproduce — {rq}: {title}")
    print(f"Paper reference: {paper_ref}")
    print(bar)


def print_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Emit a Github-flavoured Markdown table to stdout."""
    print("| " + " | ".join(str(h) for h in headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


def try_load_dataset(name: str, *, source_url: str, license_note: str) -> Any | None:
    """Attempt to load ``name`` from the local ``datasets`` cache.

    Returns the dataset on success. On ``ImportError`` (datasets not
    installed) or ``FileNotFoundError`` / network failure we print a clear
    actionable message and return ``None`` — the caller decides whether to
    exit or to emit synthetic rows.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[skip] `datasets` not installed. Run: pip install nuggetindex[eval]",
            file=sys.stderr,
        )
        return None

    try:
        return load_dataset(name)
    except Exception as e:  # noqa: BLE001 — offline/missing is expected
        print(f"[skip] could not load dataset '{name}': {e}", file=sys.stderr)
        print(f"        source:  {source_url}", file=sys.stderr)
        print(f"        licence: {license_note}", file=sys.stderr)
        return None
