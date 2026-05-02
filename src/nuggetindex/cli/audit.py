"""``nuggetindex audit`` -- zero-index audit subcommand.

Reads a plain-text context file (one passage per paragraph, separated by
blank lines), runs :func:`nuggetindex.audit` against it, and renders the
resulting :class:`~nuggetindex.audit.api.AuditReport` in the user-selected
format.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from nuggetindex.audit.api import audit as _audit

console = Console()


def audit_command(
    query: str = typer.Option(..., "--query", help="User question being audited."),
    context: Path = typer.Option(
        ...,
        "--context",
        help="Text file with one passage per paragraph (blank-line separated).",
    ),
    time: str | None = typer.Option(
        None,
        "--time",
        help="ISO-8601 query time; defaults to now(UTC).",
    ),
    fmt: str = typer.Option(
        "console",
        "--format",
        help="Output format: console | json | markdown.",
    ),
    model: str = typer.Option(
        "trigger",
        "--model",
        help=(
            'Extractor: "trigger" (default; LLM-free) or an LLM model id '
            'such as "gpt-4o-mini". The legacy "rule_based" alias is '
            "accepted with a deprecation warning."
        ),
    ),
) -> None:
    """Zero-index audit: extract nuggets from passages and report conflicts / staleness."""
    import warnings

    if model == "rule_based":
        warnings.warn(
            "--model rule_based is deprecated: delegating to "
            "TriggerExtractor ('trigger'). Pass --model trigger to silence "
            "this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        model = "trigger"
    if not context.exists():
        typer.echo(f"Error: context file not found: {context}", err=True)
        raise typer.Exit(code=2)

    query_time = datetime.fromisoformat(time) if time else datetime.now(UTC)
    if query_time.tzinfo is None:
        query_time = query_time.replace(tzinfo=UTC)

    text = context.read_text(encoding="utf-8")
    passages = [p.strip() for p in text.split("\n\n") if p.strip()]

    report = asyncio.run(
        _audit(
            query=query,
            passages=passages,
            query_time=query_time,
            extractor=model,
        )
    )

    fmt_lower = fmt.lower()
    if fmt_lower == "json":
        typer.echo(report.to_json())
    elif fmt_lower in ("markdown", "md"):
        typer.echo(report.to_markdown())
    else:
        console.print(report.to_rich_console())
