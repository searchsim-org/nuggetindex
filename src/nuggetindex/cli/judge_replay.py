"""``nuggetindex judge-replay`` -- summarize an LLM-judge log file."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def judge_replay_command(
    log: Path = typer.Argument(
        ...,
        help="Path to a judge log (one JSON object per line).",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        help="Number of most-recent rationales to print.",
    ),
) -> None:
    """Print a decision histogram + the last ``--limit`` rationales."""
    if not log.exists():
        typer.echo(f"Error: judge log not found: {log}", err=True)
        raise typer.Exit(code=2)

    rows: list[dict[str, object]] = []
    with log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed rows rather than aborting; logs are
                # append-only and partial writes are possible.
                continue

    counts = Counter(str(r.get("decision", "UNKNOWN")) for r in rows)

    table = Table(title=f"judge-replay -- {log} ({len(rows)} entries)")
    table.add_column("decision", style="cyan")
    table.add_column("count", justify="right")
    for decision, count in counts.most_common():
        table.add_row(decision, str(count))
    console.print(table)

    if rows and limit > 0:
        console.print(f"\n[bold]Last {min(limit, len(rows))} rationales:[/bold]")
        for r in rows[-limit:]:
            console.print(
                f"- [cyan]{r.get('decision', '?')}[/cyan] "
                f"({r.get('a_object', '?')} vs {r.get('b_object', '?')}): "
                f"{r.get('rationale', '')}"
            )
