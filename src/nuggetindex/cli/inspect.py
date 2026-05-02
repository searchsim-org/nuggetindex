"""``nuggetindex inspect`` -- dump summary stats about a store."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nuggetindex.core.enums import LifecycleStatus

console = Console()


async def _collect_stats(db: Path) -> dict[str, object]:
    from nuggetindex.store import NuggetStore

    store = NuggetStore(db_path=db)
    try:
        total = await store.acount()
        active = await store.acount(status=LifecycleStatus.ACTIVE)
        deprecated = await store.acount(status=LifecycleStatus.DEPRECATED)
        contested = await store.acount(status=LifecycleStatus.CONTESTED)
    finally:
        await store.aclose()

    conn = sqlite3.connect(db)
    try:
        unique_keys = conn.execute("SELECT COUNT(DISTINCT key) FROM nuggets").fetchone()[0]
        top_preds = conn.execute(
            "SELECT predicate, COUNT(*) AS c FROM nuggets "
            "GROUP BY predicate ORDER BY c DESC LIMIT 10"
        ).fetchall()
        temporal = conn.execute(
            "SELECT MIN(validity_start), MAX(validity_start) FROM nuggets"
        ).fetchone()
    finally:
        conn.close()

    return {
        "total": total,
        "active": active,
        "deprecated": deprecated,
        "contested": contested,
        "unique_keys": unique_keys,
        "top_predicates": top_preds,
        "earliest": temporal[0],
        "latest": temporal[1],
    }


def inspect_command(
    db: Path = typer.Option(
        Path("nuggetindex.db"),
        "--db",
        help="Path to the NuggetStore SQLite file.",
    ),
) -> None:
    """Print a summary panel + top-predicates table for ``db``."""
    if not db.exists():
        typer.echo(f"Error: database file not found: {db}", err=True)
        raise typer.Exit(code=2)

    stats = asyncio.run(_collect_stats(db))

    summary = Panel.fit(
        f"[bold]Total:[/bold] {stats['total']}    "
        f"[bold]Active:[/bold] {stats['active']}    "
        f"[bold]Contested:[/bold] {stats['contested']}    "
        f"[bold]Deprecated:[/bold] {stats['deprecated']}\n"
        f"[bold]Unique keys:[/bold] {stats['unique_keys']}\n"
        f"[bold]Temporal span:[/bold] {stats['earliest']} .. {stats['latest']}",
        title=f"nuggetindex inspect -- {db}",
    )
    console.print(summary)

    table = Table(title="Top predicates")
    table.add_column("predicate", style="cyan")
    table.add_column("count", justify="right")
    top_preds: list[tuple[str, int]] = stats["top_predicates"]  # type: ignore[assignment]
    for row in top_preds:
        table.add_row(str(row[0]), str(row[1]))
    console.print(table)
