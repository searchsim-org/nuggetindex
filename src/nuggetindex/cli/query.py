"""``nuggetindex query`` -- hybrid retrieval over an existing store."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


async def _run_query(
    text: str,
    db: Path,
    qt: datetime | None,
    top_k: int,
    view: str,
    fusion: str,
) -> list[object]:
    from nuggetindex.store import NuggetStore

    store = NuggetStore(db_path=db)
    try:
        results = await store.aretrieve(
            text,
            query_time=qt,
            top_k=top_k,
            view=view,
            fusion=fusion,
        )
    finally:
        await store.aclose()
    return results


def query_command(
    text: str = typer.Argument(..., help="Free-text query."),
    db: Path = typer.Option(
        Path("nuggetindex.db"),
        "--db",
        help="Path to the NuggetStore SQLite file.",
    ),
    time: str | None = typer.Option(
        None,
        "--time",
        help="ISO-8601 query time; defaults to now(UTC).",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Number of results to return."),
    view: str = typer.Option(
        "active",
        "--view",
        help="Lifecycle view: active | active_contested | all.",
    ),
    fusion: str = typer.Option(
        "rrf",
        "--fusion",
        help="Fusion mode: rrf | weighted_minmax.",
    ),
) -> None:
    """Run a hybrid query against ``db`` and print a Rich table of hits."""
    if not db.exists():
        typer.echo(f"Error: database file not found: {db}", err=True)
        raise typer.Exit(code=2)

    qt: datetime | None = None
    if time is not None:
        qt = datetime.fromisoformat(time)
        if qt.tzinfo is None:
            qt = qt.replace(tzinfo=UTC)

    results = asyncio.run(_run_query(text, db, qt, top_k, view, fusion))

    table = Table(title=f"query: {text}", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("status", style="cyan")
    table.add_column("valid_from")
    table.add_column("valid_until")
    table.add_column("subject")
    table.add_column("predicate")
    table.add_column("object")
    table.add_column("source")

    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    for r in results:
        n = r.nugget  # type: ignore[attr-defined]
        sources = ",".join(p.source_id for p in n.provenance) or "-"
        table.add_row(
            str(r.rank),  # type: ignore[attr-defined]
            str(n.epistemic.status),
            n.validity.start.isoformat(),
            n.validity.end.isoformat() if n.validity.end else "open",
            n.fact.subject,
            n.fact.predicate,
            n.fact.object,
            sources,
        )
    console.print(table)
