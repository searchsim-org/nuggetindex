"""``nuggetindex diff`` -- compare two NuggetStore databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _read_id_status(db: Path) -> dict[str, str]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT id, status FROM nuggets").fetchall()
    finally:
        conn.close()
    return {r[0]: r[1] for r in rows}


def diff_command(
    old: Path = typer.Option(..., "--old", help="Previous DB snapshot."),
    new: Path = typer.Option(..., "--new", help="Current DB snapshot."),
) -> None:
    """Summarize the diff between two store snapshots."""
    if not old.exists():
        typer.echo(f"Error: old database not found: {old}", err=True)
        raise typer.Exit(code=2)
    if not new.exists():
        typer.echo(f"Error: new database not found: {new}", err=True)
        raise typer.Exit(code=2)

    old_map = _read_id_status(old)
    new_map = _read_id_status(new)

    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    changed: list[tuple[str, str, str]] = []
    for nid in sorted(set(old_map) & set(new_map)):
        if old_map[nid] != new_map[nid]:
            changed.append((nid, old_map[nid], new_map[nid]))

    console.print(
        f"[bold]added:[/bold] {len(added)}    "
        f"[bold]removed:[/bold] {len(removed)}    "
        f"[bold]status-changed:[/bold] {len(changed)}"
    )

    if changed:
        table = Table(title="Status changes")
        table.add_column("id", style="cyan")
        table.add_column("old status")
        table.add_column("new status")
        for nid, old_s, new_s in changed:
            table.add_row(nid, old_s, new_s)
        console.print(table)
