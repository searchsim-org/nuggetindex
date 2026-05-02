"""``nuggetindex resolve`` -- human-in-the-loop adjudication of contested facts.

Walks every ``(subject, predicate, scope)`` key with at least one
``Contested`` member in the target store, renders the rival objects
side-by-side with their evidence spans + sources + validity intervals,
and lets the user pick a winner. The winner is pinned via
:meth:`NuggetStore.amark_preferred`; the losers are recorded as
suppressed via :meth:`NuggetStore.asuppress`. Provenance is preserved
on both paths.

Decisions are logged to ``~/.nuggetindex/resolve_log.jsonl`` so the
adjudication trail is reproducible and auditable.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from nuggetindex.core.models import Nugget

console = Console()


def _resolve_log_path() -> Path:
    home = Path(os.environ.get("NUGGETINDEX_HOME", str(Path.home() / ".nuggetindex")))
    home.mkdir(parents=True, exist_ok=True)
    return home / "resolve_log.jsonl"


def _append_log(entry: dict[str, object]) -> None:
    entry = {**entry, "timestamp": datetime.now(UTC).isoformat()}
    with _resolve_log_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _render_card(idx: int, n: "Nugget") -> Panel:
    body = (
        f"[bold]{n.fact.object}[/bold]\n"
        f"valid: {n.validity.start.date()} → "
        f"{n.validity.end.date() if n.validity.end else 'present'}\n"
        f"status: {n.epistemic.status.value}  rank: {n.epistemic.rank.value}"
    )
    if n.provenance:
        p = n.provenance[0]
        body += f"\nsource: {p.source_id}"
        if p.evidence_span:
            ev = p.evidence_span.strip().replace("\n", " ")
            if len(ev) > 160:
                ev = ev[:157] + "..."
            body += f'\nevidence: "{ev}"'
    return Panel(body, title=f"[{idx}] id={n.id[:12]}", border_style="cyan")


def _prompt_choice(n_candidates: int) -> str:
    """Return one of: '1', '2', ..., 'skip', 'leave', 'all-wrong'."""
    options = ", ".join(str(i + 1) for i in range(n_candidates))
    while True:
        raw = typer.prompt(
            f"Pick a winner [{options}], or 'skip' / 'leave' / 'all-wrong'",
            default="leave",
        ).strip().lower()
        if raw in {"skip", "leave", "all-wrong"}:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= n_candidates:
            return raw
        console.print(
            f"[yellow]Invalid choice {raw!r}. Enter 1..{n_candidates}, "
            "skip, leave, or all-wrong.[/yellow]"
        )


async def _resolve_one_key(
    store, subject: str, predicate: str, scope: str, *, dry_run: bool
) -> str:
    """Adjudicate a single contested key. Returns the action taken
    (``"winner-N"``, ``"skip"``, ``"leave"``, ``"all-wrong"``)."""
    nuggets = await store._backend_impl.afind_by_key((subject, predicate, scope))
    # Show the current Contested members + Active members so the user has
    # full context for the adjudication.
    nuggets = sorted(nuggets, key=lambda n: n.validity.start)
    if not nuggets:
        return "empty"
    console.print(
        f"\n[bold magenta]{subject} / {predicate}[/bold magenta] "
        f"(scope={scope}, {len(nuggets)} member"
        f"{'s' if len(nuggets) != 1 else ''})"
    )
    for i, n in enumerate(nuggets, start=1):
        console.print(_render_card(i, n))
    if dry_run:
        console.print("[dim](dry-run; no changes applied)[/dim]")
        return "dry-run"
    choice = _prompt_choice(len(nuggets))
    if choice in {"skip", "leave"}:
        _append_log(
            {
                "action": choice,
                "subject": subject,
                "predicate": predicate,
                "scope": scope,
                "candidate_ids": [n.id for n in nuggets],
            }
        )
        return choice
    if choice == "all-wrong":
        suppressed: list[str] = []
        for n in nuggets:
            await store.asuppress(n.id)
            suppressed.append(n.id)
        _append_log(
            {
                "action": "all-wrong",
                "subject": subject,
                "predicate": predicate,
                "scope": scope,
                "suppressed": suppressed,
            }
        )
        console.print(f"[yellow]Suppressed all {len(suppressed)} candidates.[/yellow]")
        return "all-wrong"
    winner_idx = int(choice) - 1
    winner = nuggets[winner_idx]
    losers = [n for n in nuggets if n.id != winner.id]
    await store.amark_preferred(winner.id)
    for loser in losers:
        await store.asuppress(loser.id)
    _append_log(
        {
            "action": "winner",
            "subject": subject,
            "predicate": predicate,
            "scope": scope,
            "winner": winner.id,
            "winner_object": winner.fact.object,
            "suppressed": [n.id for n in losers],
        }
    )
    console.print(
        f"[green]Pinned[/green] {winner.fact.object!r} as Active+Preferred; "
        f"suppressed {len(losers)} loser"
        f"{'s' if len(losers) != 1 else ''}."
    )
    return f"winner-{winner.id[:12]}"


async def _amain(
    store_path: Path, key: str | None, dry_run: bool, only_first: int | None
) -> None:
    from nuggetindex.store import NuggetStore

    store = NuggetStore(db_path=store_path)
    try:
        if key:
            parts = key.split("/")
            if len(parts) == 2:
                subject, predicate = parts
                scope = "global"
            elif len(parts) == 3:
                subject, predicate, scope = parts
            else:
                raise typer.BadParameter(
                    "--key must be 'subject/predicate' or "
                    "'subject/predicate/scope'"
                )
            await _resolve_one_key(
                store, subject, predicate, scope, dry_run=dry_run
            )
            return

        keys = await store.acontested_keys()
        if not keys:
            console.print(
                "[green]No contested keys in the store. Nothing to resolve."
                "[/green]"
            )
            return

        table = Table(title=f"Contested keys ({len(keys)} total)")
        table.add_column("#", style="dim", justify="right")
        table.add_column("Subject")
        table.add_column("Predicate")
        table.add_column("Scope")
        table.add_column("Members", justify="right")
        for i, (s, p, sc, n) in enumerate(keys, start=1):
            table.add_row(str(i), s, p, sc, str(n))
        console.print(table)

        keys_to_walk = keys if only_first is None else keys[:only_first]
        for s, p, sc, _n in keys_to_walk:
            await _resolve_one_key(store, s, p, sc, dry_run=dry_run)
    finally:
        await store._backend_impl.aclose()


def resolve_command(
    store_path: Path = typer.Option(
        ...,
        "--store",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the NuggetStore SQLite file to adjudicate.",
    ),
    key: str | None = typer.Option(
        None,
        "--key",
        help=(
            "Jump straight to one key, formatted "
            "'subject/predicate' or 'subject/predicate/scope'. "
            "Without this, the CLI walks every contested key."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the contested cards without prompting for or applying any changes.",
    ),
    only_first: int | None = typer.Option(
        None,
        "--only-first",
        min=1,
        help="Stop after adjudicating the first N keys (handy for piloting).",
    ),
) -> None:
    """Walk contested facts; pin a winner; suppress the losers."""
    asyncio.run(_amain(store_path, key, dry_run, only_first))
