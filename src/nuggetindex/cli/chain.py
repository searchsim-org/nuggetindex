"""``nuggetindex chain`` -- walk a temporal provenance chain in a store.

Dispatches on ``--type``:

* ``succession`` -- :meth:`NuggetStore.achain_succession`
* ``rename`` -- :meth:`NuggetStore.achain_rename`
* ``join`` -- :meth:`NuggetStore.achain_join`

All three render a Rich timeline with columns ``idx``, ``status``,
``valid_from``, ``valid_until``, ``object``, ``evidence``. ``--format json``
emits the full chain (including edges) as JSON for scripting.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _parse_iso(ts: str | None) -> datetime | None:
    if ts is None:
        return None
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


async def _run_chain(
    *,
    chain_type: str,
    db: Path,
    subject: str | None,
    predicate: str | None,
    direction: str,
    start_subject: str | None,
    start_predicate: str | None,
    then: list[str],
    as_of: datetime | None,
    max_depth: int,
    include_contested: bool,
) -> Any:
    from nuggetindex.store import NuggetStore

    store = NuggetStore(db_path=db)
    try:
        if chain_type == "succession":
            if subject is None or predicate is None:
                raise typer.BadParameter(
                    "--subject and --predicate are required for type=succession"
                )
            return await store.achain_succession(
                subject=subject,
                predicate=predicate,
                as_of=as_of,
                include_contested=include_contested,
                max_depth=max_depth,
            )
        if chain_type == "rename":
            if subject is None:
                raise typer.BadParameter(
                    "--subject is required for type=rename"
                )
            return await store.achain_rename(
                subject=subject,
                as_of=as_of,
                direction=direction,  # type: ignore[arg-type]
                max_depth=max_depth,
                include_contested=include_contested,
            )
        if chain_type == "join":
            if start_subject is None or start_predicate is None:
                raise typer.BadParameter(
                    "--start-subject and --start-predicate required for type=join"
                )
            return await store.achain_join(
                start=(start_subject, start_predicate),
                then=then,
                as_of=as_of,
            )
        raise typer.BadParameter(f"unknown --type: {chain_type}")
    finally:
        await store.aclose()


async def _fetch_candidate_keys(
    *,
    db: Path,
    subject_contains: str | None,
    predicate_contains: str | None,
    limit: int = 20,
) -> list[tuple[str, str, str]]:
    from nuggetindex.store import NuggetStore

    store = NuggetStore(db_path=db)
    try:
        # If the user typed an alias for the predicate, canonicalise it so
        # the substring search hits the stored canonical form too. E.g.
        # typing ``ceo`` should surface keys stored under
        # ``chiefExecutiveOfficer``.
        if predicate_contains is not None:
            predicate_contains = store.schema.canonicalize(predicate_contains)
        return await store.acandidate_keys(
            subject_contains=subject_contains,
            predicate_contains=predicate_contains,
            limit=limit,
        )
    finally:
        await store.aclose()


def _render_discover_table(
    keys: list[tuple[str, str, str]],
    *,
    subject: str | None,
    predicate: str | None,
) -> Table:
    title_bits = ["Candidate keys"]
    if subject:
        title_bits.append(f"subject~'{subject}'")
    if predicate:
        title_bits.append(f"predicate~'{predicate}'")
    table = Table(title=" ".join(title_bits), show_lines=False)
    table.add_column("subject")
    table.add_column("predicate")
    table.add_column("scope")
    for subj, pred, scope in keys:
        table.add_row(subj, pred, scope)
    return table


def _render_table(chain: Any, title: str) -> Table:
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("status", style="cyan")
    table.add_column("valid_from")
    table.add_column("valid_until")
    table.add_column("subject")
    table.add_column("predicate")
    table.add_column("object")
    table.add_column("evidence")
    for i, n in enumerate(chain.nuggets):
        sources = ",".join(p.source_id for p in n.provenance) or "-"
        table.add_row(
            str(i),
            str(n.epistemic.status),
            n.validity.start.isoformat(),
            n.validity.end.isoformat() if n.validity.end else "open",
            n.fact.subject,
            n.fact.predicate,
            n.fact.object,
            sources,
        )
    return table


def _chain_to_json(chain: Any) -> str:
    payload = {
        "chain_type": chain.chain_type,
        "as_of": chain.as_of.isoformat() if chain.as_of else None,
        "truncated": chain.truncated,
        "nuggets": [json.loads(n.model_dump_json()) for n in chain.nuggets],
        "edges": [
            {
                "from_idx": e.from_idx,
                "to_idx": e.to_idx,
                "edge_type": str(e.edge_type),
                "gap_seconds": e.gap.total_seconds() if e.gap else None,
            }
            for e in chain.edges
        ],
    }
    return json.dumps(payload, indent=2)


def chain_command(
    type_: str = typer.Option(
        ...,
        "--type",
        help="Chain type: succession | rename | join.",
    ),
    subject: str | None = typer.Option(
        None, "--subject", help="Subject (succession + rename).",
    ),
    predicate: str | None = typer.Option(
        None, "--predicate", help="Predicate (succession).",
    ),
    direction: str = typer.Option(
        "forward",
        "--direction",
        help="Rename walk direction: forward | backward | both.",
    ),
    start_subject: str | None = typer.Option(
        None, "--start-subject", help="Join: subject of the first hop.",
    ),
    start_predicate: str | None = typer.Option(
        None, "--start-predicate", help="Join: predicate of the first hop.",
    ),
    then: str | None = typer.Option(
        None,
        "--then",
        help="Join: comma-separated predicates for subsequent hops.",
    ),
    as_of: str | None = typer.Option(
        None, "--as-of", help="ISO-8601 temporal cutoff.",
    ),
    max_depth: int = typer.Option(
        50, "--max-depth", help="Max chain length before truncation.",
    ),
    include_contested: bool = typer.Option(
        False, "--include-contested", help="Include CONTESTED nuggets.",
    ),
    db: Path = typer.Option(
        Path("nuggetindex.db"),
        "--db",
        help="Path to the NuggetStore SQLite file.",
    ),
    format_: str = typer.Option(
        "console", "--format", help="Output format: console | json.",
    ),
    discover: bool = typer.Option(
        False,
        "--discover/--no-discover",
        help=(
            "When the exact-match chain is empty, print a table of "
            "nearby (subject, predicate, scope) triples found via "
            "case-insensitive substring search."
        ),
    ),
) -> None:
    """Walk a temporal provenance chain in a NuggetStore."""
    if not db.exists():
        typer.echo(f"Error: database file not found: {db}", err=True)
        raise typer.Exit(code=2)

    ct = type_.lower()
    if ct not in ("succession", "rename", "join"):
        raise typer.BadParameter(
            f"--type must be succession | rename | join (got {type_!r})"
        )
    dir_ = direction.lower()
    if dir_ not in ("forward", "backward", "both"):
        raise typer.BadParameter(
            f"--direction must be forward | backward | both (got {direction!r})"
        )

    as_of_dt = _parse_iso(as_of)
    then_list = (
        [p.strip() for p in then.split(",") if p.strip()] if then else []
    )

    chain = asyncio.run(
        _run_chain(
            chain_type=ct,
            db=db,
            subject=subject,
            predicate=predicate,
            direction=dir_,
            start_subject=start_subject,
            start_predicate=start_predicate,
            then=then_list,
            as_of=as_of_dt,
            max_depth=max_depth,
            include_contested=include_contested,
        )
    )

    if format_ == "json":
        typer.echo(_chain_to_json(chain))
        return

    title_bits = [f"chain type={ct}"]
    if subject:
        title_bits.append(f"subject={subject}")
    if predicate:
        title_bits.append(f"predicate={predicate}")
    if ct == "join":
        title_bits.append(f"start={start_subject}.{start_predicate}")
        if then_list:
            title_bits.append(f"then={','.join(then_list)}")
    if as_of:
        title_bits.append(f"as_of={as_of}")
    title = " ".join(title_bits)

    if not chain.nuggets:
        console.print("[yellow]No results.[/yellow]")
        # Identify the lookup key used for the hint / discovery query.
        if ct == "succession":
            key_subject = subject
            key_predicate = predicate
        elif ct == "join":
            key_subject = start_subject
            key_predicate = start_predicate
        else:  # rename
            key_subject = subject
            key_predicate = None
        if discover:
            keys = asyncio.run(
                _fetch_candidate_keys(
                    db=db,
                    subject_contains=key_subject,
                    predicate_contains=key_predicate,
                    limit=20,
                )
            )
            if keys:
                console.print(
                    _render_discover_table(
                        keys,
                        subject=key_subject,
                        predicate=key_predicate,
                    )
                )
            else:
                console.print(
                    "[yellow]No candidate keys matched the "
                    "substring filters.[/yellow]"
                )
        else:
            hint_target = (
                f"({key_subject!r}, {key_predicate!r})"
                if key_predicate is not None
                else f"({key_subject!r})"
            )
            console.print(
                f"[dim]No chain found for {hint_target}. "
                "Run again with --discover to see nearby keys.[/dim]"
            )
        return
    console.print(_render_table(chain, title))
    if chain.truncated:
        console.print(
            f"[yellow]Chain truncated at max_depth={max_depth}.[/yellow]"
        )
