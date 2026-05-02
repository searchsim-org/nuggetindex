"""``nuggetindex seeds propose`` -- automated seed-query proposer.

Thin CLI wrapper around :func:`nuggetindex.audit.seeds.propose_seeds`. Reads
the same JSONL document format as ``nuggetindex doctor`` (one ``{source_id,
text, uri?, source_date?}`` record per line) and prints / writes the
returned :class:`SeedProposal`.

Output formats are selected by the ``--out`` file extension:

* ``.md`` / ``.markdown`` -- :attr:`SeedProposal.rendered_markdown`.
* ``.json``               -- full dataclass serialisation.

When ``--out`` is omitted the Markdown report is written to stdout.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from nuggetindex.audit.seeds import SeedProposal, propose_seeds
from nuggetindex.pipeline.constructor import Document

seeds_app = typer.Typer(
    help="Automated seed-query proposer for offline nuggetindex construction.",
    no_args_is_help=True,
    add_completion=False,
)


def _parse_source_date(raw: Any) -> datetime | None:
    """Best-effort ISO-8601 parser (mirrors ``nuggetindex doctor``)."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _load_jsonl_docs(path: Path, *, verbose: bool) -> list[Document]:
    """Load :class:`Document` records from a JSONL file.

    Same lenient parsing as the doctor CLI: bad lines are skipped with a
    stderr warning, ``source_id`` + ``text`` are required.
    """
    docs: list[Document] = []
    total = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                typer.echo(
                    f"seeds: skipping {path.name}:{lineno} -- invalid JSON: {exc}",
                    err=True,
                )
                continue
            source_id = row.get("source_id")
            text = row.get("text")
            if not source_id or not text:
                typer.echo(
                    f"seeds: skipping {path.name}:{lineno} -- missing required "
                    "'source_id' or 'text'",
                    err=True,
                )
                continue
            docs.append(
                Document(
                    source_id=str(source_id),
                    text=str(text),
                    uri=row.get("uri"),
                    source_date=_parse_source_date(row.get("source_date")),
                )
            )
    if verbose:
        typer.echo(
            f"seeds: loaded {len(docs)} document(s) from {path} (skipped {total - len(docs)}).",
            err=True,
        )
    return docs


def _write_proposal(proposal: SeedProposal, out: Path) -> None:
    """Dispatch on ``out``'s extension: ``.md`` / ``.markdown`` / ``.json``."""
    ext = out.suffix.lower()
    if ext in {".md", ".markdown"}:
        out.write_text(proposal.rendered_markdown, encoding="utf-8")
        return
    if ext == ".json":
        out.write_text(
            json.dumps(dataclasses.asdict(proposal), indent=2, default=str),
            encoding="utf-8",
        )
        return
    raise typer.BadParameter(
        f"--out must end in .md, .markdown, or .json (got {out.suffix!r})",
        param_hint="--out",
    )


@seeds_app.command("propose")
def propose_command(
    index_path: Path = typer.Option(
        ...,
        "--index-path",
        help=("Path to a .jsonl file (one document per line with 'source_id'/'text')."),
    ),
    budget: int = typer.Option(
        50,
        "--budget",
        help="Maximum number of seed queries to return.",
    ),
    sample_size: int = typer.Option(
        500,
        "--sample-size",
        help="Target number of documents to draw for entity ranking.",
    ),
    stratify_by: str = typer.Option(
        "composite",
        "--stratify-by",
        help=(
            "Sampler stratification: 'composite' (default; language x domain),"
            " 'source_date', 'domain', 'language', or 'none'."
        ),
    ),
    min_entity_frequency: int = typer.Option(
        3,
        "--min-entity-frequency",
        help="Minimum observed count for an entity to qualify for ranking.",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help=(
            "Optional sentence-transformers model id. Default: TF-IDF "
            "character n-grams (no extra deps)."
        ),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Write the proposal to this file (.md or .json). If omitted the "
            "Markdown report is printed to stdout."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Emit load counts on stderr.",
    ),
) -> None:
    """Propose a budgeted, diversity-optimised set of seed queries."""
    if stratify_by not in {"source_date", "none", "domain", "language", "composite"}:
        typer.echo(
            f"seeds: --stratify-by must be one of 'source_date', 'none', "
            f"'domain', 'language', 'composite' (got {stratify_by!r}).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not index_path.exists():
        typer.echo(f"seeds: --index-path not found: {index_path}", err=True)
        raise typer.Exit(code=1)

    suffix = index_path.suffix.lower()
    if suffix not in {".jsonl", ".ndjson"}:
        typer.echo(
            f"seeds: unsupported --index-path extension {index_path.suffix!r}; "
            "expected .jsonl or .ndjson.",
            err=True,
        )
        raise typer.Exit(code=1)

    docs = _load_jsonl_docs(index_path, verbose=verbose)

    proposal: SeedProposal = asyncio.run(
        propose_seeds(
            docs=docs,
            budget=budget,
            sample_size=sample_size,
            stratify_by=stratify_by,  # type: ignore[arg-type]
            min_entity_frequency=min_entity_frequency,
            embedding_model=embedding_model,
        )
    )

    if out is not None:
        try:
            _write_proposal(proposal, out)
        except typer.BadParameter as exc:
            typer.echo(f"seeds: {exc.message}", err=True)
            raise typer.Exit(code=1) from exc
        if verbose:
            typer.echo(f"seeds: wrote proposal to {out}", err=True)
    else:
        typer.echo(proposal.rendered_markdown)


__all__ = ["propose_command", "seeds_app"]
