"""``nuggetindex schema discover`` -- corpus-driven predicate discovery CLI.

Thin CLI wrapper around :func:`nuggetindex.audit.discover.discover_schema`.
Reads the same JSONL document format as ``nuggetindex doctor`` (one
``{source_id, text, uri?, source_date?}`` record per line) and prints /
writes the returned :class:`SchemaProposal`.

Output behaviour:

* ``--out`` (optional) -- write the proposal YAML to this file. When
  omitted, the YAML is printed to stdout.
* ``--report`` (optional) -- write the Markdown report to this file.
  When omitted, the Markdown report is printed to stderr.

The ``--extractor`` flag accepts ``trigger`` (default; zero LLM cost) or
an OpenAI-compatible model id (e.g. ``gpt-4o-mini``). LLM extractors are
lazy-imported so running the default ``trigger`` path never touches the
``openai`` package.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from nuggetindex.audit.discover import SchemaProposal, discover_schema
from nuggetindex.pipeline.constructor import Document

schema_app = typer.Typer(
    help="Corpus-driven predicate discovery for offline nuggetindex construction.",
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

    Lenient parser: bad lines are skipped with a stderr warning,
    ``source_id`` + ``text`` are required.
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
                    f"schema: skipping {path.name}:{lineno} -- invalid JSON: {exc}",
                    err=True,
                )
                continue
            source_id = row.get("source_id")
            text = row.get("text")
            if not source_id or not text:
                typer.echo(
                    f"schema: skipping {path.name}:{lineno} -- missing required "
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
            f"schema: loaded {len(docs)} document(s) from {path} (skipped {total - len(docs)}).",
            err=True,
        )
    return docs


def _resolve_extractor(spec: str) -> Any | None:
    """Map the ``--extractor`` flag onto an extractor instance.

    ``trigger`` (the default) returns ``None`` so
    :func:`discover_schema` picks its own :class:`TriggerExtractor`.
    Anything else is interpreted as an OpenAI-compatible model id.
    """
    normalised = spec.strip().lower()
    if normalised in {"", "trigger", "triggers"}:
        return None

    # LLM path: lazy-import so the ``trigger`` path never touches openai.
    try:
        from nuggetindex.extractors.clients.base import LLMConfig, build_client
        from nuggetindex.extractors.llm import LLMExtractor
    except ImportError as exc:  # pragma: no cover -- defensive
        raise typer.BadParameter(
            f"--extractor {spec!r} requires the LLM extras; "
            f"install nuggetindex[llm] or use 'trigger'. (ImportError: {exc})",
            param_hint="--extractor",
        ) from exc

    config = LLMConfig(model=spec)
    client = build_client(config)
    return LLMExtractor(client=client)


@schema_app.command("discover")
def discover_command(
    index_path: Path = typer.Option(
        ...,
        "--index",
        help=("Path to a .jsonl file (one document per line with 'source_id'/'text')."),
    ),
    sample_size: int = typer.Option(
        500,
        "--sample-size",
        help="Target number of documents to draw for extraction.",
    ),
    extractor_spec: str = typer.Option(
        "trigger",
        "--extractor",
        help=(
            "Extractor backend: 'trigger' (default; zero LLM cost) or an "
            "OpenAI-compatible model id (e.g. 'gpt-4o-mini')."
        ),
    ),
    min_frequency: int = typer.Option(
        3,
        "--min-frequency",
        help="Predicates seen fewer than this many times are dropped as noise.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=("Write the YAML proposal to this file. If omitted the YAML is printed to stdout."),
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help=("Optional Markdown report destination. If omitted the report is printed to stderr."),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Emit load counts on stderr.",
    ),
) -> None:
    """Discover predicates in a corpus and propose a schema."""
    if not index_path.exists():
        typer.echo(f"schema: --index not found: {index_path}", err=True)
        raise typer.Exit(code=1)

    suffix = index_path.suffix.lower()
    if suffix not in {".jsonl", ".ndjson"}:
        typer.echo(
            f"schema: unsupported --index extension {index_path.suffix!r}; "
            "expected .jsonl or .ndjson.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        extractor = _resolve_extractor(extractor_spec)
    except typer.BadParameter as exc:
        typer.echo(f"schema: {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    docs = _load_jsonl_docs(index_path, verbose=verbose)

    proposal: SchemaProposal = asyncio.run(
        discover_schema(
            docs=docs,
            sample_size=sample_size,
            extractor=extractor,
            min_frequency=min_frequency,
        )
    )

    # YAML -> --out or stdout.
    if out is not None:
        out.write_text(proposal.rendered_yaml, encoding="utf-8")
        if verbose:
            typer.echo(f"schema: wrote YAML proposal to {out}", err=True)
    else:
        typer.echo(proposal.rendered_yaml)

    # Markdown report -> --report or stderr.
    if report is not None:
        report.write_text(proposal.rendered_markdown, encoding="utf-8")
        if verbose:
            typer.echo(f"schema: wrote Markdown report to {report}", err=True)
    else:
        typer.echo(proposal.rendered_markdown, err=True)


__all__ = ["discover_command", "schema_app"]
