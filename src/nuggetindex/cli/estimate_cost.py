"""``nuggetindex estimate-cost`` -- preview ingest cost before committing.

Reads a JSONL corpus (same shape as ``nuggetindex doctor --index-path``:
each line a JSON object with at least ``source_id`` and ``text``) and
prints a cost / wall-time estimate produced by
:func:`nuggetindex.audit.cost.estimate_ingest_cost`.

Usage::

    nuggetindex estimate-cost --index docs.jsonl --model gpt-4o-mini
    nuggetindex estimate-cost --index docs.jsonl --cache .cache.db

The command never runs an extractor and never touches the LLM. All it
does is sample the corpus, count tokens, multiply by the model's posted
price, and (optionally) probe the cache file for hit-rate.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from nuggetindex.pipeline.constructor import Document


def _parse_source_date(raw: Any) -> datetime | None:
    """Best-effort ISO-8601 parser for the JSONL ``source_date`` field."""
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


def _load_jsonl_docs(path: Path) -> list[Document]:
    """Parse each non-blank line of ``path`` into a :class:`Document`."""
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                typer.echo(
                    f"estimate-cost: skipping {path.name}:{lineno} -- "
                    f"invalid JSON: {exc}",
                    err=True,
                )
                continue
            source_id = row.get("source_id")
            text = row.get("text")
            if not source_id or not text:
                typer.echo(
                    f"estimate-cost: skipping {path.name}:{lineno} -- "
                    "missing required 'source_id' or 'text'",
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
    return docs


def estimate_cost_command(
    index: Path = typer.Option(
        ...,
        "--index",
        help=(
            "Path to a .jsonl file: one document per line with "
            "'source_id' and 'text' fields (same shape as doctor)."
        ),
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        help=(
            "Model id used for pricing. 'trigger' yields $0. Unknown "
            "models use a conservative default with a warning."
        ),
    ),
    cache: Path | None = typer.Option(
        None,
        "--cache",
        help=(
            "Optional path to the extractor cache. When set, the probe "
            "reports an expected cache hit-rate and net cost."
        ),
    ),
    sample_size: int = typer.Option(
        100,
        "--sample-size",
        help="Number of documents to sample for the estimate.",
    ),
) -> None:
    """Print a cost / wall-time estimate for the configured ingest."""
    if not index.exists():
        typer.echo(f"estimate-cost: --index not found: {index}", err=True)
        raise typer.Exit(code=1)
    if index.suffix.lower() not in {".jsonl", ".ndjson"}:
        typer.echo(
            f"estimate-cost: --index must be .jsonl / .ndjson "
            f"(got {index.suffix!r})",
            err=True,
        )
        raise typer.Exit(code=1)

    docs = _load_jsonl_docs(index)

    from nuggetindex.audit.cost import estimate_ingest_cost

    estimate = asyncio.run(
        estimate_ingest_cost(
            docs=docs,
            sample_size=sample_size,
            model_id=model,
            cache_path=cache,
        )
    )
    typer.echo(estimate.rendered_markdown)


__all__ = ["estimate_cost_command"]
