"""``nuggetindex build`` / ``nuggetindex ingest`` -- create or grow a store.

``build`` scans a folder (recursively) for ``.txt`` / ``.md`` files,
instantiates a :class:`~nuggetindex.store.NuggetStore`, and ingests each
file as a :class:`~nuggetindex.pipeline.constructor.Document`.

``ingest`` is the append-only variant (same callback, ``--append`` set).

The default extractor is :class:`~nuggetindex.extractors.TriggerExtractor`
(LLM-free, zero setup). Pass ``--model gpt-4o-mini`` (or any other
OpenAI-compatible id, ``claude-*`` for Anthropic, ``gemini-*`` for Google)
to route through :class:`~nuggetindex.extractors.LLMExtractor` instead.

The legacy ``rule_based`` sentinel is retained for script compatibility:
it emits a :class:`DeprecationWarning` and delegates to ``trigger``.
"""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from pathlib import Path

import typer
from rich.console import Console

console = Console()

_TEXT_SUFFIXES = {".txt", ".md"}


def _build_extractor(model: str) -> object:
    """Return a concrete extractor for the CLI.

    ``"trigger"`` builds a LLM-free :class:`TriggerExtractor`. Any other
    string is treated as a model id and routed to :class:`LLMExtractor`
    via provider-prefix inference. The legacy ``"rule_based"`` sentinel
    is softened to a :class:`DeprecationWarning` + delegate to ``trigger``
    so older scripts keep working.
    """
    if model == "rule_based":
        warnings.warn(
            "--model rule_based is deprecated: delegating to "
            "TriggerExtractor ('trigger'). Pass --model trigger to "
            "silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        model = "trigger"
    if model == "trigger":
        from nuggetindex.extractors.trigger import TriggerExtractor

        return TriggerExtractor()

    from nuggetindex.extractors.clients.base import LLMConfig
    from nuggetindex.extractors.llm import LLMExtractor

    if model.startswith("claude-"):
        provider = "anthropic"
    elif model.startswith(("gemini-", "models/gemini-")):
        provider = "google"
    else:
        provider = "openai"
    return LLMExtractor(LLMConfig(provider=provider, model=model))


def _wrap_with_cache(extractor: object, cache_path: Path | None) -> object:
    """Wrap ``extractor`` in a :class:`CachedExtractor` when ``cache_path`` is set."""
    if cache_path is None:
        return extractor
    from nuggetindex.extractors.base import BaseExtractor
    from nuggetindex.extractors.cache import CachedExtractor

    assert isinstance(extractor, BaseExtractor), (
        "extractor returned by _build_extractor must inherit BaseExtractor"
    )
    return CachedExtractor(extractor, cache_path=cache_path)


def _discover_files(folder: Path) -> list[Path]:
    """Return text files under ``folder`` sorted deterministically."""
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES
    )


async def _run_build(
    folder: Path,
    db: Path,
    model: str,
    append: bool,
    cache_path: Path | None,
) -> tuple[int, int, int, int]:
    """Core async routine.

    Returns ``(files_ingested, nuggets_added, nuggets_merged, conflicts)``.
    """
    from nuggetindex.pipeline.constructor import Document
    from nuggetindex.store import NuggetStore

    if not append and db.exists():
        db.unlink()

    extractor = _wrap_with_cache(_build_extractor(model), cache_path)
    store = NuggetStore(db_path=db, extractor=extractor)

    files = _discover_files(folder)
    total_files = 0
    total_added = 0
    total_merged = 0
    total_conflicts = 0
    try:
        for path in files:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            doc = Document(
                source_id=str(path.relative_to(folder)),
                text=text,
                uri=str(path),
            )
            result = await store.aingest(doc)
            total_files += 1
            total_added += result.nuggets_added
            total_merged += result.nuggets_merged
            total_conflicts += result.conflicts_detected
    finally:
        await store.aclose()
        # Close the cache's SQLite handle if we opened one.
        inner_close = getattr(extractor, "close", None)
        if callable(inner_close):
            with contextlib.suppress(Exception):  # pragma: no cover -- finaliser
                inner_close()

    return total_files, total_added, total_merged, total_conflicts


async def _run_dry_run(
    folder: Path,
    model: str,
    cache_path: Path | None,
) -> str:
    """Return the rendered Markdown cost estimate without touching the store."""
    from nuggetindex.audit.cost import estimate_ingest_cost
    from nuggetindex.pipeline.constructor import Document

    files = _discover_files(folder)
    docs: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        docs.append(
            Document(
                source_id=str(path.relative_to(folder)),
                text=text,
                uri=str(path),
            )
        )
    estimate = await estimate_ingest_cost(
        docs=docs,
        model_id=model if model != "rule_based" else "trigger",
        cache_path=cache_path,
    )
    return estimate.rendered_markdown


def build_command(
    folder: Path = typer.Argument(
        ...,
        help="Folder to scan recursively for .txt / .md files.",
    ),
    db: Path = typer.Option(
        Path("nuggetindex.db"),
        "--db",
        help="Path to the NuggetStore SQLite file.",
    ),
    model: str = typer.Option(
        "trigger",
        "--model",
        help=(
            'Extractor: "trigger" (default; LLM-free) or an LLM model id '
            'such as "gpt-4o-mini". The legacy "rule_based" alias is '
            'accepted with a deprecation warning.'
        ),
    ),
    append: bool = typer.Option(
        False,
        "--append",
        help="Keep the existing DB and append; otherwise the DB is recreated.",
    ),
    cache: Path | None = typer.Option(
        None,
        "--cache",
        help=(
            "Optional path to a SQLite extractor-cache file. When set, the "
            "extractor is wrapped in CachedExtractor so re-ingests are served "
            "from cache at ~0 cost."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Skip the real ingest and print a cost/wall-time estimate for the "
            "configured model. Honours --cache for the cache hit-rate probe."
        ),
    ),
) -> None:
    """Scan ``folder`` and ingest every text file into ``db``."""
    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: folder not found or not a directory: {folder}", err=True)
        raise typer.Exit(code=2)

    if dry_run:
        rendered = asyncio.run(_run_dry_run(folder, model, cache))
        typer.echo(rendered)
        return

    files_done, added, merged, conflicts = asyncio.run(
        _run_build(folder, db, model, append, cache)
    )

    console.print(
        f"[bold green]build complete[/bold green]: "
        f"files={files_done} added={added} merged={merged} "
        f"conflicts={conflicts} db={db}"
    )


def ingest_command(
    folder: Path = typer.Argument(
        ...,
        help="Folder to scan recursively for .txt / .md files.",
    ),
    db: Path = typer.Option(
        Path("nuggetindex.db"),
        "--db",
        help="Path to the NuggetStore SQLite file.",
    ),
    model: str = typer.Option(
        "trigger",
        "--model",
        help=(
            'Extractor: "trigger" (default; LLM-free) or an LLM model id '
            'such as "gpt-4o-mini". The legacy "rule_based" alias is '
            'accepted with a deprecation warning.'
        ),
    ),
    cache: Path | None = typer.Option(
        None,
        "--cache",
        help=(
            "Optional path to a SQLite extractor-cache file. When set, the "
            "extractor is wrapped in CachedExtractor so re-ingests are served "
            "from cache at ~0 cost."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Skip the real ingest and print a cost/wall-time estimate for the "
            "configured model. Honours --cache for the cache hit-rate probe."
        ),
    ),
) -> None:
    """Append-only build: ingest ``folder`` into an existing ``db``."""
    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: folder not found or not a directory: {folder}", err=True)
        raise typer.Exit(code=2)

    if dry_run:
        rendered = asyncio.run(_run_dry_run(folder, model, cache))
        typer.echo(rendered)
        return

    files_done, added, merged, conflicts = asyncio.run(
        _run_build(folder, db, model, append=True, cache_path=cache)
    )

    console.print(
        f"[bold green]ingest complete[/bold green]: "
        f"files={files_done} added={added} merged={merged} "
        f"conflicts={conflicts} db={db}"
    )
