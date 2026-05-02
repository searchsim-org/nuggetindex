"""``nuggetindex auto`` -- one-call adoption facade CLI.

Thin wrapper around :func:`nuggetindex.auto.auto`. Reads a JSONL of
documents, runs discovery + seeds + ingest + sidecar, and optionally
writes the report's Markdown dump to a file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer


def auto_command(
    index_path: Path | None = typer.Option(
        None,
        "--index",
        help=(
            "Path to a .jsonl file (one document per line with "
            "'source_id'/'text'). Mutually exclusive with "
            "--corpus-url / --corpus-name."
        ),
    ),
    corpus_url: str | None = typer.Option(
        None,
        "--corpus-url",
        help=(
            "Base URL of a live corpus endpoint (e.g. "
            "'https://search.example.com'). Requires --corpus-name."
        ),
    ),
    corpus_name: str | None = typer.Option(
        None,
        "--corpus-name",
        help="Corpus name on the endpoint (e.g. 'my-corpus').",
    ),
    bootstrap: str = typer.Option(
        "caller",
        "--bootstrap",
        help=(
            "Bootstrap sampling strategy for the corpus: 'caller' "
            "(default; requires --index), 'topic_diverse' (recommended "
            "for live corpora), 'uniform', or 'random_ids'."
        ),
    ),
    sample_size: int = typer.Option(
        500,
        "--sample-size",
        help=(
            "Target size of the bootstrap sample pulled from the corpus "
            "(ignored with --bootstrap=caller)."
        ),
    ),
    budget: int = typer.Option(
        100,
        "--budget",
        help="Seed-query budget (proposer upper bound).",
    ),
    mode: str = typer.Option(
        "offline-curated",
        "--mode",
        help="Sidecar runtime mode: 'offline-curated' or 'just-in-time'.",
    ),
    extractor_spec: str = typer.Option(
        "trigger",
        "--extractor",
        help=(
            "Extractor backend: 'trigger' (default; zero-cost) or an "
            "OpenAI-compatible model id (e.g. 'gpt-4o-mini')."
        ),
    ),
    store_path: Path = typer.Option(
        Path(".nuggetindex/store.db"),
        "--store-path",
        help="Output path for the NuggetStore SQLite file.",
    ),
    cache_path: Path = typer.Option(
        Path(".nuggetindex/extractor-cache.db"),
        "--cache-path",
        help="Path to the extractor cache SQLite file.",
    ),
    no_schema_discovery: bool = typer.Option(
        False,
        "--no-schema-discovery",
        help="Skip schema inference; use the default schema only.",
    ),
    two_pass: bool = typer.Option(
        False,
        "--two-pass/--no-two-pass",
        help=(
            "After the bootstrap pass, run a second ingest pass driven "
            "by the proposed seeds (deep targeted pulls from --corpus-url). "
            "Requires --corpus-url/--corpus-name."
        ),
    ),
    deep_docs_per_seed: int = typer.Option(
        10,
        "--deep-docs-per-seed",
        help=(
            "Per-seed corpus.search() limit for the Pass-2 deep pull "
            "(ignored without --two-pass)."
        ),
    ),
    deep_budget: int | None = typer.Option(
        None,
        "--deep-budget",
        help=(
            "Max seeds fanned out during the Pass-2 deep pull. "
            "Defaults to the same value as --budget."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Emit progress lines on stderr.",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Write the AutoReport Markdown dump to this file.",
    ),
) -> None:
    """Build an end-to-end sidecar from a JSONL corpus or live endpoint."""
    if mode not in {"offline-curated", "just-in-time"}:
        typer.echo(
            f"auto: --mode must be 'offline-curated' or 'just-in-time' "
            f"(got {mode!r}).",
            err=True,
        )
        raise typer.Exit(code=1)
    if bootstrap not in {"caller", "topic_diverse", "uniform", "random_ids"}:
        typer.echo(
            f"auto: --bootstrap must be one of "
            f"'caller' / 'topic_diverse' / 'uniform' / 'random_ids' "
            f"(got {bootstrap!r}).",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- corpus-URL path vs --index path ---------------------------------
    corpus_url_set = corpus_url is not None
    corpus_name_set = corpus_name is not None
    if corpus_url_set ^ corpus_name_set:
        typer.echo(
            "auto: --corpus-url and --corpus-name must be provided together.",
            err=True,
        )
        raise typer.Exit(code=1)

    corpus: Any = None
    if corpus_url_set and corpus_name_set:
        if index_path is not None:
            typer.echo(
                "auto: pass either --index or --corpus-url/--corpus-name, "
                "not both.",
                err=True,
            )
            raise typer.Exit(code=1)
        if bootstrap == "caller":
            typer.echo(
                "auto: --corpus-url/--corpus-name require a non-'caller' "
                "--bootstrap (try --bootstrap topic_diverse).",
                err=True,
            )
            raise typer.Exit(code=1)
        from nuggetindex.adapters import VespaCorpus

        corpus = VespaCorpus(base_url=corpus_url, corpus=corpus_name)  # type: ignore[arg-type]
    else:
        if index_path is None:
            typer.echo(
                "auto: pass --index PATH or --corpus-url + --corpus-name.",
                err=True,
            )
            raise typer.Exit(code=1)
        if not index_path.exists():
            typer.echo(f"auto: --index not found: {index_path}", err=True)
            raise typer.Exit(code=1)
        suffix = index_path.suffix.lower()
        if suffix not in {".jsonl", ".ndjson"}:
            typer.echo(
                f"auto: unsupported --index extension {index_path.suffix!r}; "
                "expected .jsonl or .ndjson.",
                err=True,
            )
            raise typer.Exit(code=1)

    extractor = _resolve_extractor(extractor_spec)

    from nuggetindex.auto import auto

    if two_pass and corpus is None:
        typer.echo(
            "auto: --two-pass requires --corpus-url/--corpus-name "
            "(Pass 2 issues corpus.search(seed) to pull deeper docs).",
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        auto_kwargs: dict[str, Any] = {
            "budget": budget,
            "sample_size": sample_size,
            "mode": mode,
            "extractor": extractor,
            "store_path": store_path,
            "cache_path": cache_path,
            "schema_discovery": not no_schema_discovery,
            "two_pass": two_pass,
            "deep_docs_per_seed": deep_docs_per_seed,
            "deep_budget": deep_budget,
            "verbose": verbose,
        }
        if corpus is not None:
            auto_kwargs["corpus"] = corpus
            auto_kwargs["bootstrap"] = bootstrap
        else:
            auto_kwargs["docs"] = index_path

        try:
            _sidecar, report_obj = await auto(**auto_kwargs)
        finally:
            # Close any adapter we built to avoid leaked httpx clients.
            if corpus is not None and hasattr(corpus, "aclose"):
                await corpus.aclose()
        typer.echo(report_obj.rendered_markdown)
        if report is not None:
            report.write_text(report_obj.rendered_markdown, encoding="utf-8")
            if verbose:
                typer.echo(f"auto: wrote report to {report}", err=True)

    asyncio.run(_run())


def _resolve_extractor(spec: str):  # type: ignore[no-untyped-def]
    """Map the ``--extractor`` flag to an instance (or ``None`` for default)."""
    normalised = spec.strip().lower()
    if normalised in {"", "trigger", "triggers"}:
        return None
    try:
        from nuggetindex.extractors.clients.base import LLMConfig, build_client
        from nuggetindex.extractors.llm import LLMExtractor
    except ImportError as exc:  # pragma: no cover -- defensive
        raise typer.BadParameter(
            f"--extractor {spec!r} requires the LLM extras; "
            f"install nuggetindex[openai] or use 'trigger'. (ImportError: {exc})",
            param_hint="--extractor",
        ) from exc

    provider = _infer_provider(spec)
    config = LLMConfig(provider=provider, model=spec)
    client = build_client(config)
    return LLMExtractor(cfg=config, client=client)


def _infer_provider(model_spec: str) -> str:
    """Map a model id to the LLM provider."""
    m = model_spec.lower()
    if m.startswith(("claude-", "anthropic/")):
        return "anthropic"
    if m.startswith(("gemini-", "google/")):
        return "google"
    if "/" in m:
        return "openai_compat"
    return "openai"


__all__ = ["auto_command"]
