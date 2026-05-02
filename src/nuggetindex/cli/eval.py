"""``nuggetindex eval`` -- benchmark eval runner CLI.

Thin wrapper around :func:`nuggetindex.eval.run_eval`. Loads a benchmark
(or a pre-saved one), runs the baseline + sidecar on it, and writes the
resulting :class:`EvalReport` as Markdown or JSON.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer


def eval_command(
    benchmark: str = typer.Option(
        "sanity",
        "--benchmark",
        help="Benchmark name: 'sanity', 'timeqa', or 'situatedqa'.",
    ),
    store_path: Path = typer.Option(
        Path(".nuggetindex/store.db"),
        "--store",
        help="Path to a pre-built NuggetStore SQLite file.",
    ),
    retriever_spec: str = typer.Option(
        "bm25",
        "--retriever",
        help="Baseline retriever: 'bm25' (uses the store's FTS5) or 'none'.",
    ),
    answerer_spec: str = typer.Option(
        "string-match",
        "--answerer",
        help="Answerer: 'string-match' (default) or an LLM model id.",
    ),
    max_queries: int = typer.Option(
        100,
        "--max-queries",
        help="Cap on benchmark size.",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Write the report here (.md / .json inferred from extension).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Emit progress lines on stderr.",
    ),
) -> None:
    """Score baseline vs sidecar on a benchmark."""
    if benchmark not in {"sanity", "timeqa", "situatedqa"}:
        typer.echo(
            f"eval: --benchmark must be 'sanity', 'timeqa', or 'situatedqa' (got {benchmark!r}).",
            err=True,
        )
        raise typer.Exit(code=1)
    if retriever_spec not in {"bm25", "none"}:
        typer.echo(
            f"eval: --retriever must be 'bm25' or 'none' (got {retriever_spec!r}).",
            err=True,
        )
        raise typer.Exit(code=1)

    # Lazy import to keep the CLI cheap.
    from nuggetindex.eval import run_eval
    from nuggetindex.sidecar import Sidecar
    from nuggetindex.store import NuggetStore

    store = NuggetStore(db_path=store_path)
    sidecar = Sidecar(store=store, mode="offline-curated")

    baseline_retriever = _build_retriever(retriever_spec, store=store)
    answerer = _resolve_answerer(answerer_spec)

    async def _run() -> None:
        report_obj = await run_eval(
            benchmark=benchmark,  # type: ignore[arg-type]
            sidecar=sidecar,
            baseline_retriever=baseline_retriever,
            answerer=answerer,
            max_queries=max_queries,
        )
        if verbose:
            typer.echo(
                f"eval: n={report_obj.n_queries} "
                f"baseline_em={report_obj.baseline_em:.3f} "
                f"sidecar_em={report_obj.sidecar_em:.3f} "
                f"delta={report_obj.delta_em:+.3f}",
                err=True,
            )
        _write_report(report_obj, report)
        await store.backend.aclose()

    asyncio.run(_run())


def _build_retriever(spec: str, *, store: Any):  # type: ignore[no-untyped-def]
    """Return a ``(query, top_k) -> list[hit]`` function for the CLI."""
    if spec == "none":

        def _empty(_query: str, _top_k: int) -> list[Any]:
            return []

        return _empty

    # bm25 path -- query the store's own retriever (BM25 over the FTS5 table)
    # and return the raw results so the harness can extract ids + text.
    async def _bm25(query: str, top_k: int) -> list[Any]:
        results = await store.aretrieve(query=query, top_k=top_k)
        return list(results)

    return _bm25


def _resolve_answerer(spec: str):  # type: ignore[no-untyped-def]
    """Map the ``--answerer`` flag onto a ``(context, query) -> answer`` fn.

    ``'string-match'`` returns ``None`` so the harness uses its built-in
    oracle. Other values are treated as LLM model ids and dispatched via
    the client/LLM extractor stack; when the extras are missing we fall
    back to the oracle with a stderr warning.
    """
    normalised = spec.strip().lower()
    if normalised in {"", "string-match", "string_match", "oracle"}:
        return None
    try:
        from nuggetindex.extractors.clients.base import LLMConfig, build_client
    except ImportError:  # pragma: no cover -- defensive
        typer.echo(
            f"eval: --answerer {spec!r} needs the LLM extras; falling back to 'string-match'.",
            err=True,
        )
        return None

    config = LLMConfig(model=spec)
    client = build_client(config)

    async def _answer(context: str, query: str) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Answer the question from the context. "
                    f"Reply with the answer text only.\n\n"
                    f"Context:\n{context}\n\nQuestion: {query}"
                ),
            },
        ]
        raw = await client.acomplete(messages)  # type: ignore[attr-defined]
        return str(raw).strip()

    return _answer


def _write_report(report_obj: Any, out: Path | None) -> None:
    """Dispatch the :class:`EvalReport` to stdout / ``.md`` / ``.json``."""
    if out is None:
        typer.echo(report_obj.rendered_markdown)
        return
    ext = out.suffix.lower()
    if ext in {".md", ".markdown"}:
        out.write_text(report_obj.rendered_markdown, encoding="utf-8")
    elif ext == ".json":
        out.write_text(report_obj.rendered_json, encoding="utf-8")
    else:
        raise typer.BadParameter(
            f"--report must end in .md / .markdown / .json (got {out.suffix!r})",
            param_hint="--report",
        )


__all__ = ["eval_command"]
