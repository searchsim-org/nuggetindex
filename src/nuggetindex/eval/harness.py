"""Eval harness orchestrator: baseline vs sidecar on a benchmark.

Reads a benchmark (or inline list of :class:`BenchmarkQuery`), runs the
caller-supplied baseline retriever and the sidecar over each query, scores
answers with exact-match, and returns an :class:`EvalReport` with a
diff-style breakdown (``fixed_by_sidecar``, ``broken_by_sidecar``) plus
pre-rendered Markdown + JSON so downstream tooling can dump either
without re-computing.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex.eval.benchmarks import BenchmarkQuery, load_benchmark
from nuggetindex.eval.metrics import contains_expected, exact_match, f1_score

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.sidecar import Sidecar


# --------------------------------------------------------------------------- #
# Report types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalResult:
    """Per-query result from :func:`run_eval`.

    ``baseline_hits`` / ``sidecar_hits`` are lists of passage ids (or
    whatever ids the retriever returns); they are captured primarily for
    debugging — the harness scores on the answerer's output, not on
    retrieval.
    """

    query: BenchmarkQuery
    baseline_answer: str
    sidecar_answer: str
    baseline_correct: bool
    sidecar_correct: bool
    baseline_hits: list[str]
    sidecar_hits: list[str]
    sidecar_context_block: str


@dataclass(frozen=True)
class EvalReport:
    """Aggregate output of :func:`run_eval`.

    ``fixed_by_sidecar`` lists queries where the baseline was wrong and
    the sidecar was right — the headline "proof of value". ``broken_by_sidecar``
    lists the opposite; it should be empty or very small if the sidecar
    is working as intended.
    """

    benchmark: str
    n_queries: int
    baseline_em: float
    sidecar_em: float
    delta_em: float
    baseline_f1: float
    sidecar_f1: float
    delta_f1: float
    fixed_by_sidecar: list[EvalResult]
    broken_by_sidecar: list[EvalResult]
    results: list[EvalResult] = field(default_factory=list)
    rendered_markdown: str = ""
    rendered_json: str = ""


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def _coerce_hit_id(hit: Any) -> str:
    """Duck-type the baseline retriever's hit into a stable id string.

    Accepts ``str``, an object with ``.id`` / ``.source_id`` / ``.doc_id``
    / ``.meta_id``, or a dict with the same keys. Falls back to ``repr``
    so unknown shapes still serialise.
    """
    if isinstance(hit, str):
        return hit
    for attr in ("id", "source_id", "doc_id", "meta_id"):
        val = getattr(hit, attr, None)
        if isinstance(val, (str, int)):
            return str(val)
    if isinstance(hit, dict):
        for key in ("id", "source_id", "doc_id"):
            v = hit.get(key)
            if isinstance(v, (str, int)):
                return str(v)
    return repr(hit)


def _coerce_hit_text(hit: Any) -> str:
    """Duck-type a baseline hit into its text body (for the context block)."""
    if isinstance(hit, str):
        return hit
    for attr in ("content", "text", "page_content"):
        val = getattr(hit, attr, None)
        if isinstance(val, str):
            return val
    if isinstance(hit, dict):
        for key in ("content", "text", "page_content"):
            v = hit.get(key)
            if isinstance(v, str):
                return v
    return ""


def _format_baseline_context(hits: list[Any]) -> str:
    """Concatenate baseline hit texts into a single context block."""
    parts = [_coerce_hit_text(h) for h in hits if _coerce_hit_text(h)]
    return "\n".join(parts)


async def run_eval(
    *,
    benchmark: Literal["timeqa", "situatedqa", "sanity"] | list[BenchmarkQuery],
    sidecar: Sidecar,
    baseline_retriever: Callable[[str, int], list[Any]]
    | Callable[[str, int], Awaitable[list[Any]]],
    answerer: Callable[[str, str], str]
    | Callable[[str, str], Awaitable[str]]
    | None = None,
    max_queries: int | None = None,
    top_k: int = 5,
) -> EvalReport:
    """Run baseline vs sidecar on a benchmark and return the diff report.

    Parameters
    ----------
    benchmark:
        A benchmark name (``"sanity"`` / ``"timeqa"`` / ``"situatedqa"``)
        or an explicit list of :class:`BenchmarkQuery`.
    sidecar:
        A configured :class:`Sidecar` (any mode).
    baseline_retriever:
        Caller-supplied ``(query, top_k) -> list[hit]`` function. May be
        sync or async. The hit shape is free; we duck-type ids + text out
        of it.
    answerer:
        Optional LLM-backed ``(context, query) -> answer`` callable. When
        absent, a cheap "does the context contain the expected answer?"
        oracle is used — production callers should supply a real
        answerer.
    max_queries:
        Optional cap on benchmark size (applied after loading).
    top_k:
        Number of passages requested from the baseline retriever and
        passed into the sidecar.
    """
    queries = _resolve_queries(benchmark, max_queries=max_queries)
    bench_name = (
        benchmark if isinstance(benchmark, str) else "inline"
    )

    results: list[EvalResult] = []
    for q in queries:
        baseline_hits = await _maybe_await(
            baseline_retriever(q.query, top_k)
        )
        baseline_context = _format_baseline_context(baseline_hits)
        baseline_answer = await _call_answerer(
            answerer, baseline_context, q
        )

        sidecar_resp = await sidecar.ahandle(
            q.query,
            query_time=q.query_time,
            top_k=top_k,
            original_hits=baseline_hits,
        )
        sidecar_context = "\n".join(
            part
            for part in (baseline_context, sidecar_resp.context_block)
            if part
        )
        sidecar_answer = await _call_answerer(
            answerer, sidecar_context, q
        )

        results.append(
            EvalResult(
                query=q,
                baseline_answer=baseline_answer,
                sidecar_answer=sidecar_answer,
                baseline_correct=exact_match(baseline_answer, q.expected_answer),
                sidecar_correct=exact_match(sidecar_answer, q.expected_answer),
                baseline_hits=[_coerce_hit_id(h) for h in baseline_hits],
                sidecar_hits=[_coerce_hit_id(h) for h in baseline_hits]
                + (
                    ["nuggetindex-governance"]
                    if sidecar_resp.context_block
                    else []
                ),
                sidecar_context_block=sidecar_resp.context_block,
            )
        )

    report = _assemble_report(bench_name, results)
    return report


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _resolve_queries(
    benchmark: str | list[BenchmarkQuery],
    *,
    max_queries: int | None,
) -> list[BenchmarkQuery]:
    """Normalise the ``benchmark`` argument into a list of queries."""
    if isinstance(benchmark, str):
        return load_benchmark(benchmark, max_queries=max_queries)
    if max_queries is not None:
        return list(benchmark)[:max_queries]
    return list(benchmark)


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it's a coroutine; otherwise return it as-is.

    Lets the harness accept both sync and async retrievers without forcing
    the caller to wrap their function.
    """
    if hasattr(value, "__await__"):
        return await value
    return value


async def _call_answerer(
    answerer: Callable[[str, str], Any] | None,
    context: str,
    q: BenchmarkQuery,
) -> str:
    """Invoke ``answerer(context, query)`` or the fallback oracle.

    Empty contexts always go through the oracle so that a blank
    retriever output still produces a deterministic (wrong) answer.
    """
    if answerer is None or not context:
        # Fallback oracle: answer = expected iff the context contains it.
        return q.expected_answer if contains_expected(context, q.expected_answer) else ""
    try:
        result = answerer(context, q.query)
    except Exception:  # noqa: BLE001 -- answerer is best-effort
        return ""
    awaited = await _maybe_await(result)
    return str(awaited) if awaited is not None else ""


def _assemble_report(
    benchmark: str,
    results: list[EvalResult],
) -> EvalReport:
    """Aggregate per-query results into an :class:`EvalReport`."""
    n = len(results)
    if n == 0:
        return EvalReport(
            benchmark=benchmark,
            n_queries=0,
            baseline_em=0.0,
            sidecar_em=0.0,
            delta_em=0.0,
            baseline_f1=0.0,
            sidecar_f1=0.0,
            delta_f1=0.0,
            fixed_by_sidecar=[],
            broken_by_sidecar=[],
            results=[],
            rendered_markdown=_render_markdown(benchmark, []),
            rendered_json=_render_json(benchmark, []),
        )

    baseline_em = sum(1 for r in results if r.baseline_correct) / n
    sidecar_em = sum(1 for r in results if r.sidecar_correct) / n
    baseline_f1 = (
        sum(f1_score(r.baseline_answer, r.query.expected_answer) for r in results)
        / n
    )
    sidecar_f1 = (
        sum(f1_score(r.sidecar_answer, r.query.expected_answer) for r in results)
        / n
    )
    fixed = [
        r for r in results
        if r.sidecar_correct and not r.baseline_correct
    ]
    broken = [
        r for r in results
        if r.baseline_correct and not r.sidecar_correct
    ]

    rendered_md = _render_markdown(benchmark, results, fixed=fixed, broken=broken)
    rendered_json = _render_json(benchmark, results)

    return EvalReport(
        benchmark=benchmark,
        n_queries=n,
        baseline_em=baseline_em,
        sidecar_em=sidecar_em,
        delta_em=sidecar_em - baseline_em,
        baseline_f1=baseline_f1,
        sidecar_f1=sidecar_f1,
        delta_f1=sidecar_f1 - baseline_f1,
        fixed_by_sidecar=fixed,
        broken_by_sidecar=broken,
        results=list(results),
        rendered_markdown=rendered_md,
        rendered_json=rendered_json,
    )


def _render_markdown(
    benchmark: str,
    results: list[EvalResult],
    *,
    fixed: list[EvalResult] | None = None,
    broken: list[EvalResult] | None = None,
) -> str:
    """Build a concise Markdown report (headline metrics + diff lists)."""
    lines = [f"# Eval report — benchmark: `{benchmark}`", ""]
    lines.append(f"- queries: **{len(results)}**")
    if not results:
        lines.append("- no queries ran; nothing to score.")
        return "\n".join(lines)

    baseline_em = sum(1 for r in results if r.baseline_correct) / len(results)
    sidecar_em = sum(1 for r in results if r.sidecar_correct) / len(results)
    lines.append(f"- baseline EM: **{baseline_em:.3f}**")
    lines.append(f"- sidecar  EM: **{sidecar_em:.3f}**")
    lines.append(f"- delta EM:   **{sidecar_em - baseline_em:+.3f}**")
    lines.append("")

    fixed = fixed or []
    broken = broken or []
    if fixed:
        lines.append(f"## Fixed by sidecar ({len(fixed)})")
        lines.append("")
        for r in fixed:
            lines.append(
                f"- `{r.query.query}` — baseline=`{r.baseline_answer}` "
                f"sidecar=`{r.sidecar_answer}` expected=`{r.query.expected_answer}`"
            )
        lines.append("")
    if broken:
        lines.append(f"## Broken by sidecar ({len(broken)})")
        lines.append("")
        for r in broken:
            lines.append(
                f"- `{r.query.query}` — baseline=`{r.baseline_answer}` "
                f"sidecar=`{r.sidecar_answer}` expected=`{r.query.expected_answer}`"
            )
        lines.append("")
    return "\n".join(lines)


def _render_json(benchmark: str, results: list[EvalResult]) -> str:
    """Dataclass-as-dict JSON dump for machine-readable downstream tooling."""
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "n_queries": len(results),
        "results": [dataclasses.asdict(r) for r in results],
    }
    return json.dumps(payload, indent=2, default=str)


__all__ = ["EvalReport", "EvalResult", "run_eval"]
