"""Ingest cost estimator (``--dry-run`` backend).

Given a corpus of :class:`Document` objects and an LLM model id, return a
:class:`CostEstimate` with projected input / output tokens, dollar cost,
expected cache hit-rate, and wall-time.

The estimate is built from a stratified sub-sample of the corpus (the
same sampler the doctor / seeds modules use) to keep the pre-run check
cheap and honest: for a 100k-doc corpus we read at most 100 documents to
produce the numbers. Users get an answer in under a second without ever
touching the LLM.

Price table is hardcoded from public pricing as of 2026-04. Unknown
models fall back to a pessimistic default and flag the discrepancy in
the rendered Markdown so the number a user ships into a budget doesn't
silently under-promise.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ``Document`` lives in ``pipeline.constructor`` which imports
    # ``BaseExtractor`` at module load. Importing it eagerly here creates
    # a cycle via ``extractors -> trigger -> audit.heuristics -> sample ->
    # pipeline.constructor``. The annotation is only needed for type
    # checking, so we keep it behind ``TYPE_CHECKING``.
    from nuggetindex.pipeline.constructor import Document

# ---------------------------------------------------------------------------
# Price table ($ per 1K tokens). Kept conservative and current as of 2026-04.
# ---------------------------------------------------------------------------

# Each entry is (input_price_per_1k, output_price_per_1k).
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "claude-haiku-4-5-20251001": (0.00080, 0.00400),
    "claude-sonnet-4-6": (0.00300, 0.01500),
    "Qwen/Qwen3-32B": (0.00040, 0.00120),
    # Trigger / rule-based: no LLM call -> no cost.
    "trigger": (0.0, 0.0),
}

# Pessimistic defaults for unknown model ids.
_UNKNOWN_IN_PRICE = 0.00050
_UNKNOWN_OUT_PRICE = 0.00200

# Throughput heuristics for wall-time estimation (serial, single worker).
_LLM_TPS = 30.0
_CACHE_TPS = 150.0

# Per-doc empirical bound on output token count (based on ~10 triples/doc
# at ~80 tokens each, clipped at 800 for very long docs).
_OUTPUT_CHAR_DIVISOR = 10
_OUTPUT_TOKEN_CEILING = 800


@dataclass(frozen=True)
class CostEstimate:
    """Projected cost / wall-time for a full ingest run.

    ``n_docs_total`` is ``None`` when the caller passed a streaming
    iterable (we can't get a length in a single pass). In that case,
    ``total_input_tokens_est`` / ``total_output_tokens_est`` are set to
    the sampled-only totals and ``rendered_markdown`` flags the gap so
    users know they're seeing a lower bound.
    """

    n_docs_total: int | None
    n_docs_sampled: int
    mean_input_tokens: float
    mean_output_tokens: float
    total_input_tokens_est: int
    total_output_tokens_est: int
    model_id: str
    input_price_per_1k: float
    output_price_per_1k: float
    total_cost_usd_est: float
    cache_hit_rate_expected: float
    net_cost_usd_est: float
    wall_time_seconds_est: float
    rendered_markdown: str


async def estimate_ingest_cost(
    *,
    docs: Iterable[Document] | AsyncIterable[Document],
    sample_size: int = 100,
    model_id: str = "gpt-4o-mini",
    cache_path: Path | str | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> CostEstimate:
    """Sample the corpus and project the full-run cost.

    Parameters
    ----------
    docs:
        Either a concrete sequence (recommended: we get ``n_docs_total``
        for free and the stratified sampler does its proper bucketing),
        or a sync / async iterable (degrades to reservoir sampling;
        ``n_docs_total`` is ``None``).
    sample_size:
        Target number of documents to draw for the estimate. Defaults to
        100 -- big enough to stabilise the mean, small enough to keep the
        dry-run under a second.
    model_id:
        Canonical model identifier used to look up pricing. ``"trigger"``
        is recognised as the LLM-free path and yields ``$0``.
    cache_path:
        Optional path to a :class:`~nuggetindex.extractors.cache.CachedExtractor`
        SQLite file. When set, we open the file read-only and count how
        many of the sampled documents already have a matching
        ``content_hash`` entry, taking that fraction as the expected
        cache-hit rate.
    tokenizer:
        Optional custom token counter. When ``None`` we try
        ``tiktoken`` (for OpenAI-ish vocab) and fall back to a
        character-based heuristic (``len(text) // 4``). The heuristic is
        intentionally simple: we're projecting order-of-magnitude cost,
        not billing.
    """
    from nuggetindex.audit.heuristics.sample import stratified_sample

    sampled, n_total = await stratified_sample(
        docs,
        sample_size=sample_size,
        stratify_by="composite",
    )

    tok = tokenizer if tokenizer is not None else _default_tokenizer(model_id)

    n_sampled = len(sampled)
    if n_sampled == 0:
        return _empty_estimate(model_id=model_id, n_total=n_total)

    # --- token counts per doc ------------------------------------------
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    for doc in sampled:
        text = doc.text or ""
        n_in = max(1, int(tok(text)))
        n_out = min(_OUTPUT_TOKEN_CEILING, len(text) // _OUTPUT_CHAR_DIVISOR)
        input_tokens.append(n_in)
        output_tokens.append(max(1, n_out))

    mean_in = sum(input_tokens) / n_sampled
    mean_out = sum(output_tokens) / n_sampled

    # --- project to full corpus ----------------------------------------
    projection_factor = n_total if n_total is not None else n_sampled
    total_in = int(round(mean_in * projection_factor))
    total_out = int(round(mean_out * projection_factor))

    # --- pricing lookup ------------------------------------------------
    in_price, out_price, unknown = _price_for(model_id)
    gross_cost = (total_in / 1000.0) * in_price + (total_out / 1000.0) * out_price

    # --- cache hit rate ------------------------------------------------
    hit_rate = 0.0
    if cache_path is not None:
        hit_rate = _probe_cache_hit_rate(
            Path(cache_path),
            sampled_texts=[d.text or "" for d in sampled],
            model_id=model_id,
        )
    net_cost = gross_cost * (1.0 - hit_rate)

    # --- wall-time -----------------------------------------------------
    seconds_live = (total_in + total_out) / _LLM_TPS if model_id != "trigger" else 0.0
    # Cached hits bypass the LLM; approximate their wall-time share by
    # the much higher cached-read TPS.
    if hit_rate > 0:
        seconds_cached = hit_rate * (total_in + total_out) / _CACHE_TPS
        seconds_live = (1.0 - hit_rate) * seconds_live
        wall_time = seconds_live + seconds_cached
    else:
        wall_time = seconds_live

    rendered = _render_markdown(
        model_id=model_id,
        n_total=n_total,
        n_sampled=n_sampled,
        mean_in=mean_in,
        mean_out=mean_out,
        total_in=total_in,
        total_out=total_out,
        in_price=in_price,
        out_price=out_price,
        gross_cost=gross_cost,
        hit_rate=hit_rate,
        net_cost=net_cost,
        wall_time=wall_time,
        unknown_model=unknown,
        cache_path=Path(cache_path) if cache_path is not None else None,
    )

    return CostEstimate(
        n_docs_total=n_total,
        n_docs_sampled=n_sampled,
        mean_input_tokens=mean_in,
        mean_output_tokens=mean_out,
        total_input_tokens_est=total_in,
        total_output_tokens_est=total_out,
        model_id=model_id,
        input_price_per_1k=in_price,
        output_price_per_1k=out_price,
        total_cost_usd_est=gross_cost,
        cache_hit_rate_expected=hit_rate,
        net_cost_usd_est=net_cost,
        wall_time_seconds_est=wall_time,
        rendered_markdown=rendered,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _price_for(model_id: str) -> tuple[float, float, bool]:
    """Return ``(input_price, output_price, is_unknown)`` for ``model_id``."""
    if model_id in _PRICE_TABLE:
        in_p, out_p = _PRICE_TABLE[model_id]
        return in_p, out_p, False
    return _UNKNOWN_IN_PRICE, _UNKNOWN_OUT_PRICE, True


def _default_tokenizer(model_id: str) -> Callable[[str], int]:
    """Return a token counter. Prefer ``tiktoken``; fall back to ``len(s) // 4``.

    ``tiktoken`` is an optional dependency; this function must never import
    it at module load time. For LLM-free models (``"trigger"``) we skip the
    tokenizer entirely since the output never reaches an LLM.
    """
    if model_id == "trigger":
        # Irrelevant, but we still need a counter for the "mean input
        # tokens" row. A coarse char/4 is fine — it's not billed.
        return _char_div4
    try:  # pragma: no branch
        import tiktoken
    except Exception:
        return _char_div4
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover -- defensive
        return _char_div4
    return lambda s: len(enc.encode(s or ""))


def _char_div4(text: str) -> int:
    """Classic back-of-envelope heuristic: ~4 characters per token."""
    return max(1, len(text) // 4)


def _probe_cache_hit_rate(
    cache_path: Path,
    *,
    sampled_texts: list[str],
    model_id: str,
) -> float:
    """Open the cache read-only and compute ``#hits / #sampled``.

    The cache's content hash is ``sha256(text + '|' + extractor_id)`` — we
    assemble the same id here (no inner-extractor instance needed) so the
    probe is side-effect-free. Best-effort: any error (missing file,
    missing table, schema drift) returns ``0.0`` rather than bubbling up.
    """
    if not cache_path.exists() or not sampled_texts:
        return 0.0

    extractor_id = _cache_probe_extractor_id(model_id)
    try:
        conn = sqlite3.connect(
            f"file:{cache_path}?mode=ro",
            uri=True,
            isolation_level=None,
        )
    except sqlite3.DatabaseError:
        return 0.0
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extractor_cache'"
        )
        if cursor.fetchone() is None:
            return 0.0

        from nuggetindex.extractors.cache import content_hash_for

        hits = 0
        for text in sampled_texts:
            key = content_hash_for(text, extractor_id)
            row = conn.execute(
                "SELECT 1 FROM extractor_cache WHERE content_hash = ?",
                (key,),
            ).fetchone()
            if row is not None:
                hits += 1
        return hits / len(sampled_texts)
    finally:
        conn.close()


def _cache_probe_extractor_id(model_id: str) -> str:
    """Reconstruct the cache's ``extractor_id`` for a given model id.

    Mirrors :func:`nuggetindex.extractors.cache._infer_extractor_id` without
    instantiating the inner extractor. We pick the provider by prefix in the
    same way the CLI builders do.
    """
    if model_id == "trigger":
        from nuggetindex.audit.heuristics.triggers import TRIGGER_VERSION

        return f"trigger:{TRIGGER_VERSION}"
    from nuggetindex.extractors.prompts import PROMPT_VERSION

    provider = _provider_for(model_id)
    return f"llm:{provider}:{model_id}:{PROMPT_VERSION}"


def _provider_for(model_id: str) -> str:
    """Provider inference mirrors the CLI builders."""
    if model_id.startswith("claude-"):
        return "anthropic"
    if model_id.startswith(("gemini-", "models/gemini-")):
        return "google"
    return "openai"


def _empty_estimate(*, model_id: str, n_total: int | None) -> CostEstimate:
    """Fallback for an empty corpus -- zero everything, short markdown."""
    in_price, out_price, unknown = _price_for(model_id)
    rendered = (
        "## Ingest cost estimate\n\n"
        "Sampled 0 documents -- nothing to project. "
        "Is the input corpus empty?\n"
    )
    if unknown:
        rendered += (
            "\n> **Note:** pricing for `"
            + model_id
            + "` is unknown; falling back to a **pessimistic default** "
            f"(${in_price*1000:.2f} / ${out_price*1000:.2f} per 1M tokens).\n"
        )
    return CostEstimate(
        n_docs_total=n_total,
        n_docs_sampled=0,
        mean_input_tokens=0.0,
        mean_output_tokens=0.0,
        total_input_tokens_est=0,
        total_output_tokens_est=0,
        model_id=model_id,
        input_price_per_1k=in_price,
        output_price_per_1k=out_price,
        total_cost_usd_est=0.0,
        cache_hit_rate_expected=0.0,
        net_cost_usd_est=0.0,
        wall_time_seconds_est=0.0,
        rendered_markdown=rendered,
    )


def _render_markdown(
    *,
    model_id: str,
    n_total: int | None,
    n_sampled: int,
    mean_in: float,
    mean_out: float,
    total_in: int,
    total_out: int,
    in_price: float,
    out_price: float,
    gross_cost: float,
    hit_rate: float,
    net_cost: float,
    wall_time: float,
    unknown_model: bool,
    cache_path: Path | None,
) -> str:
    """Render a human-readable Markdown table of the estimate."""
    lines: list[str] = []
    lines.append("## Ingest cost estimate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Model | `{model_id}` |")
    lines.append(
        f"| Documents | {n_total if n_total is not None else '?'} "
        f"(sampled {n_sampled}) |"
    )
    lines.append(f"| Mean input tokens / doc | {mean_in:.1f} |")
    lines.append(f"| Mean output tokens / doc | {mean_out:.1f} |")
    lines.append(f"| Total input tokens (est) | {total_in:,} |")
    lines.append(f"| Total output tokens (est) | {total_out:,} |")
    lines.append(
        f"| Price | ${in_price*1000:.2f} in / ${out_price*1000:.2f} out per 1M |"
    )
    lines.append(f"| Gross cost | ${gross_cost:.4f} |")
    lines.append(f"| Cache hit rate (expected) | {hit_rate*100:.1f}% |")
    lines.append(f"| Net cost | ${net_cost:.4f} |")
    lines.append(f"| Wall-time (est, serial) | {wall_time:.1f} s |")
    lines.append("")

    if n_total is None:
        lines.append(
            "> **Streaming input:** corpus size is unknown, so totals reflect "
            "the sampled subset only. For a full-corpus projection pass a "
            "concrete list/sequence of Documents."
        )
        lines.append("")
    if cache_path is not None and not cache_path.exists():
        lines.append(
            f"> **Cache probe:** `{cache_path}` does not exist yet; "
            "hit-rate is 0% (first run)."
        )
        lines.append("")
    if unknown_model:
        lines.append(
            f"> **Note:** pricing for `{model_id}` is unknown; using a "
            f"**pessimistic default** of ${in_price*1000:.2f} / "
            f"${out_price*1000:.2f} per 1M tokens. Pass an explicit model id "
            "from the bundled table (`gpt-4o-mini`, `gpt-4o`, "
            "`claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, "
            "`Qwen/Qwen3-32B`, `trigger`) for a calibrated number."
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CostEstimate",
    "estimate_ingest_cost",
]
