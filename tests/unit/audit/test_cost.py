"""Unit tests for :func:`nuggetindex.audit.cost.estimate_ingest_cost`.

The estimator is a pure projection built on top of a stratified sample
of the corpus. These tests stay offline: no tokenizer extras are
required (the char/4 fallback is deterministic), and the cache probe
uses the real :class:`CachedExtractor` schema with hand-seeded rows.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex.audit.cost import (
    CostEstimate,
    estimate_ingest_cost,
)
from nuggetindex.pipeline.constructor import Document


def _make_docs(n: int) -> list[Document]:
    """Return ``n`` small, non-blank docs with staggered source dates."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    docs: list[Document] = []
    for i in range(n):
        text = (
            f"Document {i}: Microsoft acquired LinkedIn for $26.2 billion "
            "and later announced Satya Nadella as CEO of Microsoft."
        )
        docs.append(
            Document(
                source_id=f"d{i:03d}",
                text=text,
                uri=f"https://example.com/{i}",
                source_date=base.replace(day=(i % 28) + 1),
            )
        )
    return docs


@pytest.mark.asyncio
async def test_estimate_returns_shape() -> None:
    """A synthetic 50-doc corpus + sample_size=10 yields a populated estimate."""
    docs = _make_docs(50)
    est = await estimate_ingest_cost(
        docs=docs,
        sample_size=10,
        model_id="gpt-4o-mini",
    )
    assert isinstance(est, CostEstimate)
    assert est.n_docs_total == 50
    assert est.n_docs_sampled > 0
    assert est.mean_input_tokens > 0
    assert est.mean_output_tokens > 0
    assert est.total_input_tokens_est > 0
    assert est.total_output_tokens_est > 0
    assert est.model_id == "gpt-4o-mini"
    assert est.input_price_per_1k == pytest.approx(0.00015)
    assert est.output_price_per_1k == pytest.approx(0.00060)
    assert est.total_cost_usd_est > 0
    assert est.cache_hit_rate_expected == 0.0
    assert est.net_cost_usd_est == pytest.approx(est.total_cost_usd_est)
    assert est.wall_time_seconds_est > 0
    assert "Ingest cost estimate" in est.rendered_markdown
    assert "gpt-4o-mini" in est.rendered_markdown


@pytest.mark.asyncio
async def test_zero_cost_for_trigger_model() -> None:
    """``model_id='trigger'`` -> $0 gross and $0 net (no LLM calls)."""
    docs = _make_docs(20)
    est = await estimate_ingest_cost(
        docs=docs,
        sample_size=10,
        model_id="trigger",
    )
    assert est.total_cost_usd_est == 0.0
    assert est.net_cost_usd_est == 0.0
    assert est.wall_time_seconds_est == 0.0
    assert est.input_price_per_1k == 0.0
    assert est.output_price_per_1k == 0.0


@pytest.mark.asyncio
async def test_cache_hit_rate_lowers_net_cost(tmp_path: Path) -> None:
    """A 50%-populated cache should roughly halve ``net_cost_usd_est``."""
    docs = _make_docs(20)

    # Build a realistic cache file with the same schema the CachedExtractor
    # writes. Pre-populate entries for exactly half of the sampled corpus
    # using the real ``content_hash_for`` helper + the production
    # extractor_id that the cost probe will reconstruct for 'gpt-4o-mini'.
    from nuggetindex.audit.heuristics.sample import stratified_sample
    from nuggetindex.extractors.cache import content_hash_for
    from nuggetindex.extractors.prompts import PROMPT_VERSION

    sampled, _ = await stratified_sample(
        docs, sample_size=20, stratify_by="composite"
    )
    extractor_id = f"llm:openai:gpt-4o-mini:{PROMPT_VERSION}"

    cache_path = tmp_path / "cache.db"
    conn = sqlite3.connect(str(cache_path))
    conn.execute(
        "CREATE TABLE extractor_cache ("
        "content_hash TEXT PRIMARY KEY,"
        "extractor_id TEXT NOT NULL,"
        "results_json TEXT NOT NULL,"
        "created_at TEXT NOT NULL"
        ")"
    )
    # Pre-populate the first half of the sample so the probe reports
    # exactly 50% on the same sample.
    half = len(sampled) // 2
    for doc in sampled[:half]:
        key = content_hash_for(doc.text, extractor_id)
        conn.execute(
            "INSERT INTO extractor_cache VALUES (?, ?, ?, ?)",
            (key, extractor_id, json.dumps([]), "2024-01-01T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()

    est = await estimate_ingest_cost(
        docs=docs,
        sample_size=20,
        model_id="gpt-4o-mini",
        cache_path=cache_path,
    )
    # Stratified sampler is deterministic given the same seed, so the
    # probe should see exactly half of the entries it pre-populated --
    # modulo a small tolerance if rounding clips an edge doc.
    assert 0.4 <= est.cache_hit_rate_expected <= 0.6
    assert est.net_cost_usd_est == pytest.approx(
        est.total_cost_usd_est * (1 - est.cache_hit_rate_expected),
        rel=0.05,
    )


@pytest.mark.asyncio
async def test_unknown_model_flags_pessimistic() -> None:
    """Unknown model ids render a "pessimistic default" warning."""
    docs = _make_docs(10)
    est = await estimate_ingest_cost(
        docs=docs,
        sample_size=5,
        model_id="completely-unknown-model-v99",
    )
    assert "pessimistic default" in est.rendered_markdown.lower()
    # Fallback pricing is > 0 so net > 0 for non-empty corpus.
    assert est.net_cost_usd_est > 0.0


@pytest.mark.asyncio
async def test_empty_corpus_handled_gracefully() -> None:
    """Empty input returns a zero-filled CostEstimate, not a crash."""
    est = await estimate_ingest_cost(
        docs=[],
        sample_size=10,
        model_id="gpt-4o-mini",
    )
    assert est.n_docs_sampled == 0
    assert est.total_cost_usd_est == 0.0
    assert est.net_cost_usd_est == 0.0
    assert "nothing to project" in est.rendered_markdown.lower()


@pytest.mark.asyncio
async def test_missing_cache_file_is_zero_rate(tmp_path: Path) -> None:
    """A cache path that doesn't exist reports 0% hit-rate, not a crash."""
    docs = _make_docs(10)
    est = await estimate_ingest_cost(
        docs=docs,
        sample_size=5,
        model_id="gpt-4o-mini",
        cache_path=tmp_path / "does-not-exist.db",
    )
    assert est.cache_hit_rate_expected == 0.0
    assert est.net_cost_usd_est == pytest.approx(est.total_cost_usd_est)


def test_content_hash_for_is_stable() -> None:
    """Cache-probe hash = sha256(text + '|' + extractor_id)."""
    from nuggetindex.extractors.cache import content_hash_for

    digest = content_hash_for("hello", "id-1")
    expected = hashlib.sha256(b"hello|id-1").hexdigest()
    assert digest == expected
