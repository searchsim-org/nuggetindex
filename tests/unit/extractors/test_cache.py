"""Unit tests for :class:`nuggetindex.extractors.cache.CachedExtractor`.

The cache wraps any :class:`BaseExtractor` and memoizes its output by a
content hash of ``(text, extractor_id)``. The tests below lock:

* write-through on misses + cache hits on repeat calls;
* extractor-id participation in the key (different id -> different entry);
* ``stats()`` bookkeeping;
* on-disk persistence across independent wrapper instances.

An in-memory fake extractor is used so nothing touches the LLM or the
real trigger scanner — cache semantics are independent of the inner
extractor's behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.extractors.cache import CachedExtractor


class _CountingExtractor(BaseExtractor):
    """Tiny counter-based extractor that emits one deterministic nugget."""

    def __init__(self, tag: str = "A") -> None:
        self._tag = tag
        self.calls = 0

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",  # noqa: ARG002
        source_id: str | None = None,
    ) -> list[ExtractionResult]:
        self.calls += 1
        nugget = Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject="subj",
                predicate="pred",
                object=f"{self._tag}-{text[:10]}",
                text=text[:20] or "x",
            ),
            validity=ValidityInterval(start=datetime(2024, 1, 1, tzinfo=UTC)),
            epistemic=EpistemicState(confidence=0.9),
            provenance=(
                ProvenanceRecord(
                    source_id=source_id or "counter",
                    evidence_span=text[:20] or "x",
                ),
            ),
            extraction_confidence=0.9,
        )
        return [ExtractionResult(nugget=nugget, confidence=0.9)]


@pytest.mark.asyncio
async def test_second_call_hits_cache_zero_inner_calls(
    tmp_path: Path,
) -> None:
    """First call hits the inner extractor; second call is served from cache."""
    inner = _CountingExtractor()
    cached = CachedExtractor(
        inner,
        cache_path=tmp_path / "c.db",
        extractor_id="test:v1",
    )

    text = "Microsoft acquired LinkedIn for $26.2 billion."
    out1 = await cached.aextract(text)
    out2 = await cached.aextract(text)

    assert inner.calls == 1, "inner extractor should only be called once"
    assert len(out1) == len(out2) == 1
    # Round-trip preserves the nugget id.
    assert out1[0].nugget.id == out2[0].nugget.id
    cached.close()


@pytest.mark.asyncio
async def test_different_extractor_id_means_different_cache_entry(
    tmp_path: Path,
) -> None:
    """Changing ``extractor_id`` invalidates the cache (two inner calls)."""
    inner = _CountingExtractor()
    text = "Microsoft acquired LinkedIn for $26.2 billion."

    cache_path = tmp_path / "c.db"
    first = CachedExtractor(inner, cache_path=cache_path, extractor_id="test:v1")
    await first.aextract(text)
    first.close()

    second = CachedExtractor(inner, cache_path=cache_path, extractor_id="test:v2")
    await second.aextract(text)
    second.close()

    assert inner.calls == 2, "different extractor_id should bypass the cache"


@pytest.mark.asyncio
async def test_stats_reports_hits_and_misses(tmp_path: Path) -> None:
    inner = _CountingExtractor()
    cached = CachedExtractor(
        inner,
        cache_path=tmp_path / "c.db",
        extractor_id="test:v1",
    )

    await cached.aextract("alpha")
    await cached.aextract("alpha")
    await cached.aextract("beta")

    stats = cached.stats()
    assert stats == {"hits": 1, "misses": 2, "total": 3}
    cached.close()


@pytest.mark.asyncio
async def test_cache_persists_across_instances(tmp_path: Path) -> None:
    """A fresh wrapper instance pointing at the same file sees prior writes."""
    cache_path = tmp_path / "c.db"
    text = "Twitter Inc. was renamed to X Corp. in 2023."

    inner_a = _CountingExtractor()
    first = CachedExtractor(inner_a, cache_path=cache_path, extractor_id="test:v1")
    await first.aextract(text)
    first.close()
    assert inner_a.calls == 1

    inner_b = _CountingExtractor()
    second = CachedExtractor(inner_b, cache_path=cache_path, extractor_id="test:v1")
    results = await second.aextract(text)
    second.close()

    assert inner_b.calls == 0, "persisted cache should satisfy the second call"
    assert len(results) == 1
    assert second.stats()["hits"] == 1


@pytest.mark.asyncio
async def test_context_is_folded_into_cache_key(tmp_path: Path) -> None:
    """Changing ``context`` changes the effective extractor_id -> cache miss."""
    inner = _CountingExtractor()
    cached = CachedExtractor(
        inner,
        cache_path=tmp_path / "c.db",
        extractor_id="test:v1",
    )
    text = "Microsoft acquired LinkedIn."
    await cached.aextract(text, context="ctx-A")
    await cached.aextract(text, context="ctx-A")
    await cached.aextract(text, context="ctx-B")
    assert inner.calls == 2
    cached.close()


def test_schema_mismatch_raises(tmp_path: Path) -> None:
    """An existing cache file with an unexpected schema raises RuntimeError."""
    import sqlite3

    path = tmp_path / "c.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE extractor_cache (wrong TEXT)")
    conn.commit()
    conn.close()

    inner = _CountingExtractor()
    with pytest.raises(RuntimeError, match="schema mismatch"):
        CachedExtractor(inner, cache_path=path, extractor_id="test:v1")
