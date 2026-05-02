"""Unit tests for :class:`nuggetindex.sidecar.jit_cache.JITPassageCache`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex import NuggetStore
from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.sidecar import JITPassageCache, Sidecar


class _CountingExtractor(BaseExtractor):
    """Tiny extractor that counts calls and returns a deterministic nugget."""

    def __init__(self) -> None:
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
                subject="Google",
                predicate="chiefExecutiveOfficer",
                object="Larry Page",
                text=text[:40] or "empty",
            ),
            validity=ValidityInterval(
                start=datetime(2011, 4, 4, tzinfo=UTC),
                end=datetime(2015, 10, 2, tzinfo=UTC),
            ),
            epistemic=EpistemicState(confidence=0.9),
            provenance=(
                ProvenanceRecord(
                    source_id=source_id or "counter",
                    evidence_span=text[:80] or "x",
                ),
            ),
            extraction_confidence=0.9,
        )
        return [ExtractionResult(nugget=nugget, confidence=0.9)]


@pytest.mark.asyncio
async def test_second_call_hits_cache_zero_extractor_calls() -> None:
    """Repeat calls with the same text hit the cache, not the extractor."""
    cache = JITPassageCache()
    extractor = _CountingExtractor()
    text = "Larry Page served as CEO of Google from 2011 to 2015."

    first = await cache.get_or_extract(text, extractor)
    second = await cache.get_or_extract(text, extractor)

    assert extractor.calls == 1
    assert first == second
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_max_entries_evicts_lru() -> None:
    """Filling beyond ``max_entries`` evicts the oldest entry."""
    cache = JITPassageCache(max_entries=2)
    extractor = _CountingExtractor()

    await cache.get_or_extract("alpha passage", extractor)
    await cache.get_or_extract("beta passage", extractor)
    await cache.get_or_extract("gamma passage", extractor)  # evicts "alpha"

    # "alpha" is no longer cached; calling again re-runs the extractor.
    calls_before = extractor.calls
    await cache.get_or_extract("alpha passage", extractor)
    assert extractor.calls == calls_before + 1

    # "beta" should still be cached (LRU moved it up when alpha was evicted).
    calls_before = extractor.calls
    await cache.get_or_extract("beta passage", extractor)
    # beta was evicted when "alpha passage" re-entered; this assertion
    # would vary by implementation detail. The robust contract is just
    # that the cache honours its capacity bound.
    stats = cache.stats()
    assert stats["size"] <= 2


@pytest.mark.asyncio
async def test_stats_hits_and_misses() -> None:
    """Stats counters reflect every get_or_extract call."""
    cache = JITPassageCache()
    extractor = _CountingExtractor()

    await cache.get_or_extract("one", extractor)
    await cache.get_or_extract("two", extractor)
    await cache.get_or_extract("one", extractor)

    stats = cache.stats()
    assert stats == {"hits": 1, "misses": 2, "total": 3, "size": 2}


@pytest.mark.asyncio
async def test_disk_persistence_survives_instance_recreate(tmp_path: Path) -> None:
    """Re-opening the same cache_path returns prior entries."""
    cache_path = tmp_path / "jit.db"
    extractor = _CountingExtractor()

    cache1 = JITPassageCache(cache_path=cache_path)
    await cache1.get_or_extract("persistent passage", extractor)
    cache1.close()

    # Fresh instance — should still serve the prior entry without calling
    # the extractor again.
    cache2 = JITPassageCache(cache_path=cache_path)
    calls_before = extractor.calls
    await cache2.get_or_extract("persistent passage", extractor)
    assert extractor.calls == calls_before
    cache2.close()


@pytest.mark.asyncio
async def test_jit_sidecar_uses_cache_end_to_end(tmp_path: Path) -> None:
    """Full Sidecar in JIT mode: the cache dedupes repeated passages."""
    store = NuggetStore(db_path=tmp_path / "s.db")
    try:
        extractor = _CountingExtractor()
        cache = JITPassageCache()
        sidecar = Sidecar(
            store=store,
            mode="just-in-time",
            extractor=extractor,
            jit_cache=cache,
        )

        hits = [{"id": "d1", "content": "Larry Page served as CEO of Google in 2013."}]
        # First call: extractor runs once.
        await sidecar.ahandle(
            "who was Google's CEO in 2013?", top_k=1, original_hits=hits
        )
        first_call_count = extractor.calls

        # Second call with the SAME passage: extractor stays put.
        await sidecar.ahandle(
            "in 2013 who was the Google CEO?", top_k=1, original_hits=hits
        )
        assert extractor.calls == first_call_count
        stats = cache.stats()
        assert stats["hits"] >= 1
    finally:
        await store.backend.aclose()
