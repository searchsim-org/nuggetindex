"""Tests for ``GovernancePostProcessor`` — the Tier-1 session-cached wedge."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.governance.postprocessor import (
    GovernancePostProcessor,
    RetrievedPassage,
)
from nuggetindex.governance.session_cache import passage_hash
from nuggetindex.pipeline.constructor import Document
from tests.fixtures import RuleBasedExtractor


class _CountingExtractor(BaseExtractor):
    """Extractor that records call count + returns a single deterministic nugget per call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def aextract(self, text: str, *, context: str = "") -> list[ExtractionResult]:
        self.calls.append(text)
        nugget = Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject=f"subj-{len(self.calls)}",
                predicate="is",
                object=f"obj-{len(self.calls)}",
                text=text,
            ),
            validity=ValidityInterval(start=datetime(2024, 1, 1, tzinfo=UTC)),
            epistemic=EpistemicState(confidence=0.9),
            provenance=(ProvenanceRecord(source_id="stub", evidence_span=text),),
            extraction_confidence=0.9,
        )
        return [ExtractionResult(nugget=nugget, confidence=0.9, rationale="stub")]


# --- RetrievedPassage basics ------------------------------------------------


def test_retrieved_passage_defaults():
    p = RetrievedPassage(source_id="doc-1", text="hello")
    assert p.source_id == "doc-1"
    assert p.text == "hello"
    assert p.score == 0.0
    assert p.metadata is None


# --- __init__ is loop-safe --------------------------------------------------


@pytest.mark.asyncio
async def test_init_inside_event_loop_does_not_raise(tmp_path):
    # Constructing the postprocessor while a loop is running must not call
    # asyncio.run(). If it did, Python would raise RuntimeError here.
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
    )
    assert pp is not None


# --- Single-query behaviour -------------------------------------------------


@pytest.mark.asyncio
async def test_first_query_passes_active_passage_through(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=_CountingExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    passages = [RetrievedPassage(source_id="a", text="Alpha is great")]
    out = await pp.apostprocess(passages)
    assert len(out) == 1
    assert out[0].text == "Alpha is great"


@pytest.mark.asyncio
async def test_empty_passages_list_no_op(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=_CountingExtractor(),
    )
    assert await pp.apostprocess([]) == []


@pytest.mark.asyncio
async def test_passage_with_no_extractable_nugget_passes_through(tmp_path):
    # Rule-based extractor returns nothing for passages with no matches.
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    passages = [RetrievedPassage(source_id="a", text="xyzzy plugh frobnicate")]
    out = await pp.apostprocess(passages)
    assert len(out) == 1
    assert out[0].text == "xyzzy plugh frobnicate"


# --- Content-addressed extraction cache -------------------------------------


@pytest.mark.asyncio
async def test_second_call_same_passage_skips_extraction(tmp_path):
    extractor = _CountingExtractor()
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=extractor,
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    p = RetrievedPassage(source_id="a", text="Alpha is great")
    await pp.apostprocess([p])
    assert len(extractor.calls) == 1
    await pp.apostprocess([p])
    # Same passage hash -> extraction NOT re-run.
    assert len(extractor.calls) == 1


@pytest.mark.asyncio
async def test_distinct_passages_each_extract_once(tmp_path):
    extractor = _CountingExtractor()
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=extractor,
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    p1 = RetrievedPassage(source_id="a", text="Alpha is great")
    p2 = RetrievedPassage(source_id="b", text="Beta is better")
    await pp.apostprocess([p1, p2])
    assert len(extractor.calls) == 2


# --- Session cache persistence ----------------------------------------------


@pytest.mark.asyncio
async def test_rebuilt_postprocessor_reuses_cache_on_disk(tmp_path):
    db = tmp_path / "persisted.db"
    extractor1 = _CountingExtractor()
    pp1 = GovernancePostProcessor(
        cache_path=db,
        extractor=extractor1,
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    await pp1.apostprocess([RetrievedPassage(source_id="a", text="Alpha is great")])
    assert len(extractor1.calls) == 1

    # Rebuild the postprocessor against the same cache file — the sidecar
    # hashes file should make the second postprocessor skip re-extraction.
    extractor2 = _CountingExtractor()
    pp2 = GovernancePostProcessor(
        cache_path=db,
        extractor=extractor2,
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    await pp2.apostprocess([RetrievedPassage(source_id="a", text="Alpha is great")])
    assert len(extractor2.calls) == 0


# --- Filter + flag ----------------------------------------------------------


@pytest.mark.asyncio
async def test_deprecated_nugget_filters_passage(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    passages = [RetrievedPassage(source_id="doc-x", text="Tim Cook is CEO of Apple.")]
    out = await pp.apostprocess(passages)
    assert len(out) == 1

    # Manually mark all nuggets sourced from doc-x as DEPRECATED so the
    # filter_deprecated branch triggers.
    nuggets = await pp._store.backend.aget_nuggets_by_source("doc-x")
    assert nuggets
    for n in nuggets:
        updated = n.model_copy(
            update={
                "epistemic": EpistemicState(
                    status=LifecycleStatus.DEPRECATED,
                    rank=n.epistemic.rank,
                    confidence=n.epistemic.confidence,
                ),
            }
        )
        await pp._store.backend.aupsert(updated)

    # Re-run with the same passage — extractor is skipped, but the governance
    # state is now DEPRECATED so the passage should be filtered out.
    out2 = await pp.apostprocess(passages)
    assert out2 == []


@pytest.mark.asyncio
async def test_contested_nugget_flags_passage(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    passages = [RetrievedPassage(source_id="doc-y", text="Tim Cook is CEO of Apple.")]
    await pp.apostprocess(passages)
    nuggets = await pp._store.backend.aget_nuggets_by_source("doc-y")
    for n in nuggets:
        updated = n.model_copy(
            update={
                "epistemic": EpistemicState(
                    status=LifecycleStatus.CONTESTED,
                    rank=n.epistemic.rank,
                    confidence=n.epistemic.confidence,
                ),
            }
        )
        await pp._store.backend.aupsert(updated)

    out = await pp.apostprocess(passages)
    assert len(out) == 1
    assert out[0].text.startswith("[DISPUTED] ")


@pytest.mark.asyncio
async def test_filter_deprecated_disabled_keeps_passage(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
        filter_deprecated=False,
        flag_contested=False,
    )
    passages = [RetrievedPassage(source_id="doc-z", text="Tim Cook is CEO of Apple.")]
    await pp.apostprocess(passages)
    nuggets = await pp._store.backend.aget_nuggets_by_source("doc-z")
    for n in nuggets:
        updated = n.model_copy(
            update={
                "epistemic": EpistemicState(
                    status=LifecycleStatus.DEPRECATED,
                    rank=n.epistemic.rank,
                    confidence=n.epistemic.confidence,
                ),
            }
        )
        await pp._store.backend.aupsert(updated)

    out = await pp.apostprocess(passages)
    assert len(out) == 1
    assert out[0].text == "Tim Cook is CEO of Apple."


# --- Cross-document conflict detection story --------------------------------


@pytest.mark.asyncio
async def test_second_query_triggers_contested(tmp_path):
    """The adoption story: Query 1 sees one fact; Query 2 introduces a conflicting
    fact; the postprocessor's shared cache picks up the conflict and flags at
    least one passage as DISPUTED.
    """
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "session.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )

    # Query 1: Tim Cook is CEO of Apple -> nothing in the cache yet -> ACTIVE.
    q1 = [RetrievedPassage(source_id="doc-1", text="Tim Cook is CEO of Apple.")]
    out1 = await pp.apostprocess(q1)
    assert len(out1) == 1
    assert not out1[0].text.startswith("[DISPUTED] ")

    # Query 2: Steve Jobs is CEO of Apple -> same (subject="Apple", pred="ceo")
    # overlapping validity + functional -> symmetric 1-vs-1 evidence -> CONTESTED.
    q2 = [RetrievedPassage(source_id="doc-2", text="Steve Jobs is CEO of Apple.")]
    out2 = await pp.apostprocess(q2)
    assert len(out2) == 1
    assert out2[0].text.startswith("[DISPUTED] "), (
        f"expected [DISPUTED] prefix, got: {out2[0].text!r}"
    )


# --- acreate_warm ------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_cache_classmethod_prepopulates(tmp_path):
    warm_docs = [
        Document(
            source_id="warm-1",
            text="Tim Cook is CEO of Apple.",
            source_date=datetime(2024, 6, 1, tzinfo=UTC),
        ),
    ]
    pp = await GovernancePostProcessor.acreate_warm(
        warm_cache=warm_docs,
        cache_path=tmp_path / "warm.db",
        extractor=RuleBasedExtractor(),
    )
    assert pp.count_cached_nuggets() > 0
    # And the warm doc's hash is already marked as seen.
    assert passage_hash("Tim Cook is CEO of Apple.") in pp._known_passage_hashes


@pytest.mark.asyncio
async def test_warm_cache_with_conflict_flags_retrieval(tmp_path):
    """Pre-populating the cache with one CEO triggers CONTESTED on first query."""
    warm_docs = [
        Document(
            source_id="warm-1",
            text="Tim Cook is CEO of Apple.",
            source_date=datetime(2024, 6, 1, tzinfo=UTC),
        ),
    ]
    pp = await GovernancePostProcessor.acreate_warm(
        warm_cache=warm_docs,
        cache_path=tmp_path / "warm.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    passages = [RetrievedPassage(source_id="q1", text="Steve Jobs is CEO of Apple.")]
    out = await pp.apostprocess(passages)
    assert len(out) == 1
    assert out[0].text.startswith("[DISPUTED] ")


# --- Bounded concurrency --------------------------------------------------


@pytest.mark.asyncio
async def test_extractions_bounded_by_semaphore(tmp_path):
    """Measure actual peak extractor concurrency under gathered ingestion.

    Now that the SQLite backend funnels every write through a single writer
    task (v0.2 Phase 4), the semaphore path is no longer backpressured by
    unpredictable "database is locked" retries on the shared connection. A
    0.1s sleep inside the extractor is therefore long enough to observe the
    real cap.
    """
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    class _SlowExtractor(BaseExtractor):
        def __init__(self) -> None:
            self._counter = 0

        async def aextract(
            self,
            text: str,
            *,
            context: str = "",
            source_id: str | None = None,
        ) -> list[ExtractionResult]:
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.1)
            finally:
                async with lock:
                    in_flight -= 1
            self._counter += 1
            nugget = Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject=f"s-{self._counter}",
                    predicate="is",
                    object=f"o-{self._counter}",
                    text=text,
                ),
                validity=ValidityInterval(start=datetime(2024, 1, 1, tzinfo=UTC)),
                epistemic=EpistemicState(confidence=0.9),
                provenance=(
                    ProvenanceRecord(
                        source_id=source_id or "stub", evidence_span=text
                    ),
                ),
                extraction_confidence=0.9,
            )
            return [ExtractionResult(nugget=nugget, confidence=0.9, rationale="stub")]

    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=_SlowExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
        max_extraction_concurrency=2,
    )
    passages = [
        RetrievedPassage(source_id=f"d-{i}", text=f"text-{i}") for i in range(8)
    ]
    await pp.apostprocess(passages)
    assert peak <= 2, f"expected peak <= 2, observed peak={peak}"
    assert peak >= 1, "extractor never observed any concurrency"
