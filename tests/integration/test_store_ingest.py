"""Integration tests for ``NuggetStore.aingest`` end-to-end.

Wires RuleBasedExtractor + pipeline + SQLite backend together and exercises
the scenarios from spec §2.3:

* Idempotency: ingesting the same document twice leaves the same row count.
* Cross-document conflict detection via ``fetch_existing_by_key``.
* The two-doc Pichai mini-corpus from phase-04 exit criteria.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
from nuggetindex.pipeline.constructor import Document
from nuggetindex.store.base import NuggetStore
from tests.fixtures import RuleBasedExtractor


@pytest.mark.asyncio
async def test_aingest_requires_extractor(tmp_db_path: Path) -> None:
    store = NuggetStore(tmp_db_path)
    doc = Document(source_id="d1", text="hello")
    with pytest.raises(RuntimeError, match="no extractor configured"):
        await store.aingest(doc)
    await store.aclose()


@pytest.mark.asyncio
async def test_aingest_end_to_end_rule_based(tmp_db_path: Path) -> None:
    store = NuggetStore(tmp_db_path, extractor=RuleBasedExtractor(source_id="d1"))
    doc = Document(
        source_id="d1",
        text="Sundar Pichai is CEO of Google.",
        source_date=datetime(2019, 1, 1, tzinfo=UTC),
    )
    result = await store.aingest(doc)
    assert result.nuggets_added >= 1
    assert await store.acount() >= 1
    await store.aclose()


def test_sync_ingest_wrapper(tmp_db_path: Path) -> None:
    store = NuggetStore(tmp_db_path, extractor=RuleBasedExtractor(source_id="d1"))
    doc = Document(
        source_id="d1",
        text="Sundar Pichai is CEO of Google.",
        source_date=datetime(2019, 1, 1, tzinfo=UTC),
    )
    result = store.ingest(doc)
    assert result.nuggets_added >= 1
    store.close()


class _FixedExtractor(BaseExtractor):
    """Extractor that returns a predefined set of results, verbatim."""

    def __init__(self, results: list[ExtractionResult]) -> None:
        self._results = results

    async def aextract(self, text: str, *, context: str = "") -> list[ExtractionResult]:
        return list(self._results)


def _extraction(
    *, subject: str, predicate: str, obj: str, sentence: str, source_id: str
) -> ExtractionResult:
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text=sentence),
        validity=ValidityInterval(start=datetime(1970, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=sentence),),
    )
    return ExtractionResult(nugget=n, confidence=0.95, rationale=None)


@pytest.mark.asyncio
async def test_aingest_is_idempotent(tmp_db_path: Path) -> None:
    store = NuggetStore(
        tmp_db_path,
        extractor=_FixedExtractor(
            [
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Pichai",
                    sentence="Pichai became CEO in 2019",
                    source_id="d1",
                )
            ]
        ),
    )
    doc = Document(
        source_id="d1",
        text="Pichai became CEO in 2019",
        source_date=datetime(2019, 6, 1, tzinfo=UTC),
    )
    r1 = await store.aingest(doc)
    count_after_1 = await store.acount()
    r2 = await store.aingest(doc)
    count_after_2 = await store.acount()
    # Row count must not grow on a repeat ingest (§2.3 idempotency invariant).
    assert count_after_1 == count_after_2 == 1
    assert r1.nuggets_added == 1
    # Second call dedupes the repeat away before upsert -> no added, no merged.
    assert r2.nuggets_added == 0
    await store.aclose()


@pytest.mark.asyncio
async def test_aingest_two_doc_pichai_mini_corpus(tmp_db_path: Path) -> None:
    """Phase-04 exit criteria mini-corpus.

    Doc 1: "Pichai is CEO of Google (2018)"
    Doc 2: "Sundar Pichai became Alphabet CEO in 2019"

    The pipeline should dedupe within-subject and run conflict detection on
    the Alphabet vs. Google streams independently. For this test we just
    verify that ingestion is idempotent across two docs, cross-document peers
    are visible (fetch_existing_by_key is wired), and all nuggets end in
    a legal lifecycle state.
    """
    extractor = _FixedExtractor(
        [
            _extraction(
                subject="Google",
                predicate="ceo",
                obj="Pichai",
                sentence="Pichai became CEO in 2018",
                source_id="d1",
            )
        ]
    )
    store = NuggetStore(tmp_db_path, extractor=extractor)
    await store.aingest(
        Document(
            source_id="d1",
            text="Pichai became CEO in 2018",
            source_date=datetime(2018, 6, 1, tzinfo=UTC),
        )
    )

    # Re-use the store with a new extractor output for doc 2.
    store._extractor = _FixedExtractor(
        [
            _extraction(
                subject="Alphabet",
                predicate="ceo",
                obj="Pichai",
                sentence="Pichai became CEO in 2019",
                source_id="d2",
            )
        ]
    )
    # Reset the lazily-built constructor so it picks up the new extractor.
    store._constructor = None

    await store.aingest(
        Document(
            source_id="d2",
            text="Pichai became CEO in 2019",
            source_date=datetime(2019, 6, 1, tzinfo=UTC),
        )
    )

    # Two separate keys -> two rows, both ACTIVE (no conflict: different subjects).
    assert await store.acount() == 2
    assert await store.acount(status=LifecycleStatus.ACTIVE) == 2
    await store.aclose()


@pytest.mark.asyncio
async def test_aingest_cross_document_conflict_via_fetch_by_key(tmp_db_path: Path) -> None:
    """When two docs contribute competing CEOs for the same company with
    overlapping validity and (1,1) evidence, both end CONTESTED because
    ``fetch_existing_by_key`` brings the prior nugget into conflict scope."""
    store = NuggetStore(
        tmp_db_path,
        extractor=_FixedExtractor(
            [
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Pichai",
                    sentence="Pichai became CEO in 2018",
                    source_id="d1",
                )
            ]
        ),
    )
    await store.aingest(
        Document(
            source_id="d1",
            text="Pichai became CEO in 2018",
            source_date=datetime(2018, 6, 1, tzinfo=UTC),
        )
    )

    store._extractor = _FixedExtractor(
        [
            _extraction(
                subject="Google",
                predicate="ceo",
                obj="Rival",
                sentence="Rival became CEO in 2019",
                source_id="d2",
            )
        ]
    )
    store._constructor = None
    await store.aingest(
        Document(
            source_id="d2",
            text="Rival became CEO in 2019",
            source_date=datetime(2019, 6, 1, tzinfo=UTC),
        )
    )

    # Both should be CONTESTED (symmetric 1,1-evidence on functional key).
    total = await store.acount()
    contested = await store.acount(status=LifecycleStatus.CONTESTED)
    assert total == 2
    assert contested == 2
    await store.aclose()
