"""Tests for the store-scoped :class:`AliasResolver` (fix 10).

The resolver lives for the lifetime of a :class:`NuggetStore` instance and
is seeded on first ingest from every distinct subject + object string
already in the backend. That way "Microsoft" in doc A and "Microsoft
Corporation" in doc B collapse to a single canonical.

The LLM extractor is intentionally out-of-scope here; we exercise two
layers:

1. ``NuggetStore._ensure_alias_resolver`` correctly seeds its pool from
   ``adistinct_entities`` after nuggets have been ``aadd``-ed.
2. ``DocumentConstructor.aprocess`` accepts a pre-built resolver and
   reuses it across documents so mentions accumulate.

Both layers are gated on sklearn availability because the tests assert
behaviour that depends on the normalized-alias tier; without sklearn
that tier still works (tier 2 is pure Python) but the gating is kept
consistent with :mod:`tests.unit.pipeline.test_aliases`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.store import NuggetStore


def _make_nugget(subj: str, obj: str, *, source_id: str = "doc") -> Nugget:
    now = datetime.now(tz=UTC)
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subj,
            predicate="chiefExecutiveOfficer",
            object=obj,
            text=f"{obj} runs {subj}",
        ),
        validity=ValidityInterval(start=now, end=None),
        epistemic=EpistemicState(
            status=LifecycleStatus.ACTIVE,
            rank=EpistemicRank.NORMAL,
            confidence=0.9,
        ),
        provenance=(
            ProvenanceRecord(
                source_id=source_id,
                evidence_span="...",
                char_start=0,
                char_end=10,
                created_at=now,
            ),
        ),
        extraction_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_ensure_alias_resolver_seeds_from_backend(tmp_path: Path) -> None:
    """After ``aadd``-ing a nugget, the lazy-inited store-scoped resolver
    must have the nugget's subject + object in its canonical pool."""
    store = NuggetStore(db_path=tmp_path / "s.db")
    try:
        await store.aadd(_make_nugget("Microsoft", "Satya Nadella"))
        resolver = await store._ensure_alias_resolver()
        pool = set(resolver.pool())
        assert "Microsoft" in pool
        assert "Satya Nadella" in pool
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_ensure_alias_resolver_is_idempotent(tmp_path: Path) -> None:
    """Calling ``_ensure_alias_resolver`` twice returns the same instance;
    new mentions added to the resolver between calls are preserved."""
    store = NuggetStore(db_path=tmp_path / "s.db")
    try:
        await store.aadd(_make_nugget("Microsoft", "Satya Nadella"))
        r1 = await store._ensure_alias_resolver()
        r1.resolve("SomeOtherOrg")
        r2 = await store._ensure_alias_resolver()
        assert r1 is r2
        assert "SomeOtherOrg" in r2.pool()
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_cross_doc_alias_merges_microsoft_corporation(
    tmp_path: Path,
) -> None:
    """End-to-end test at the resolver level: seed the store-scoped resolver
    from an already-ingested Microsoft mention, then resolving
    ``"Microsoft Corporation"`` through that same resolver (as the pipeline
    would for a later doc) must NOT introduce a second canonical -- it
    must fold back to "Microsoft"."""
    store = NuggetStore(db_path=tmp_path / "s.db")
    try:
        # Doc A: direct aadd to populate the backend with the canonical form.
        await store.aadd(_make_nugget("Microsoft", "Satya Nadella", source_id="doc-a"))
        # Store-scoped resolver should now seed from the backend and already
        # know "Microsoft" as a canonical.
        resolver = await store._ensure_alias_resolver()
        # Doc B: the pipeline would route "Microsoft Corporation" through
        # this same resolver. Legal-suffix normalization must catch it.
        res = resolver.resolve("Microsoft Corporation")
        assert res.canonical == "Microsoft"
        # Pool did not grow with a duplicate canonical.
        assert "Microsoft Corporation" not in resolver.pool()
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_document_constructor_accepts_external_resolver(
    tmp_path: Path,
) -> None:
    """``DocumentConstructor.aprocess`` must use the supplied resolver
    instead of instantiating its own, so mentions accumulate across calls.
    """
    from nuggetindex.core.schema import RelationSchema
    from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
    from nuggetindex.pipeline.aliases import AliasResolver
    from nuggetindex.pipeline.conflict import ConflictDetector
    from nuggetindex.pipeline.constructor import Document, DocumentConstructor
    from nuggetindex.pipeline.dedup import Deduplicator

    class _StubExtractor(BaseExtractor):
        def __init__(self, results: list[ExtractionResult]) -> None:
            self._results = results

        async def aextract(
            self, text: str, *, context: str = "",
        ) -> list[ExtractionResult]:
            return list(self._results)

    def _result(subject: str, obj: str, sentence: str) -> ExtractionResult:
        n = Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject=subject,
                predicate="chiefExecutiveOfficer",
                object=obj,
                text=sentence,
                subject_type="ORG",
                object_type="PERSON",
            ),
            validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
            epistemic=EpistemicState(),
            provenance=(
                ProvenanceRecord(source_id="d", evidence_span=sentence),
            ),
        )
        return ExtractionResult(nugget=n, confidence=0.9, rationale=None)

    schema = RelationSchema.default()
    shared_resolver = AliasResolver()

    # Doc A introduces "Microsoft".
    ctor_a = DocumentConstructor(
        extractor=_StubExtractor(
            [_result("Microsoft", "Satya Nadella", "Satya Nadella runs Microsoft.")]
        ),
        schema=schema,
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(schema, judge=None),
    )
    doc_a = Document(
        source_id="doc-a",
        text="Satya Nadella runs Microsoft.",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    out_a = await ctor_a.aprocess(doc_a, alias_resolver=shared_resolver)
    assert len(out_a) == 1
    assert "Microsoft" in shared_resolver.pool()

    # Doc B references the same org as "Microsoft Corporation".
    ctor_b = DocumentConstructor(
        extractor=_StubExtractor(
            [
                _result(
                    "Microsoft Corporation",
                    "Satya Nadella",
                    "Satya Nadella runs Microsoft Corporation.",
                )
            ]
        ),
        schema=schema,
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(schema, judge=None),
    )
    doc_b = Document(
        source_id="doc-b",
        text="Satya Nadella runs Microsoft Corporation.",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    out_b = await ctor_b.aprocess(doc_b, alias_resolver=shared_resolver)
    # The alias resolver should have folded "Microsoft Corporation" back to
    # "Microsoft" -- so the emitted nugget's subject is the canonical.
    assert len(out_b) == 1
    assert out_b[0].fact.subject == "Microsoft"
    # And the pool has not grown a duplicate.
    assert "Microsoft Corporation" not in shared_resolver.pool()


@pytest.mark.asyncio
async def test_backend_adistinct_entities_returns_union(tmp_path: Path) -> None:
    """Smoke test for the new backend helper that ``_ensure_alias_resolver``
    depends on. Returns the de-duplicated union of all subject + object
    strings currently stored."""
    store = NuggetStore(db_path=tmp_path / "s.db")
    try:
        await store.aadd(_make_nugget("Microsoft", "Satya Nadella", source_id="a"))
        await store.aadd(_make_nugget("Google", "Sundar Pichai", source_id="b"))
        got = set(await store.backend.adistinct_entities())
        assert got == {"Microsoft", "Satya Nadella", "Google", "Sundar Pichai"}
    finally:
        await store.aclose()
