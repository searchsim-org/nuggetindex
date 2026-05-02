"""Tests for ``DocumentConstructor`` orchestrator."""

from __future__ import annotations

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
from nuggetindex.core.schema import RelationSchema
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.pipeline.conflict import ConflictDetector
from nuggetindex.pipeline.constructor import Document, DocumentConstructor
from nuggetindex.pipeline.dedup import Deduplicator


class _StubExtractor(BaseExtractor):
    def __init__(self, results: list[ExtractionResult]) -> None:
        self._results = results

    async def aextract(self, text: str, *, context: str = "") -> list[ExtractionResult]:
        return list(self._results)


def _extraction(
    *,
    subject: str,
    predicate: str,
    obj: str,
    sentence: str,
    source_id: str = "doc",
    subject_type: str | None = "ORG",
    object_type: str | None = "PERSON",
) -> ExtractionResult:
    """Build an ExtractionResult that models a cooperating LLM extractor.

    Defaults to ``subject_type="ORG"`` / ``object_type="PERSON"`` to match
    the canonical CEO-predicate shape used throughout these tests; callers
    that want to exercise the spaCy-NER fallback path can pass
    ``subject_type=None, object_type=None``. These defaults make the
    pipeline tests independent of whichever entities happen to be in
    spaCy's ``en_core_web_sm`` model (the small model returns NONE for
    e.g. "Google" and "Elon Musk", which previously reduced the
    direction check to "reject" for any test that used those strings).
    """
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=sentence,
            subject_type=subject_type,
            object_type=object_type,
        ),
        validity=ValidityInterval(start=datetime(1970, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=sentence),),
    )
    return ExtractionResult(nugget=n, confidence=0.9, rationale=None)


def _build_constructor(results: list[ExtractionResult]) -> DocumentConstructor:
    schema = RelationSchema.default()
    return DocumentConstructor(
        extractor=_StubExtractor(results),
        schema=schema,
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(schema, judge=None),
    )


@pytest.mark.asyncio
async def test_basic_pipeline_end_to_end() -> None:
    constructor = _build_constructor(
        [
            _extraction(
                subject="Google",
                predicate="ceo",  # will be canonicalized
                obj="Sundar Pichai",
                sentence="Pichai became CEO in 2019",
            )
        ]
    )
    doc = Document(
        source_id="doc-1",
        text="Pichai became CEO in 2019",
        source_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    out = await constructor.aprocess(doc)
    assert len(out) == 1
    n = out[0]
    assert n.fact.predicate == "chiefExecutiveOfficer"
    assert n.validity.start.year == 2019
    assert n.epistemic.status == LifecycleStatus.ACTIVE


@pytest.mark.asyncio
async def test_pipeline_drops_duplicates() -> None:
    # Two extractions with same (canonical) key and identical objects.
    e1 = _extraction(
        subject="Google", predicate="ceo", obj="Sundar Pichai",
        sentence="Pichai became CEO in 2019",
    )
    e2 = _extraction(
        subject="Google", predicate="ceo", obj="Sundar Pichai",
        sentence="Sundar Pichai has been CEO since 2019",
    )
    constructor = _build_constructor([e1, e2])
    doc = Document(
        source_id="doc-1",
        text="...",
        source_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    out = await constructor.aprocess(doc)
    # Same validity.start (2019) -> same content hash -> dedup catches it OR
    # they are detected as duplicates by object similarity. Either way: 1.
    assert len(out) == 1


@pytest.mark.asyncio
async def test_pipeline_uses_fetch_existing_by_key() -> None:
    # Simulate a store where a prior Pichai nugget with 2 pieces of evidence
    # has already been persisted. Incoming single-evidence Page should be
    # deprecated by evidence asymmetry.
    prior = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="chiefExecutiveOfficer",
            object="Pichai",
            text="x",
        ),
        validity=ValidityInterval(start=datetime(2019, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(source_id="A", evidence_span="x", char_start=0, char_end=1),
            ProvenanceRecord(source_id="B", evidence_span="x", char_start=2, char_end=3),
        ),
    )

    async def fetch_by_key(key: tuple[str, str, str]) -> list[Nugget]:
        if key == prior.key:
            return [prior]
        return []

    e = _extraction(
        subject="Google", predicate="ceo", obj="Page",
        sentence="Page became CEO in 2015",
        source_id="doc-new",
    )
    constructor = _build_constructor([e])
    # Use source_date = 2020 so dates are sensible.
    doc = Document(
        source_id="doc-new",
        text="Page became CEO in 2015",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )

    out = await constructor.aprocess(doc, fetch_existing_by_key=fetch_by_key)
    # Output should contain the incoming nugget. Page has less evidence AND
    # is older than prior -> older wins -> incoming is DEPRECATED.
    ids = {n.fact.object for n in out}
    assert "Page" in ids
    page = next(n for n in out if n.fact.object == "Page")
    assert page.epistemic.status == LifecycleStatus.DEPRECATED


@pytest.mark.asyncio
async def test_pipeline_two_doc_succession() -> None:
    # Doc 1: Page was CEO until 2015. Doc 2: Pichai became CEO in 2015.
    # They should coexist (non-overlapping), both ACTIVE.
    schema = RelationSchema.default()
    constructor_1 = DocumentConstructor(
        extractor=_StubExtractor(
            [
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Page",
                    sentence="Page was CEO until 2014",
                    source_id="d1",
                )
            ]
        ),
        schema=schema,
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(schema, judge=None),
    )
    constructor_2 = DocumentConstructor(
        extractor=_StubExtractor(
            [
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Pichai",
                    sentence="Pichai became CEO in 2015",
                    source_id="d2",
                )
            ]
        ),
        schema=schema,
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(schema, judge=None),
    )

    d1 = Document(
        source_id="d1", text="Page was CEO until 2014",
        source_date=datetime(2014, 12, 31, tzinfo=UTC),
    )
    d2 = Document(
        source_id="d2", text="Pichai became CEO in 2015",
        source_date=datetime(2016, 1, 1, tzinfo=UTC),
    )

    state: list[Nugget] = []
    state.extend(await constructor_1.aprocess(d1, existing=state))
    state.extend(await constructor_2.aprocess(d2, existing=state))

    assert len(state) == 2
    for n in state:
        assert n.epistemic.status == LifecycleStatus.ACTIVE


@pytest.mark.asyncio
async def test_pipeline_ambiguous_year_reduces_confidence() -> None:
    # The sentence contains a bare year without a trigger phrase.
    e = _extraction(
        subject="Acme",
        predicate="industry",
        obj="widgets",
        sentence="Acme in the 2018 rankings",
    )
    # But "in 2018" matches _IN_DATE -> full confidence. We need a truly
    # bare-year case. Hack: rewrite the sentence.
    e.nugget.fact  # noqa: B018 -- no-op
    constructor = _build_constructor([e])
    # Build a doc whose source_date is some known value to verify fallback.
    source = datetime(2020, 1, 1, tzinfo=UTC)
    # Replace the extraction's nugget text to be a bare-year mention (no "in").
    # We do this by making a fresh extraction via _extraction.
    e_bare = _extraction(
        subject="Acme",
        predicate="industry",
        obj="widgets",
        sentence="An article about 2018 changes at Acme.",
    )
    constructor = _build_constructor([e_bare])
    doc = Document(source_id="doc", text="An article about 2018 changes at Acme.", source_date=source)
    out = await constructor.aprocess(doc)
    assert len(out) == 1
    # Ambiguous -> confidence multiplier 0.75.
    # Original extraction confidence was 1.0 (EpistemicState default).
    assert out[0].epistemic.confidence == pytest.approx(0.75)


def _placeholder_extraction(
    *,
    subject: str,
    predicate: str,
    obj: str,
    sentence: str,
    source_id: str = "doc",
    subject_type: str | None = "ORG",
    object_type: str | None = "PERSON",
) -> ExtractionResult:
    """Build an ExtractionResult with placeholder validity, so the pipeline's
    temporal-inference stage is the one that decides the final interval.
    This mirrors what ``RuleBasedExtractor`` emits in production.

    The ``subject_type`` / ``object_type`` defaults mirror ``_extraction``:
    most tests use the CEO predicate shape, so ORG/PERSON are the
    cooperating-LLM defaults. Pass ``None`` explicitly to exercise the
    spaCy-NER fallback path.
    """
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=sentence,
            subject_type=subject_type,
            object_type=object_type,
        ),
        validity=ValidityInterval.unknown(),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=sentence),),
    )
    return ExtractionResult(nugget=n, confidence=0.9, rationale=None)


@pytest.mark.asyncio
async def test_validity_known_true_when_date_extracted() -> None:
    # Evidence sentence contains an explicit "became ... in <date>" cue.
    # The temporal-inference stage parses the real calendar date and does
    # NOT fall back to source_date, so validity_known stays True.
    e = _placeholder_extraction(
        subject="Google",
        predicate="ceo",
        obj="Sundar Pichai",
        sentence="Sundar Pichai became CEO of Google in October 2015.",
    )
    constructor = _build_constructor([e])
    doc = Document(
        source_id="doc-1",
        text="Sundar Pichai became CEO of Google in October 2015.",
        source_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    out = await constructor.aprocess(doc)
    assert len(out) == 1
    n = out[0]
    # The extractor parsed a real calendar date from the evidence.
    assert n.validity.validity_known is True
    # Sanity-check: the start was NOT the source_date fallback.
    assert n.validity.start != doc.source_date
    assert n.validity.start.year == 2015


@pytest.mark.asyncio
async def test_validity_known_false_when_source_date_fallback() -> None:
    # Evidence sentence has no date cue at all. The temporal-inference stage
    # falls back to source_date as validity_start and marks validity_known
    # as False.
    e = _placeholder_extraction(
        subject="Google",
        predicate="ceo",
        obj="Sundar Pichai",
        sentence="Sundar Pichai is the CEO of Google.",
    )
    constructor = _build_constructor([e])
    source = datetime(2024, 6, 15, tzinfo=UTC)
    doc = Document(
        source_id="doc-1",
        text="Sundar Pichai is the CEO of Google.",
        source_date=source,
    )
    out = await constructor.aprocess(doc)
    assert len(out) == 1
    n = out[0]
    assert n.validity.validity_known is False
    assert n.validity.start == source


@pytest.mark.asyncio
async def test_pipeline_object_validator_drops_malformed_objects() -> None:
    # LLM extractors sometimes emit bare years or interrogative titles as
    # "objects" — those must be dropped BEFORE conflict detection so they
    # can't drive phantom CONTESTED flags. This mirrors the real-corpus
    # Mode-B failure described in the 0.3 fix.
    valid = _extraction(
        subject="Apple",
        predicate="chiefExecutiveOfficer",
        obj="Tim Cook",
        sentence="Tim Cook is CEO of Apple.",
    )
    bare_year = _extraction(
        subject="Apple",
        predicate="chiefExecutiveOfficer",
        obj="2000",
        sentence="Some article referenced 2000.",
    )
    interrogative = _extraction(
        subject="Apple",
        predicate="chiefExecutiveOfficer",
        obj="Следующие CEO Apple?",
        sentence="A Russian-language article title.",
    )
    constructor = _build_constructor([valid, bare_year, interrogative])
    doc = Document(
        source_id="doc-mixed",
        text="...",
        source_date=datetime(2024, 1, 1, tzinfo=UTC),
    )
    with pytest.warns(UserWarning, match="object_validator rejected 2"):
        out = await constructor.aprocess(doc)
    assert len(out) == 1
    assert out[0].fact.object == "Tim Cook"


def _typed_extraction(
    *,
    subject: str,
    predicate: str,
    obj: str,
    sentence: str,
    subject_type: str,
    object_type: str,
    source_id: str = "doc",
) -> ExtractionResult:
    """Like ``_extraction`` but emits LLM-style entity types on the
    FactTriple so fix-9 tests can exercise the flip / reject branches
    without depending on spaCy / ``en_core_web_sm`` being installed.
    """
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=sentence,
            subject_type=subject_type,
            object_type=object_type,
        ),
        validity=ValidityInterval(start=datetime(1970, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=sentence),),
    )
    return ExtractionResult(nugget=n, confidence=0.9, rationale=None)


@pytest.mark.asyncio
async def test_pipeline_entity_type_flip_and_reject() -> None:
    """Integration test for Fix B + Fix C / fix 9: direction flip and type
    reject driven by LLM-emitted types.

    This used to require spaCy + ``en_core_web_sm``; fix 9 makes the LLM
    types take priority so the test runs everywhere.
    """
    correct = _typed_extraction(
        subject="Google",
        predicate="chiefExecutiveOfficer",
        obj="Tim Cook",
        sentence="Tim Cook is CEO of Google.",
        subject_type="ORG",
        object_type="PERSON",
    )
    inverted = _typed_extraction(
        subject="Elon Musk",
        predicate="chiefExecutiveOfficer",
        obj="SpaceX",
        sentence="Elon Musk is CEO of SpaceX.",
        subject_type="PERSON",
        object_type="ORG",
    )
    bad_type = _typed_extraction(
        subject="Apple",
        predicate="chiefExecutiveOfficer",
        obj="A Day In The Life of Apple's CEO",
        sentence="A YouTube video title extracted as an object.",
        subject_type="ORG",
        object_type="WORK_OF_ART",
    )
    constructor = _build_constructor([correct, inverted, bad_type])
    doc = Document(
        source_id="doc-mixed",
        text="...",
        source_date=datetime(2024, 1, 1, tzinfo=UTC),
    )
    # Accept any UserWarning; we just want the ingest to complete.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("always")
        out = await constructor.aprocess(doc)

    # Build a lookup of (subject, object) tuples from the result.
    triples = {(n.fact.subject, n.fact.object) for n in out}

    # Correct triple persisted as-is.
    assert ("Google", "Tim Cook") in triples
    # Inverted triple flipped: subject should be the ORG, not the PERSON.
    assert ("SpaceX", "Elon Musk") in triples
    # After flipping, types must be swapped in lockstep with subject/object.
    flipped = next(
        n for n in out if (n.fact.subject, n.fact.object) == ("SpaceX", "Elon Musk")
    )
    assert flipped.fact.subject_type == "ORG"
    assert flipped.fact.object_type == "PERSON"
    # Bad-type triple must be rejected.
    assert not any(
        n.fact.object == "A Day In The Life of Apple's CEO" for n in out
    )


@pytest.mark.asyncio
async def test_pipeline_with_quality_gate_filters_low_confidence() -> None:
    from nuggetindex.extractors.quality import QualityGate

    # Two extractions: one above accept threshold, one below.
    high = _extraction(
        subject="Google", predicate="ceo", obj="Pichai",
        sentence="Pichai became CEO in 2019",
    )
    high_ref = ExtractionResult(nugget=high.nugget, confidence=0.95, rationale=None)

    low = _extraction(
        subject="Foo", predicate="ceo", obj="Bar",
        sentence="Bar is CEO of Foo",
    )
    low_ref = ExtractionResult(nugget=low.nugget, confidence=0.4, rationale=None)

    stub = _StubExtractor([high_ref, low_ref])
    schema = RelationSchema.default()
    qg = QualityGate(stub, accept_threshold=0.85, review_threshold=0.6)
    constructor = DocumentConstructor(
        extractor=stub,
        schema=schema,
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(schema, judge=None),
        quality_gate=qg,
    )
    doc = Document(
        source_id="doc",
        text="...",
        source_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    out = await constructor.aprocess(doc)
    assert len(out) == 1
    assert out[0].fact.subject == "Google"
