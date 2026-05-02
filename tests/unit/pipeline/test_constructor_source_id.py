"""Tests for ``DocumentConstructor.aprocess`` forwarding ``source_id``.

The constructor inspects the extractor's ``aextract`` signature at init time
and forwards ``doc.source_id`` only to extractors that accept it. Extractors
predating 0.2 continue to receive the legacy signature.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.core.schema import RelationSchema
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.extractors.quality import QualityGate
from nuggetindex.pipeline.conflict import ConflictDetector
from nuggetindex.pipeline.constructor import Document, DocumentConstructor
from nuggetindex.pipeline.dedup import Deduplicator
from tests.fixtures import RuleBasedExtractor


def _make_result(source_id: str) -> ExtractionResult:
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="Google", predicate="is", object="company", text="Google is a company."),
        validity=ValidityInterval(start=datetime(1970, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span="Google is a company."),),
    )
    return ExtractionResult(nugget=n, confidence=0.9)


class _ModernStub(BaseExtractor):
    """Extractor that honours ``source_id``."""

    def __init__(self) -> None:
        self.last_source_id: str | None = None

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[ExtractionResult]:
        self.last_source_id = source_id
        return [_make_result(source_id or "stub-default")]


class _LegacyStub(BaseExtractor):
    """Extractor predating the 0.2 convention."""

    def __init__(self) -> None:
        self.called = False

    async def aextract(self, text: str, *, context: str = "") -> list[ExtractionResult]:
        self.called = True
        return [_make_result("legacy-default")]


def _build(extractor: BaseExtractor, *, quality_gate: QualityGate | None = None) -> DocumentConstructor:
    schema = RelationSchema.default()
    return DocumentConstructor(
        extractor=extractor,
        schema=schema,
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(schema),
        quality_gate=quality_gate,
    )


@pytest.mark.asyncio
async def test_constructor_passes_source_id_to_rule_based_extractor() -> None:
    ctor = _build(RuleBasedExtractor())
    doc = Document(
        source_id="doc-xyz",
        text="Google is a company.",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    nuggets = await ctor.aprocess(doc)
    assert nuggets, "expected at least one nugget"
    for n in nuggets:
        assert all(p.source_id == "doc-xyz" for p in n.provenance)


@pytest.mark.asyncio
async def test_constructor_forwards_source_id_to_modern_extractor() -> None:
    ex = _ModernStub()
    ctor = _build(ex)
    doc = Document(
        source_id="doc-42",
        text="Google is a company.",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    await ctor.aprocess(doc)
    assert ex.last_source_id == "doc-42"


@pytest.mark.asyncio
async def test_constructor_skips_source_id_for_legacy_extractor() -> None:
    ex = _LegacyStub()
    # If the constructor tries to call legacy extractor with source_id=, this
    # would raise TypeError; the pipeline should gate on signature inspection.
    ctor = _build(ex)
    doc = Document(
        source_id="doc-42",
        text="Google is a company.",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    nuggets = await ctor.aprocess(doc)
    assert ex.called is True
    assert nuggets, "legacy extractor should still produce nuggets"


@pytest.mark.asyncio
async def test_constructor_forwards_source_id_through_quality_gate(tmp_path) -> None:
    ex = _ModernStub()
    gate = QualityGate(
        ex,
        accept_threshold=0.5,
        review_threshold=0.1,
        review_queue_path=tmp_path / "review.jsonl",
    )
    ctor = _build(ex, quality_gate=gate)
    doc = Document(
        source_id="doc-gate",
        text="Google is a company.",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    await ctor.aprocess(doc)
    assert ex.last_source_id == "doc-gate"
