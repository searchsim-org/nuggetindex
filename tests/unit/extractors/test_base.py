"""Tests for the BaseExtractor ABC and ExtractionResult shape."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult


def _nugget() -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="A", predicate="p", object="B", text="A p B"),
        validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span="A p B"),),
    )


def test_extraction_result_shape() -> None:
    r = ExtractionResult(nugget=_nugget(), confidence=0.9, rationale="clear assertion")
    assert r.confidence == 0.9
    assert r.rationale == "clear assertion"
    assert r.nugget.fact.subject == "A"


def test_extraction_result_default_rationale_is_none() -> None:
    r = ExtractionResult(nugget=_nugget(), confidence=0.5)
    assert r.rationale is None


def test_extraction_result_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(nugget=_nugget(), confidence=1.5)
    with pytest.raises(ValidationError):
        ExtractionResult(nugget=_nugget(), confidence=-0.1)


def test_base_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseExtractor()  # type: ignore[abstract]


def test_base_extractor_sync_wrapper_uses_aextract() -> None:
    class _Dummy(BaseExtractor):
        async def aextract(self, text: str, *, context: str = "") -> list[ExtractionResult]:
            return [ExtractionResult(nugget=_nugget(), confidence=0.4)]

    results = _Dummy().extract("hi")
    assert len(results) == 1
    assert results[0].confidence == 0.4
