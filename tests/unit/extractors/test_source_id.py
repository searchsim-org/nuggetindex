"""Tests for the ``source_id`` convention on ``BaseExtractor`` subclasses.

The convention is additive: subclasses may optionally declare a ``source_id``
keyword parameter on ``aextract``. Call sites in the pipeline detect support
via :func:`accepts_source_id` and only forward the kwarg when the subclass
actually accepts it. Subclasses predating the convention keep working.
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
from nuggetindex.extractors.base import (
    BaseExtractor,
    ExtractionResult,
    accepts_source_id,
)
from tests.fixtures import RuleBasedExtractor


def _make_nugget(source_id: str = "legacy-stub") -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="X", predicate="p", object="Y", text="X p Y"),
        validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span="X p Y"),),
    )


class _LegacyExtractor(BaseExtractor):
    """Subclass that predates the source_id keyword argument."""

    async def aextract(self, text, *, context=""):
        return [ExtractionResult(nugget=_make_nugget(), confidence=0.9)]


class _ModernExtractor(BaseExtractor):
    """Subclass that honours source_id."""

    async def aextract(self, text, *, context="", source_id=None):
        sid = source_id or "modern-default"
        return [ExtractionResult(nugget=_make_nugget(sid), confidence=0.9)]


@pytest.mark.asyncio
async def test_legacy_extractor_ignores_source_id() -> None:
    ex = _LegacyExtractor()
    results = await ex.aextract("text")
    assert results[0].nugget.provenance[0].source_id == "legacy-stub"


@pytest.mark.asyncio
async def test_modern_extractor_uses_source_id() -> None:
    ex = _ModernExtractor()
    results = await ex.aextract("text", source_id="doc-42")
    assert results[0].nugget.provenance[0].source_id == "doc-42"


@pytest.mark.asyncio
async def test_modern_extractor_default_when_unset() -> None:
    ex = _ModernExtractor()
    results = await ex.aextract("text")
    assert results[0].nugget.provenance[0].source_id == "modern-default"


def test_accepts_source_id_detects_modern_subclass() -> None:
    assert accepts_source_id(_ModernExtractor()) is True


def test_accepts_source_id_detects_legacy_subclass() -> None:
    assert accepts_source_id(_LegacyExtractor()) is False


def test_accepts_source_id_rule_based_extractor() -> None:
    # After Task 2.2, RuleBasedExtractor declares the kwarg.
    assert accepts_source_id(RuleBasedExtractor()) is True


@pytest.mark.asyncio
async def test_rule_based_extractor_honours_source_id() -> None:
    ex = RuleBasedExtractor()
    assert accepts_source_id(ex)
    results = await ex.aextract("Google is a company.", source_id="wiki-google")
    assert results, "expected at least one extraction"
    for r in results:
        assert all(p.source_id == "wiki-google" for p in r.nugget.provenance)


@pytest.mark.asyncio
async def test_rule_based_extractor_falls_back_to_default_source_id() -> None:
    ex = RuleBasedExtractor()
    results = await ex.aextract("Google is a company.")
    assert results, "expected at least one extraction"
    for r in results:
        assert all(p.source_id == "rule-based" for p in r.nugget.provenance)
