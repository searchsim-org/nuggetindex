"""Unit tests for :class:`nuggetindex.extractors.trigger.TriggerExtractor`.

The trigger extractor is the LLM-free default for the CLI. These tests
lock its two most-important behaviours: role-succession and entity-rename
patterns produce ``Nugget`` records with the expected canonical
predicate, and every emission carries the flat 0.5 confidence so callers
can filter confidently.

``test_no_llm_calls`` is a belt-and-braces assertion that running this
extractor never patches httpx / openai -- the point of the fast path is
that it runs fully offline.
"""

from __future__ import annotations

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.schema import RelationSchema
from nuggetindex.extractors.base import ExtractionResult
from nuggetindex.extractors.trigger import TriggerExtractor


@pytest.mark.asyncio
async def test_extracts_role_succession() -> None:
    """``became CEO of`` -> canonical predicate ``chiefExecutiveOfficer``."""
    ext = TriggerExtractor()
    results = await ext.aextract("Satya Nadella became CEO of Microsoft in 2014.")
    assert len(results) >= 1
    schema = RelationSchema.default()
    preds = {schema.canonicalize(r.nugget.fact.predicate) for r in results}
    assert "chiefExecutiveOfficer" in preds

    ceo = next(
        r
        for r in results
        if schema.canonicalize(r.nugget.fact.predicate) == "chiefExecutiveOfficer"
    )
    # Role direction is (ORG, role, PERSON).
    assert "Microsoft" in ceo.nugget.fact.subject
    assert "Satya Nadella" in ceo.nugget.fact.object


@pytest.mark.asyncio
async def test_extracts_entity_rename() -> None:
    """``renamed to`` -> predicate ``renamedTo``."""
    ext = TriggerExtractor()
    results = await ext.aextract("Twitter Inc. was renamed to X Corp. in 2023.")
    renames = [r for r in results if r.nugget.fact.predicate == "renamedTo"]
    assert len(renames) >= 1
    r = renames[0]
    assert "Twitter" in r.nugget.fact.subject
    assert "X Corp" in r.nugget.fact.object


@pytest.mark.asyncio
async def test_emits_source_id_when_kwarg_provided() -> None:
    """``source_id="doc1"`` stamps the provenance record."""
    ext = TriggerExtractor()
    results = await ext.aextract(
        "Microsoft acquired LinkedIn for $26.2 billion.",
        source_id="doc1",
    )
    assert results, "expected at least one acquisition trigger"
    for r in results:
        for prov in r.nugget.provenance:
            assert prov.source_id == "doc1"


@pytest.mark.asyncio
async def test_confidence_is_0_5() -> None:
    """All emissions carry the canonical ``0.5`` extraction confidence."""
    ext = TriggerExtractor()
    # Independent paragraphs -- the upstream ``scan_triggers`` overlap
    # filter can absorb adjacent trigger matches into the object span of
    # a greedy regex, so we keep the inputs paragraph-separated to
    # guarantee at least three distinct emissions here.
    results = await ext.aextract(
        "Satya Nadella became CEO of Microsoft in 2014.\n\n"
        "Twitter Inc. was renamed to X Corp. in 2023.\n\n"
        "Microsoft acquired LinkedIn."
    )
    assert len(results) >= 3
    for r in results:
        assert r.confidence == 0.5
        assert r.nugget.extraction_confidence == 0.5
        assert r.nugget.kind == NuggetKind.SEMANTIC_FACT
        assert isinstance(r, ExtractionResult)


@pytest.mark.asyncio
async def test_no_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing + running TriggerExtractor must not touch httpx/openai.

    If either module is importable, set its surface attrs to sentinels
    that raise on access so any accidental call blows up loudly. We
    import ``openai`` BEFORE patching ``httpx`` because ``openai``'s
    module-level code subclasses ``httpx.Client`` at import time — if a
    patched ``httpx.Client`` is already in place, that import raises.
    (Order swap: openai first, httpx second.)
    """

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("TriggerExtractor must not invoke HTTP / LLM clients")

    try:
        import openai

        monkeypatch.setattr(openai, "AsyncOpenAI", _boom, raising=False)
        monkeypatch.setattr(openai, "OpenAI", _boom, raising=False)
    except ImportError:  # pragma: no cover -- openai is optional
        pass
    try:
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _boom, raising=False)
        monkeypatch.setattr(httpx, "Client", _boom, raising=False)
    except ImportError:  # pragma: no cover -- httpx is a runtime dep
        pass

    ext = TriggerExtractor()
    results = await ext.aextract("Microsoft acquired LinkedIn for $26.2 billion.")
    assert results, "expected at least one trigger hit"


@pytest.mark.asyncio
async def test_empty_text_returns_empty() -> None:
    ext = TriggerExtractor()
    assert await ext.aextract("") == []
    assert await ext.aextract("   ") == []


@pytest.mark.asyncio
async def test_no_triggers_returns_empty() -> None:
    """Text with no trigger-verb hits yields nothing."""
    ext = TriggerExtractor()
    results = await ext.aextract("The weather was pleasant on Tuesday.")
    assert results == []


@pytest.mark.asyncio
async def test_subject_type_populated_when_spacy_available() -> None:
    """When spaCy is installed, the subject gets a non-None NER label.

    Gated on ``importorskip`` so this test is a no-op in the default
    (no-``[doctor]``-extra) CI environment.
    """
    pytest.importorskip("spacy")
    # Also need the English model; skip if unavailable.
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("en_core_web_sm model not installed")

    ext = TriggerExtractor()
    results = await ext.aextract("Microsoft acquired LinkedIn for $26.2 billion.")
    acquired = [r for r in results if r.nugget.fact.predicate == "acquired"]
    assert acquired
    assert acquired[0].nugget.fact.subject_type is not None
