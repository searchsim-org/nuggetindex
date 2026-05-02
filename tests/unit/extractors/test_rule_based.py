"""Tests for the zero-LLM rule-based extractor."""

from __future__ import annotations

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.extractors.base import ExtractionResult
from tests.fixtures import RuleBasedExtractor


@pytest.mark.asyncio
async def test_simple_is_pattern() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Google is a company.")
    assert len(results) >= 1
    r = results[0]
    assert "Google" in r.nugget.fact.subject
    assert r.nugget.fact.predicate == "is"
    assert "company" in r.nugget.fact.object


@pytest.mark.asyncio
async def test_confidence_in_range() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Apple was founded in 1976.")
    assert len(results) >= 1
    for r in results:
        assert 0.3 <= r.confidence <= 0.6


@pytest.mark.asyncio
async def test_multiple_sentences_yield_multiple_nuggets() -> None:
    ext = RuleBasedExtractor()
    text = "Google is a company. Apple was founded in 1976. Sundar Pichai is CEO of Google."
    results = await ext.aextract(text)
    assert len(results) >= 3


@pytest.mark.asyncio
async def test_founded_pattern() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Apple was founded in 1976.")
    matches = [
        r for r in results if r.nugget.fact.predicate == "founded"
    ]
    assert matches
    r = matches[0]
    assert r.nugget.fact.subject == "Apple"
    assert r.nugget.fact.object == "1976"


@pytest.mark.asyncio
async def test_ceo_of_pattern() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Sundar Pichai is CEO of Google.")
    ceos = [r for r in results if r.nugget.fact.predicate == "ceo"]
    assert ceos
    r = ceos[0]
    # "X CEO of Y" → subject=Y, predicate="ceo", object=X
    assert r.nugget.fact.subject == "Google"
    assert r.nugget.fact.object == "Sundar Pichai"


@pytest.mark.asyncio
async def test_returns_extraction_result_objects() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Google is a company.")
    for r in results:
        assert isinstance(r, ExtractionResult)
        assert r.nugget.kind == NuggetKind.SEMANTIC_FACT


@pytest.mark.asyncio
async def test_newline_splits_sentences() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Google is a company\nApple is a fruit")
    subjects = {r.nugget.fact.subject for r in results}
    assert "Google" in subjects
    assert "Apple" in subjects


@pytest.mark.asyncio
async def test_empty_text_yields_empty() -> None:
    ext = RuleBasedExtractor()
    assert await ext.aextract("") == []
    assert await ext.aextract("   ") == []


@pytest.mark.asyncio
async def test_no_match_sentence_yields_nothing() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Hello there friend.")
    assert results == []


@pytest.mark.asyncio
async def test_validity_start_is_tz_aware() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Google is a company.")
    assert results[0].nugget.validity.start.tzinfo is not None
