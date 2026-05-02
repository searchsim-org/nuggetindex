"""Tests for ``nuggetindex.audit.heuristics.ner`` (Task 2.3).

Tests that actually require spaCy + ``en_core_web_sm`` are guarded by
``skipif(not ner_available())`` so the suite stays green in minimal dev
environments. The empty-text fast path is exercised unconditionally.
"""

from __future__ import annotations

import pytest

from nuggetindex.audit.heuristics import Entity, extract_entities, ner_available


def test_ner_available_reports_bool() -> None:
    assert isinstance(ner_available(), bool)


def test_extract_entities_empty_text_returns_empty() -> None:
    assert extract_entities("") == []


@pytest.mark.skipif(
    not ner_available(), reason="spaCy + en_core_web_sm not installed"
)
def test_extract_entities_finds_org() -> None:
    text = "Microsoft acquired GitHub in 2018."
    result = extract_entities(text)
    org_texts = {e.text for e in result if e.label == "ORG"}
    # spaCy's NER is stochastic at the margins but both names are reliably
    # tagged ORG in en_core_web_sm.
    assert "Microsoft" in org_texts
    assert "GitHub" in org_texts
    # Sanity: all returned items are Entity instances with non-empty spans.
    for ent in result:
        assert isinstance(ent, Entity)
        assert ent.start_char < ent.end_char


@pytest.mark.skipif(
    not ner_available(), reason="spaCy + en_core_web_sm not installed"
)
def test_extract_entities_filters_out_unwanted_labels() -> None:
    text = "The year 2015 was pivotal for Google."
    result = extract_entities(text)
    texts = {e.text for e in result}
    labels = {e.label for e in result}
    # Google should surface as ORG.
    assert "Google" in texts
    # 2015 is a DATE and must NOT be returned (DATE isn't in the wanted set).
    assert "2015" not in texts
    assert "DATE" not in labels
