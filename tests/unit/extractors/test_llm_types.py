"""Tests for LLM-emitted entity types flowing through :class:`LLMExtractor`
to the resulting :class:`FactTriple` (fix 8).

Uses a stubbed structured-output client so these tests run without network
access or API keys.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from nuggetindex.extractors.clients.base import LLMConfig
from nuggetindex.extractors.llm import (
    ExtractionPayload,
    LLMExtractor,
    _TripleOut,
)


class _StubClient:
    """Deterministic LLM client that returns a pre-built ExtractionPayload."""

    def __init__(self, payload: ExtractionPayload) -> None:
        self._payload = payload
        self.achat_structured = AsyncMock(side_effect=self._return)

    async def _return(
        self, messages: list[dict[str, Any]], response_model: type[BaseModel]
    ) -> BaseModel:
        assert response_model is ExtractionPayload
        return self._payload


def _cfg() -> LLMConfig:
    return LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")


@pytest.mark.asyncio
async def test_llm_extractor_threads_types_to_fact_triple() -> None:
    """When the LLM emits types, they must land on ``FactTriple``."""
    payload = ExtractionPayload(
        facts=[
            _TripleOut(
                subject="Sundar Pichai",
                predicate="chiefExecutiveOfficer",
                object="Google",
                evidence_span="Sundar Pichai is the CEO of Google.",
                confidence=0.9,
                subject_type="PERSON",
                object_type="ORG",
            )
        ]
    )
    stub = _StubClient(payload)
    ex = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]

    results = await ex.aextract("Sundar Pichai is the CEO of Google.")
    assert len(results) == 1
    n = results[0].nugget
    assert n.fact.subject_type == "PERSON"
    assert n.fact.object_type == "ORG"


@pytest.mark.asyncio
async def test_llm_extractor_missing_types_default_to_none() -> None:
    """When the LLM omits types, ``FactTriple``'s fields stay ``None`` so
    the pipeline falls back to spaCy NER (or skips the check entirely)."""
    payload = ExtractionPayload(
        facts=[
            _TripleOut(
                subject="Google",
                predicate="chiefExecutiveOfficer",
                object="Sundar Pichai",
                evidence_span="Sundar Pichai is CEO of Google.",
                confidence=0.9,
                # subject_type / object_type deliberately omitted
            )
        ]
    )
    stub = _StubClient(payload)
    ex = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]

    results = await ex.aextract("Sundar Pichai is CEO of Google.")
    assert len(results) == 1
    n = results[0].nugget
    assert n.fact.subject_type is None
    assert n.fact.object_type is None


@pytest.mark.asyncio
async def test_llm_extractor_types_survive_multiple_facts() -> None:
    """Per-fact types must be threaded through independently; one bad
    omission on fact B must not leak types from fact A."""
    payload = ExtractionPayload(
        facts=[
            _TripleOut(
                subject="Apple",
                predicate="foundedIn",
                object="1976",
                evidence_span="Apple was founded in 1976.",
                confidence=0.95,
                subject_type="ORG",
                object_type="DATE",
            ),
            _TripleOut(
                subject="Apple",
                predicate="headquarteredIn",
                object="Cupertino",
                evidence_span="Apple HQ is in Cupertino.",
                confidence=0.9,
                # type fields missing on purpose
            ),
        ]
    )
    stub = _StubClient(payload)
    ex = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]

    results = await ex.aextract("irrelevant")
    assert len(results) == 2
    # First fact carries types.
    assert results[0].nugget.fact.subject_type == "ORG"
    assert results[0].nugget.fact.object_type == "DATE"
    # Second fact does not; types stay ``None`` (no leakage).
    assert results[1].nugget.fact.subject_type is None
    assert results[1].nugget.fact.object_type is None
