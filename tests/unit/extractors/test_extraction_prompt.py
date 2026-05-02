"""Tests for the rewritten default extraction prompt (findings-F1/F2/F3).

Covers:
  * Static assertions on the prompt file contents (canonical-predicate guidance,
    camelCase exemplar, one-fact-per-pair rule).
  * Stubbed-LLM extractor behaviour when the model cooperates with the new
    prompt: canonical predicates flow through to emitted nuggets, and the
    compound-subject pattern expands to one nugget per subject.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from nuggetindex.extractors.clients.base import LLMConfig
from nuggetindex.extractors.llm import (
    _PROMPT_PATH,
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


def test_prompt_contains_canonical_predicate_guidance() -> None:
    """The default prompt must mention canonical-predicate normalisation."""
    text = _PROMPT_PATH.read_text()
    lower = text.lower()
    assert "canonical" in lower
    assert "chiefExecutiveOfficer" in text
    assert "camelcase" in lower
    assert "one fact per" in lower or "separate fact" in lower


@pytest.mark.asyncio
async def test_extractor_returns_canonical_predicate_when_llm_cooperates() -> None:
    """Given a cooperating LLM, canonical predicates flow through to nuggets."""
    payload = ExtractionPayload(
        facts=[
            _TripleOut(
                subject="Sundar Pichai",
                predicate="chiefExecutiveOfficer",
                object="Google",
                evidence_span="Sundar Pichai é o CEO do Google",
                confidence=0.95,
            ),
        ]
    )
    stub = _StubClient(payload)
    ex = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]

    results = await ex.aextract("Sundar Pichai é o CEO do Google")
    assert results
    assert results[0].nugget.fact.predicate == "chiefExecutiveOfficer"
    assert results[0].nugget.fact.subject == "Sundar Pichai"
    assert results[0].nugget.fact.object == "Google"


@pytest.mark.asyncio
async def test_extractor_expands_compound_subjects_to_separate_facts() -> None:
    """Given a compound-subject text, a cooperating LLM emits one fact per
    subject and the extractor faithfully produces one nugget per fact."""
    payload = ExtractionPayload(
        facts=[
            _TripleOut(
                subject="Apple",
                predicate="cooperatesOn",
                object="standard X",
                evidence_span=(
                    "Apple, Google, Microsoft i Mozilla współpracują przy standardzie X"
                ),
                confidence=0.9,
            ),
            _TripleOut(
                subject="Google",
                predicate="cooperatesOn",
                object="standard X",
                evidence_span="...",
                confidence=0.9,
            ),
            _TripleOut(
                subject="Microsoft",
                predicate="cooperatesOn",
                object="standard X",
                evidence_span="...",
                confidence=0.9,
            ),
            _TripleOut(
                subject="Mozilla",
                predicate="cooperatesOn",
                object="standard X",
                evidence_span="...",
                confidence=0.9,
            ),
        ]
    )
    stub = _StubClient(payload)
    ex = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]

    results = await ex.aextract(
        "Apple, Google, Microsoft i Mozilla współpracują przy standardzie X"
    )
    assert len(results) == 4
    assert {r.nugget.fact.subject for r in results} == {
        "Apple",
        "Google",
        "Microsoft",
        "Mozilla",
    }
    assert all(r.nugget.fact.predicate == "cooperatesOn" for r in results)
