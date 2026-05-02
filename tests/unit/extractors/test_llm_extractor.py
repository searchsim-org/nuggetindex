"""Tests for LLMExtractor with a stubbed structured-output client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from nuggetindex.core.enums import NuggetKind
from nuggetindex.extractors.base import ExtractionResult
from nuggetindex.extractors.clients.base import LLMConfig
from nuggetindex.extractors.llm import ExtractionPayload, LLMExtractor


class _StubClient:
    """Minimal LLMClient implementation for deterministic tests."""

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


def _payload(**overrides: Any) -> ExtractionPayload:
    base = {
        "facts": [
            {
                "subject": "Google",
                "predicate": "ceo",
                "object": "Sundar Pichai",
                "evidence_span": "Sundar Pichai is CEO of Google",
                "confidence": 0.9,
            }
        ]
    }
    base.update(overrides)
    return ExtractionPayload.model_validate(base)


@pytest.mark.asyncio
async def test_llm_extractor_parses_structured_response() -> None:
    stub = _StubClient(_payload())
    ext = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]
    results = await ext.aextract("Sundar Pichai is CEO of Google.")
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, ExtractionResult)
    assert r.nugget.kind == NuggetKind.SEMANTIC_FACT
    assert r.nugget.fact.subject == "Google"
    assert r.nugget.fact.predicate == "ceo"
    assert r.nugget.fact.object == "Sundar Pichai"
    assert r.confidence == 0.9
    assert r.nugget.extraction_confidence == 0.9
    assert r.nugget.provenance[0].evidence_span == "Sundar Pichai is CEO of Google"


@pytest.mark.asyncio
async def test_llm_extractor_empty_text_skips_llm_call() -> None:
    stub = _StubClient(_payload())
    ext = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]
    assert await ext.aextract("") == []
    assert await ext.aextract("   ") == []
    stub.achat_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_extractor_passes_context_hint() -> None:
    stub = _StubClient(_payload())
    ext = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]
    await ext.aextract("Some text.", context="document dated 2024-01-01")
    messages = stub.achat_structured.await_args.args[0]
    user_content = messages[-1]["content"]
    assert "Context: document dated 2024-01-01" in user_content
    assert "Some text." in user_content


@pytest.mark.asyncio
async def test_llm_extractor_without_context_omits_hint() -> None:
    stub = _StubClient(_payload())
    ext = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]
    await ext.aextract("Some text.")
    messages = stub.achat_structured.await_args.args[0]
    user_content = messages[-1]["content"]
    assert "Context:" not in user_content


@pytest.mark.asyncio
async def test_llm_extractor_uses_source_id_fn() -> None:
    stub = _StubClient(_payload())
    counter = {"n": 0}

    def _source() -> str:
        counter["n"] += 1
        return f"doc-{counter['n']}"

    ext = LLMExtractor(_cfg(), client=stub, source_id_fn=_source)  # type: ignore[arg-type]
    results = await ext.aextract("Sundar Pichai is CEO of Google.")
    assert results[0].nugget.provenance[0].source_id == "doc-1"


@pytest.mark.asyncio
async def test_llm_extractor_default_source_id() -> None:
    stub = _StubClient(_payload())
    ext = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]
    results = await ext.aextract("Sundar Pichai is CEO of Google.")
    assert results[0].nugget.provenance[0].source_id == "llm-extract"


@pytest.mark.asyncio
async def test_llm_extractor_multiple_facts() -> None:
    payload = ExtractionPayload.model_validate(
        {
            "facts": [
                {
                    "subject": "Google",
                    "predicate": "ceo",
                    "object": "Sundar Pichai",
                    "evidence_span": "Sundar Pichai is CEO of Google",
                    "confidence": 0.95,
                },
                {
                    "subject": "Apple",
                    "predicate": "founded",
                    "object": "1976",
                    "evidence_span": "Apple was founded in 1976",
                    "confidence": 0.8,
                },
            ]
        }
    )
    stub = _StubClient(payload)
    ext = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]
    results = await ext.aextract("irrelevant")
    assert len(results) == 2
    assert {r.nugget.fact.subject for r in results} == {"Google", "Apple"}


@pytest.mark.asyncio
async def test_llm_extractor_system_prompt_not_empty() -> None:
    stub = _StubClient(_payload())
    ext = LLMExtractor(_cfg(), client=stub)  # type: ignore[arg-type]
    await ext.aextract("irrelevant")
    messages = stub.achat_structured.await_args.args[0]
    assert messages[0]["role"] == "system"
    assert "atomic facts" in messages[0]["content"].lower()
