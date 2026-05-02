"""Tests for :class:`ChainResolver`."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from nuggetindex.chains.resolver import (
    ChainResolution,
    ChainResolver,
    _ResolverResponse,
)
from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.clients.base import LLMConfig


def _fact(obj: str, start_year: int = 2020) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="X",
            predicate="renamedTo",
            object=obj,
            text=f"X renamed to {obj}",
        ),
        validity=ValidityInterval(start=datetime(start_year, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id=f"doc-{obj}",
                evidence_span=f"X renamed to {obj}",
            ),
        ),
    )


class _StubClient:
    """Stubbed ``LLMClient`` that returns a canned index + rationale."""

    def __init__(self, picked_index: int = 0, rationale: str = "stub pick") -> None:
        self.picked_index = picked_index
        self.rationale = rationale
        self.calls: list[dict[str, Any]] = []

    async def achat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        self.calls.append({"messages": messages, "model": response_model})
        assert response_model is _ResolverResponse
        return _ResolverResponse(picked_index=self.picked_index, rationale=self.rationale)


@pytest.mark.asyncio
async def test_resolver_returns_chain_resolution(tmp_path: Path) -> None:
    stub = _StubClient(picked_index=1, rationale="B looks better")
    resolver = ChainResolver(
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        client=stub,
        log_path=tmp_path / "log.jsonl",
    )
    candidates = [_fact("A"), _fact("B")]
    result = await resolver.adisambiguate(candidates=candidates, context="test")
    assert isinstance(result, ChainResolution)
    assert result.picked == candidates[1]
    assert result.rationale == "B looks better"


@pytest.mark.asyncio
async def test_resolver_writes_log_row(tmp_path: Path) -> None:
    stub = _StubClient(picked_index=0, rationale="first")
    log_path = tmp_path / "log.jsonl"
    resolver = ChainResolver(
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        client=stub,
        log_path=log_path,
    )
    candidates = [_fact("A"), _fact("B")]
    await resolver.adisambiguate(candidates=candidates, context="ctx")
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["picked_idx"] == 0
    assert rows[0]["context"] == "ctx"
    assert rows[0]["picked_id"] == candidates[0].id
    assert set(rows[0]["candidate_ids"]) == {candidates[0].id, candidates[1].id}


@pytest.mark.asyncio
async def test_resolver_log_path_respects_nuggetindex_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NUGGETINDEX_HOME", str(tmp_path))
    stub = _StubClient()
    resolver = ChainResolver(
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        client=stub,
    )
    await resolver.adisambiguate(candidates=[_fact("A"), _fact("B")], context="ctx")
    expected_log = tmp_path / "chain_resolver_log.jsonl"
    assert expected_log.exists()


def test_resolver_default_does_not_need_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolver = ChainResolver.default()
    assert isinstance(resolver, ChainResolver)
    assert resolver.cfg.provider == "openai"
    assert resolver.cfg.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_resolver_empty_candidates_raises(tmp_path: Path) -> None:
    resolver = ChainResolver(
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        client=_StubClient(),
        log_path=tmp_path / "log.jsonl",
    )
    with pytest.raises(ValueError):
        await resolver.adisambiguate(candidates=[], context="ctx")


@pytest.mark.asyncio
async def test_resolver_clamps_out_of_range_index(tmp_path: Path) -> None:
    stub = _StubClient(picked_index=99, rationale="oops")
    resolver = ChainResolver(
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        client=stub,
        log_path=tmp_path / "log.jsonl",
    )
    candidates = [_fact("A"), _fact("B")]
    result = await resolver.adisambiguate(candidates=candidates, context="ctx")
    # Out-of-range gets clamped to 0.
    assert result.picked == candidates[0]
