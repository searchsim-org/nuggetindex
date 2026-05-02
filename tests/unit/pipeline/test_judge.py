"""Tests for ``LLMJudge`` (Improvement C).

Uses a stub ``LLMClient`` to avoid network. The point is to verify the prompt
is loaded, the response model is the 4-enum ``JudgeDecision``, and the
log JSONL is appended.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.clients.base import LLMConfig
from nuggetindex.pipeline.judge import JudgeDecision, LLMJudge


def _nugget(obj: str, start_year: int) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="Google", predicate="chiefExecutiveOfficer", object=obj, text="x"),
        validity=ValidityInterval(start=datetime(start_year, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
    )


class _StubClient:
    def __init__(self, decision: JudgeDecision, rationale: str = "test") -> None:
        self.decision = decision
        self.rationale = rationale
        self.calls: list[tuple[list[dict[str, Any]], type[BaseModel]]] = []

    async def achat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        self.calls.append((messages, response_model))
        return response_model(decision=self.decision, rationale=self.rationale)


@pytest.mark.asyncio
async def test_judge_returns_enum_decision(tmp_path: Path) -> None:
    client = _StubClient(JudgeDecision.A_WINS)
    judge = LLMJudge(
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        client=client,
        log_path=tmp_path / "judge.jsonl",
    )
    a = _nugget("Pichai", 2018)
    b = _nugget("Page", 2015)
    decision = await judge.aadjudicate(a, b)
    assert decision == JudgeDecision.A_WINS
    # Prompt + user message were sent.
    assert len(client.calls) == 1
    messages, _ = client.calls[0]
    assert messages[0]["role"] == "system"
    assert "adjudicator" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_judge_logs_decision(tmp_path: Path) -> None:
    log_path = tmp_path / "judge.jsonl"
    client = _StubClient(JudgeDecision.B_WINS, rationale="more evidence")
    judge = LLMJudge(
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        client=client,
        log_path=log_path,
    )
    a = _nugget("Pichai", 2018)
    b = _nugget("Page", 2015)
    await judge.aadjudicate(a, b)

    assert log_path.exists()
    row = json.loads(log_path.read_text().strip())
    assert row["a_id"] == a.id
    assert row["b_id"] == b.id
    assert row["decision"] == "B_WINS"
    assert row["rationale"] == "more evidence"


@pytest.mark.asyncio
async def test_judge_default_factory_builds_openai_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default() factory builds an openai client eagerly, which requires
    # an API key to be present. Set a dummy key so construction succeeds;
    # we still replace the client before any network call.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-for-testing")
    judge = LLMJudge.default()
    assert judge.cfg.provider == "openai"
    assert judge.cfg.model == "gpt-4o-mini"
    judge.client = _StubClient(JudgeDecision.NEED_MORE_EVIDENCE)  # type: ignore[assignment]
    a = _nugget("Pichai", 2018)
    b = _nugget("Page", 2015)
    decision = await judge.aadjudicate(a, b)
    assert decision == JudgeDecision.NEED_MORE_EVIDENCE


def test_judge_decision_enum_has_four_values() -> None:
    values = {str(v) for v in JudgeDecision}
    assert values == {"A_WINS", "B_WINS", "GENUINELY_CONTESTED", "NEED_MORE_EVIDENCE"}
