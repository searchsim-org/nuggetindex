"""Tests for the LLM-NLI path in :class:`TemporalFaithfulness`.

Phase 8 adds an entailment-based support check when ``self.llm`` is
configured. The fallback path (no LLM) must preserve v0.1's token-overlap
behaviour unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("ragas")

from nuggetindex.evaluation.ragas import (  # noqa: E402
    TemporalFaithfulness,
    _token_overlap_support,
)


@pytest.mark.asyncio
async def test_without_llm_uses_token_overlap_path(sample_nuggets) -> None:
    """Without an LLM the metric falls back to v0.1 token-overlap."""
    metric = TemporalFaithfulness()
    assert metric.llm is None

    # Drive ``_find_supporting_nuggets`` directly so we isolate the path
    # under test from the claim-decomposition / temporal-validity layers.
    supporting = await _call_find_supporting(
        metric, "Pichai is CEO of Google", sample_nuggets
    )
    # Token-overlap should catch at least the Pichai nugget by the token
    # "pichai" (length > 3) appearing in its object string.
    assert any("Pichai" in n.fact.object for n in supporting)

    # And the helper produces the same result when called directly.
    direct = _token_overlap_support("Pichai is CEO of Google", sample_nuggets)
    assert direct == supporting


@pytest.mark.asyncio
async def test_with_llm_calls_nli_entailment(sample_nuggets) -> None:
    """With an LLM configured the metric invokes it per nugget."""
    metric = TemporalFaithfulness()

    # Stub the Ragas LLM: ``agenerate_text`` returns a response object whose
    # first generation text is JSON-decodable and claims support=True.
    mock_llm = AsyncMock()
    mock_llm.agenerate_text.return_value = type(
        "Resp",
        (),
        {
            "generations": [
                [type("Gen", (), {"text": '{"supports": true, "rationale": "direct match"}'})()]
            ]
        },
    )()
    metric.llm = mock_llm

    supporting = await _call_find_supporting(
        metric, "Pichai runs Google", sample_nuggets
    )
    assert supporting  # at least one nugget passed entailment
    assert mock_llm.agenerate_text.called  # LLM was actually invoked


@pytest.mark.asyncio
async def test_with_llm_rejects_non_supporting(sample_nuggets) -> None:
    """When the LLM returns ``supports=false`` the nugget is not supporting."""
    metric = TemporalFaithfulness()
    mock_llm = AsyncMock()
    mock_llm.agenerate_text.return_value = type(
        "Resp",
        (),
        {
            "generations": [
                [type("Gen", (), {"text": '{"supports": false, "rationale": "unrelated"}'})()]
            ]
        },
    )()
    metric.llm = mock_llm

    supporting = await _call_find_supporting(
        metric, "Pichai runs Google", sample_nuggets
    )
    assert supporting == []
    assert mock_llm.agenerate_text.call_count == len(sample_nuggets)


@pytest.mark.asyncio
async def test_with_llm_skips_unparseable_response(sample_nuggets) -> None:
    """A non-JSON LLM response is treated conservatively (no support)."""
    metric = TemporalFaithfulness()
    mock_llm = AsyncMock()
    mock_llm.agenerate_text.return_value = type(
        "Resp",
        (),
        {"generations": [[type("Gen", (), {"text": "not json at all"})()]]},
    )()
    metric.llm = mock_llm

    supporting = await _call_find_supporting(
        metric, "Pichai runs Google", sample_nuggets
    )
    assert supporting == []


async def _call_find_supporting(metric, claim, nuggets):
    """Shim around ``_find_supporting_nuggets`` that awaits only when needed.

    Task 8.2 turns the previously-sync helper into an ``async def`` (so it
    can ``await`` ``llm.agenerate_text``). The tests expect to await it;
    this shim keeps the call site readable.
    """
    return await metric._find_supporting_nuggets(claim, nuggets)
