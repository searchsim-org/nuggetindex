"""Tests for ``TemporalFaithfulness``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("ragas")

from nuggetindex.core.enums import LifecycleStatus, NuggetKind  # noqa: E402
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.evaluation.ragas import TemporalFaithfulness  # noqa: E402


def _nugget(
    *,
    subject: str = "Google",
    predicate: str = "ceo",
    obj: str = "Sundar Pichai",
    start: datetime | None = None,
    end: datetime | None = None,
    status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=f"{subject} {predicate} {obj}",
        ),
        validity=ValidityInterval(
            start=start or datetime(2015, 10, 2, tzinfo=UTC),
            end=end,
        ),
        epistemic=EpistemicState(status=status, confidence=0.95),
        provenance=(ProvenanceRecord(source_id="d1", evidence_span="seed"),),
    )


@pytest.mark.asyncio
async def test_returns_1_when_no_claims() -> None:
    metric = TemporalFaithfulness()
    score = await metric._ascore(
        {"response": "", "retrieved_nuggets": [], "query_time": None}
    )
    assert score == 1.0


@pytest.mark.asyncio
async def test_all_claims_supported() -> None:
    metric = TemporalFaithfulness()
    row = {
        "response": "Sundar Pichai is CEO of Google.",
        "retrieved_nuggets": [_nugget().model_dump_json()],
        "query_time": "2020-01-01T00:00:00+00:00",
    }
    score = await metric._ascore(row)
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_score_is_float_between_0_and_1() -> None:
    metric = TemporalFaithfulness()
    row = {
        "response": "Sundar Pichai is CEO. The capital of France is Paris.",
        "retrieved_nuggets": [_nugget().model_dump_json()],
        "query_time": "2020-01-01T00:00:00+00:00",
    }
    score = await metric._ascore(row)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # One claim matches, one doesn't (Paris/France has no nugget support).
    assert score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_temporal_filter_excludes_out_of_window_nuggets() -> None:
    """A nugget whose validity ends before query_time does NOT support the claim."""
    metric = TemporalFaithfulness()
    expired = _nugget(
        start=datetime(2010, 1, 1, tzinfo=UTC),
        end=datetime(2015, 1, 1, tzinfo=UTC),
    )
    row = {
        "response": "Sundar Pichai is CEO of Google.",
        "retrieved_nuggets": [expired.model_dump_json()],
        "query_time": "2020-01-01T00:00:00+00:00",
    }
    score = await metric._ascore(row)
    # Claim had a matching nugget, but it was temporally invalid.
    assert score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_deprecated_nuggets_do_not_support() -> None:
    metric = TemporalFaithfulness()
    deprecated = _nugget(status=LifecycleStatus.DEPRECATED)
    row = {
        "response": "Sundar Pichai is CEO of Google.",
        "retrieved_nuggets": [deprecated.model_dump_json()],
        "query_time": "2020-01-01T00:00:00+00:00",
    }
    assert await metric._ascore(row) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ascore_without_llm_uses_sentence_split() -> None:
    metric = TemporalFaithfulness()
    assert metric.llm is None
    # Two sentences, only one supported.
    row = {
        "response": "Sundar Pichai leads Google. Bananas are yellow.",
        "retrieved_nuggets": [_nugget().model_dump_json()],
        "query_time": None,
    }
    score = await metric._ascore(row)
    assert score == pytest.approx(0.5)
