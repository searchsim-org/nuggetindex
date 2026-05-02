"""Tests for ``ConflictTransparency``."""

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
from nuggetindex.evaluation.ragas import ConflictTransparency  # noqa: E402


def _nugget(status: LifecycleStatus) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="X", predicate="p", object="Y", text="X p Y"),
        validity=ValidityInterval(start=datetime(2015, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(status=status, confidence=0.9),
        provenance=(ProvenanceRecord(source_id="s", evidence_span="seed"),),
    )


@pytest.mark.asyncio
async def test_no_contested_is_vacuously_1() -> None:
    metric = ConflictTransparency()
    row = {
        "response": "X is Y, definitely.",
        "retrieved_nuggets": [_nugget(LifecycleStatus.ACTIVE).model_dump_json()],
    }
    assert await metric._ascore(row) == 1.0


@pytest.mark.asyncio
async def test_contested_with_uncertainty_scores_1() -> None:
    metric = ConflictTransparency()
    row = {
        "response": "X might be Y, however some sources disagree.",
        "retrieved_nuggets": [_nugget(LifecycleStatus.CONTESTED).model_dump_json()],
    }
    assert await metric._ascore(row) == 1.0


@pytest.mark.asyncio
async def test_contested_without_uncertainty_scores_0() -> None:
    metric = ConflictTransparency()
    row = {
        "response": "X is definitely Y.",
        "retrieved_nuggets": [_nugget(LifecycleStatus.CONTESTED).model_dump_json()],
    }
    assert await metric._ascore(row) == 0.0


@pytest.mark.asyncio
async def test_contested_keys_column_is_respected() -> None:
    """When ``contested_keys`` is pre-computed we use it instead of reparsing nuggets."""
    metric = ConflictTransparency()
    row = {
        "response": "the answer is disputed",
        "retrieved_nuggets": [],  # empty — would otherwise score 1.0
        "contested_keys": [["X", "p", "global"]],
    }
    # Has contested + uncertainty language -> 1.0
    assert await metric._ascore(row) == 1.0

    row_no_uncertainty = {
        "response": "the answer is obvious",
        "retrieved_nuggets": [],
        "contested_keys": [["X", "p", "global"]],
    }
    assert await metric._ascore(row_no_uncertainty) == 0.0


@pytest.mark.asyncio
async def test_score_is_float_in_range() -> None:
    metric = ConflictTransparency()
    score = await metric._ascore({"response": "foo", "retrieved_nuggets": [], "contested_keys": []})
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
