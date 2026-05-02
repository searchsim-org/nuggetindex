"""Tests for the ``ChainCompleteness`` Ragas metric."""

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
from nuggetindex.evaluation.ragas import ChainCompleteness  # noqa: E402


def _google_ceo(obj: str, start_year: int, end_year: int | None) -> Nugget:
    vi = ValidityInterval(
        start=datetime(start_year, 1, 1, tzinfo=UTC),
        end=datetime(end_year, 1, 1, tzinfo=UTC) if end_year else None,
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="ceo",
            object=obj,
            text=f"{obj} is CEO",
        ),
        validity=vi,
        epistemic=EpistemicState(status=LifecycleStatus.ACTIVE),
        provenance=(
            ProvenanceRecord(
                source_id=f"doc-{obj}",
                evidence_span=f"{obj} is CEO",
            ),
        ),
    )


def _ceo_nuggets() -> list[Nugget]:
    return [
        _google_ceo("Schmidt", 2001, 2011),
        _google_ceo("Page", 2011, 2015),
        _google_ceo("Pichai", 2015, None),
    ]


def _json_nuggets(nuggets: list[Nugget]) -> list[str]:
    return [n.model_dump_json() for n in nuggets]


@pytest.mark.asyncio
async def test_no_chain_reference_returns_1() -> None:
    metric = ChainCompleteness()
    row = {
        "response": "The weather is sunny today.",
        "retrieved_nuggets": _json_nuggets(_ceo_nuggets()),
    }
    assert await metric._ascore(row) == 1.0


@pytest.mark.asyncio
async def test_correct_order_returns_1() -> None:
    metric = ChainCompleteness()
    row = {
        "response": "First Schmidt, then Page, then Pichai became CEO.",
        "retrieved_nuggets": _json_nuggets(_ceo_nuggets()),
    }
    score = await metric._ascore(row)
    assert score == 1.0


@pytest.mark.asyncio
async def test_skipped_middle_is_lower() -> None:
    metric = ChainCompleteness()
    row = {
        "response": "Schmidt handed off to Pichai.",
        "retrieved_nuggets": _json_nuggets(_ceo_nuggets()),
    }
    score = await metric._ascore(row)
    assert score < 1.0


@pytest.mark.asyncio
async def test_wrong_order_is_lower() -> None:
    metric = ChainCompleteness()
    row = {
        "response": "Pichai, then Page, then Schmidt led Google.",
        "retrieved_nuggets": _json_nuggets(_ceo_nuggets()),
    }
    score = await metric._ascore(row)
    assert score < 1.0


@pytest.mark.asyncio
async def test_empty_answer_returns_1() -> None:
    metric = ChainCompleteness()
    row = {
        "response": "",
        "retrieved_nuggets": _json_nuggets(_ceo_nuggets()),
    }
    assert await metric._ascore(row) == 1.0


@pytest.mark.asyncio
async def test_empty_nuggets_returns_1() -> None:
    metric = ChainCompleteness()
    row = {
        "response": "Schmidt then Page then Pichai.",
        "retrieved_nuggets": [],
    }
    assert await metric._ascore(row) == 1.0


@pytest.mark.asyncio
async def test_single_entity_returns_1() -> None:
    metric = ChainCompleteness()
    row = {
        "response": "Pichai is the current CEO of Google.",
        "retrieved_nuggets": _json_nuggets(_ceo_nuggets()),
    }
    # Only Pichai (ignoring Google-the-subject) -> not enough for adjacent
    # pairs -> 1.0.
    score = await metric._ascore(row)
    assert score == 1.0


@pytest.mark.asyncio
async def test_chain_hint_picks_right_group() -> None:
    metric = ChainCompleteness()
    # Two groups: (Google, ceo) and (Alphabet, ceo). Hint steers scoring.
    google = _ceo_nuggets()
    alphabet = [
        Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject="Alphabet",
                predicate="ceo",
                object="Pichai",
                text="Pichai is Alphabet CEO",
            ),
            validity=ValidityInterval(start=datetime(2019, 1, 1, tzinfo=UTC)),
            epistemic=EpistemicState(),
            provenance=(
                ProvenanceRecord(
                    source_id="alph-1",
                    evidence_span="",
                ),
            ),
        )
    ]
    row = {
        "response": "First Schmidt, then Page, then Pichai.",
        "retrieved_nuggets": _json_nuggets(google + alphabet),
        "chain_subject": "Google",
        "chain_predicate": "ceo",
    }
    score = await metric._ascore(row)
    assert score == 1.0
