"""Tests for Stage 4b conflict detection (Algorithm 2)."""

from datetime import UTC, datetime

import pytest

from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.core.schema import RelationSchema
from nuggetindex.pipeline.conflict import ConflictDetector
from nuggetindex.pipeline.judge import JudgeDecision


def _nugget(
    *,
    subject: str = "Google",
    predicate: str = "chiefExecutiveOfficer",
    obj: str = "Pichai",
    start: datetime,
    end: datetime | None = None,
    n_provenance: int = 1,
    status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> Nugget:
    provenance = tuple(
        ProvenanceRecord(source_id=f"doc-{i}", evidence_span="x", char_start=i, char_end=i + 1)
        for i in range(n_provenance)
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text="x"),
        validity=ValidityInterval(start=start, end=end),
        epistemic=EpistemicState(status=status),
        provenance=provenance,
    )


# --- Branch 1: new key -----------------------------------------------------


@pytest.mark.asyncio
async def test_new_key_no_prior_returns_unchanged() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema)
    incoming = _nugget(start=datetime(2020, 1, 1, tzinfo=UTC))
    result = await detector.aresolve(incoming, existing=[])
    assert result.incoming is incoming
    assert result.updated_existing == []
    assert not result.judge_invoked


# --- Branch 2: non-overlapping / multi-valued ------------------------------


@pytest.mark.asyncio
async def test_same_key_non_overlapping_succession() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema)
    older = _nugget(
        obj="Page",
        start=datetime(2010, 1, 1, tzinfo=UTC),
        end=datetime(2015, 10, 1, tzinfo=UTC),
    )
    newer = _nugget(
        obj="Pichai",
        start=datetime(2015, 10, 2, tzinfo=UTC),
    )
    result = await detector.aresolve(newer, existing=[older])
    assert result.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert result.updated_existing == []


@pytest.mark.asyncio
async def test_multi_valued_predicate_coexists() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema)
    # `boardMember` is multi_valued in the default schema.
    a = _nugget(
        predicate="boardMember",
        obj="Alice",
        start=datetime(2020, 1, 1, tzinfo=UTC),
    )
    b = _nugget(
        predicate="boardMember",
        obj="Bob",
        start=datetime(2020, 1, 1, tzinfo=UTC),
    )
    result = await detector.aresolve(a, existing=[b])
    assert result.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert result.updated_existing == []


# --- Branch 3: functional + overlap + asymmetric evidence ------------------


@pytest.mark.asyncio
async def test_newer_with_more_evidence_deprecates_older() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema)
    older = _nugget(
        obj="Page",
        start=datetime(2010, 1, 1, tzinfo=UTC),
        n_provenance=1,
    )
    newer = _nugget(
        obj="Pichai",
        start=datetime(2015, 10, 2, tzinfo=UTC),
        n_provenance=3,
    )
    result = await detector.aresolve(newer, existing=[older])
    assert result.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert len(result.updated_existing) == 1
    updated = result.updated_existing[0]
    assert updated.epistemic.status == LifecycleStatus.DEPRECATED
    assert updated.validity.end == newer.validity.start


@pytest.mark.asyncio
async def test_older_with_more_evidence_deprecates_newer() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema)
    older = _nugget(
        obj="Page",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        n_provenance=3,
    )
    incoming = _nugget(
        obj="Pichai",
        start=datetime(2015, 1, 1, tzinfo=UTC),
        n_provenance=1,
    )
    # older is both newer (by start) AND has more evidence; we treat it as winner.
    result = await detector.aresolve(incoming, existing=[older])
    assert result.incoming.epistemic.status == LifecycleStatus.DEPRECATED
    # Older not modified -> updated_existing empty.
    assert result.updated_existing == []


# --- Branch 4: symmetric / (1,1) -> both CONTESTED --------------------------


@pytest.mark.asyncio
async def test_symmetric_evidence_marks_both_contested_without_judge() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema, judge=None)
    older = _nugget(
        obj="Page",
        start=datetime(2018, 1, 1, tzinfo=UTC),
        n_provenance=1,
    )
    incoming = _nugget(
        obj="Pichai",
        start=datetime(2018, 6, 1, tzinfo=UTC),
        n_provenance=1,
    )
    result = await detector.aresolve(incoming, existing=[older])
    assert result.incoming.epistemic.status == LifecycleStatus.CONTESTED
    assert len(result.updated_existing) == 1
    assert result.updated_existing[0].epistemic.status == LifecycleStatus.CONTESTED
    assert not result.judge_invoked


# --- Branch 4 with judge ---------------------------------------------------


class _StubJudge:
    def __init__(self, decision: JudgeDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str]] = []

    async def aadjudicate(self, incoming: Nugget, existing: Nugget) -> JudgeDecision:
        self.calls.append((incoming.fact.object, existing.fact.object))
        return self.decision


@pytest.mark.asyncio
async def test_judge_a_wins_makes_incoming_active_and_existing_deprecated() -> None:
    schema = RelationSchema.default()
    judge = _StubJudge(JudgeDecision.A_WINS)
    detector = ConflictDetector(schema, judge=judge)

    older = _nugget(obj="Page", start=datetime(2018, 1, 1, tzinfo=UTC))
    incoming = _nugget(obj="Pichai", start=datetime(2018, 6, 1, tzinfo=UTC))

    result = await detector.aresolve(incoming, existing=[older])
    assert judge.calls == [("Pichai", "Page")]
    assert result.judge_invoked is True
    assert result.judge_decision == "A_WINS"
    assert result.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert result.updated_existing[0].epistemic.status == LifecycleStatus.DEPRECATED


@pytest.mark.asyncio
async def test_judge_b_wins_deprecates_incoming() -> None:
    schema = RelationSchema.default()
    judge = _StubJudge(JudgeDecision.B_WINS)
    detector = ConflictDetector(schema, judge=judge)

    older = _nugget(obj="Page", start=datetime(2018, 1, 1, tzinfo=UTC))
    incoming = _nugget(obj="Pichai", start=datetime(2018, 6, 1, tzinfo=UTC))

    result = await detector.aresolve(incoming, existing=[older])
    assert result.judge_invoked is True
    assert result.incoming.epistemic.status == LifecycleStatus.DEPRECATED
    assert result.updated_existing[0].epistemic.status == LifecycleStatus.ACTIVE


@pytest.mark.asyncio
async def test_judge_genuinely_contested_keeps_both_contested() -> None:
    schema = RelationSchema.default()
    judge = _StubJudge(JudgeDecision.GENUINELY_CONTESTED)
    detector = ConflictDetector(schema, judge=judge)

    older = _nugget(obj="Page", start=datetime(2018, 1, 1, tzinfo=UTC))
    incoming = _nugget(obj="Pichai", start=datetime(2018, 6, 1, tzinfo=UTC))

    result = await detector.aresolve(incoming, existing=[older])
    assert result.judge_invoked is True
    assert result.incoming.epistemic.status == LifecycleStatus.CONTESTED
    assert result.updated_existing[0].epistemic.status == LifecycleStatus.CONTESTED


@pytest.mark.asyncio
async def test_judge_not_invoked_when_evidence_asymmetric() -> None:
    schema = RelationSchema.default()
    judge = _StubJudge(JudgeDecision.A_WINS)
    detector = ConflictDetector(schema, judge=judge)

    older = _nugget(obj="Page", start=datetime(2018, 1, 1, tzinfo=UTC), n_provenance=1)
    incoming = _nugget(obj="Pichai", start=datetime(2018, 6, 1, tzinfo=UTC), n_provenance=3)

    result = await detector.aresolve(incoming, existing=[older])
    assert judge.calls == []  # not invoked; rules resolved it
    assert result.judge_invoked is False
    assert result.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert result.updated_existing[0].epistemic.status == LifecycleStatus.DEPRECATED
