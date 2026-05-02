"""Tests for ``ConflictDetector`` routing by :class:`Cardinality`.

Ensures the Mode-A fix: event-log and multi-valued predicates never flag
CONTESTED even when two claims have overlapping validity.  Functional
predicates preserve their existing behaviour.
"""

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


def _nugget(
    *,
    subject: str,
    predicate: str,
    obj: str,
    start: datetime,
    end: datetime | None = None,
    n_provenance: int = 1,
) -> Nugget:
    provenance = tuple(
        ProvenanceRecord(source_id=f"doc-{i}", evidence_span="x", char_start=i, char_end=i + 1)
        for i in range(n_provenance)
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text="x"),
        validity=ValidityInterval(start=start, end=end),
        epistemic=EpistemicState(status=LifecycleStatus.ACTIVE),
        provenance=provenance,
    )


@pytest.mark.asyncio
async def test_event_log_predicate_never_flags_contested() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema, judge=None)

    # Two Meta announcements with overlapping validity — an event stream, not
    # a contradiction.
    start = datetime(2023, 6, 1, tzinfo=UTC)
    a = _nugget(
        subject="Meta",
        predicate="announced",
        obj="cancellation of NFTs",
        start=start,
    )
    b = _nugget(
        subject="Meta",
        predicate="announced",
        obj="subscription without ads",
        start=start,
    )

    result = await detector.aresolve(a, existing=[b])
    assert result.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert result.updated_existing == []

    # Also via an alias: `announces` must canonicalize to `announced`.
    c = _nugget(
        subject="Meta",
        predicate="announces",
        obj="sunset NFT features",
        start=start,
    )
    result2 = await detector.aresolve(c, existing=[b])
    assert result2.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert result2.updated_existing == []


@pytest.mark.asyncio
async def test_multi_valued_predicate_never_flags_contested() -> None:
    schema = RelationSchema.default()
    detector = ConflictDetector(schema, judge=None)

    # `acquired` is multi-valued: a company can have many acquisitions.
    start = datetime(2020, 1, 1, tzinfo=UTC)
    a = _nugget(
        subject="Alphabet",
        predicate="acquired",
        obj="Fitbit",
        start=start,
    )
    b = _nugget(
        subject="Alphabet",
        predicate="acquired",
        obj="DeepMind",
        start=start,
    )

    result = await detector.aresolve(a, existing=[b])
    assert result.incoming.epistemic.status == LifecycleStatus.ACTIVE
    assert result.updated_existing == []


@pytest.mark.asyncio
async def test_functional_predicate_still_flags_contested() -> None:
    # Existing behaviour: two conflicting functional claims with overlapping
    # validity and (1,1) evidence -> both CONTESTED.
    schema = RelationSchema.default()
    detector = ConflictDetector(schema, judge=None)

    older = _nugget(
        subject="Apple",
        predicate="chiefExecutiveOfficer",
        obj="Tim Cook",
        start=datetime(2018, 1, 1, tzinfo=UTC),
        n_provenance=1,
    )
    incoming = _nugget(
        subject="Apple",
        predicate="chiefExecutiveOfficer",
        obj="Steve Jobs",
        start=datetime(2018, 6, 1, tzinfo=UTC),
        n_provenance=1,
    )

    result = await detector.aresolve(incoming, existing=[older])
    assert result.incoming.epistemic.status == LifecycleStatus.CONTESTED
    assert len(result.updated_existing) == 1
    assert result.updated_existing[0].epistemic.status == LifecycleStatus.CONTESTED
