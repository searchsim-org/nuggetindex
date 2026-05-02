"""Property tests for conflict detection (invariant: commutativity).

Spec §2.3 invariant: ingestion order must not change the final lifecycle
states when ``judge=None`` (rules are deterministic and symmetric).

We simulate the "ingest A then B" vs "ingest B then A" scenario by running
``ConflictDetector.aresolve`` twice in each order over a pair of nuggets
and asserting the resulting lifecycle/status tuples match (modulo order).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

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

_SCHEMA = RelationSchema.default()


def _build(
    *,
    obj: str,
    start_year: int,
    n_provenance: int,
) -> Nugget:
    provenance = tuple(
        ProvenanceRecord(
            source_id=f"doc-{obj}-{i}",
            evidence_span="x",
            char_start=i,
            char_end=i + 1,
        )
        for i in range(n_provenance)
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="chiefExecutiveOfficer",
            object=obj,
            text="x",
        ),
        validity=ValidityInterval(start=datetime(start_year, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(status=LifecycleStatus.ACTIVE),
        provenance=provenance,
    )


async def _run_pair(a: Nugget, b: Nugget) -> tuple[LifecycleStatus, LifecycleStatus]:
    """Simulate ingesting ``a`` then ``b``; return their final statuses keyed by object."""
    detector = ConflictDetector(_SCHEMA, judge=None)

    # Ingest A -> nothing prior.
    r1 = await detector.aresolve(a, existing=[])
    state_by_obj: dict[str, Nugget] = {r1.incoming.fact.object: r1.incoming}

    # Ingest B with A as prior.
    r2 = await detector.aresolve(b, existing=list(state_by_obj.values()))
    # Apply any existing updates from the resolution.
    for upd in r2.updated_existing:
        state_by_obj[upd.fact.object] = upd
    state_by_obj[r2.incoming.fact.object] = r2.incoming

    return (
        state_by_obj[a.fact.object].epistemic.status,
        state_by_obj[b.fact.object].epistemic.status,
    )


@settings(deadline=None, max_examples=30)
@given(
    a_year=st.integers(min_value=2000, max_value=2020),
    b_year=st.integers(min_value=2000, max_value=2020),
    a_ev=st.integers(min_value=1, max_value=4),
    b_ev=st.integers(min_value=1, max_value=4),
)
def test_commutativity_of_conflict_resolution_without_judge(
    a_year: int, b_year: int, a_ev: int, b_ev: int
) -> None:
    # Require distinct object values so they're true competitors (same key,
    # different value). Require distinct start years so evidence-asymmetry
    # rules can fire deterministically.
    if a_year == b_year:
        return
    a = _build(obj="Page", start_year=a_year, n_provenance=a_ev)
    b = _build(obj="Pichai", start_year=b_year, n_provenance=b_ev)

    order_ab = asyncio.run(_run_pair(a, b))
    order_ba = asyncio.run(_run_pair(b, a))
    # Invariant: the final status of "Page" in AB must equal the final status
    # of "Page" in BA (and same for Pichai).
    page_ab, pichai_ab = order_ab
    pichai_ba, page_ba = order_ba
    assert page_ab == page_ba
    assert pichai_ab == pichai_ba
