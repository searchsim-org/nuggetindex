"""Tests for the v0.2.1 staleness heuristic upgrade (findings-A2).

v0.2.0 flagged *every* open-ended-validity nugget as potentially stale, which
meant freshly-extracted content was always flagged. v0.2.1 adds a
``stale_threshold_days`` parameter (default 180) that additionally requires
the most recent provenance ``created_at`` to be older than the threshold.
Passing ``None`` restores the v0.2.0 behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex import audit
from tests.fixtures import RuleBasedExtractor


@pytest.mark.asyncio
async def test_fresh_nuggets_not_flagged() -> None:
    """Default threshold (180d) + fresh provenance ⇒ no stale flags."""
    report = await audit(
        query="Who is the CEO?",
        passages=["Sundar Pichai is the CEO of Google."],
        query_time=datetime.now(UTC),
        extractor=RuleBasedExtractor(),
    )
    assert len(report.potentially_stale) == 0


@pytest.mark.asyncio
async def test_stale_threshold_none_restores_v020_behaviour() -> None:
    """``stale_threshold_days=None`` disables the age check (v0.2.0 behaviour)."""
    report = await audit(
        query="Who is the CEO?",
        passages=["Sundar Pichai is the CEO of Google."],
        query_time=datetime.now(UTC),
        extractor=RuleBasedExtractor(),
        stale_threshold_days=None,
    )
    # Every rule-based nugget has validity.end=None, so every one should flag.
    assert len(report.potentially_stale) >= 1


@pytest.mark.asyncio
async def test_old_source_dated_nuggets_flagged_even_with_threshold() -> None:
    """A pre-built nugget with old provenance ``created_at`` still flags."""
    from nuggetindex.audit.api import _classify_stale
    from nuggetindex.core.enums import NuggetKind
    from nuggetindex.core.models import (
        EpistemicState,
        FactTriple,
        Nugget,
        ProvenanceRecord,
        ValidityInterval,
    )

    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="X", predicate="is", object="Y", text="X is Y"),
        validity=ValidityInterval(start=datetime(2010, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id="doc-1",
                evidence_span="x",
                created_at=datetime(2010, 1, 1, tzinfo=UTC),  # 14+ years old
            ),
        ),
    )
    is_stale = _classify_stale(
        n,
        query_time=datetime.now(UTC),
        stale_threshold_days=180,
    )
    assert is_stale
