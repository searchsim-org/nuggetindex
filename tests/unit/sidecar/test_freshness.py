from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nuggetindex.sidecar.freshness import FreshnessChecker


def test_fresh_when_below_threshold():
    fc = FreshnessChecker(threshold=timedelta(days=90))
    now = datetime.now(tz=UTC)
    recent = now - timedelta(days=10)
    assert fc.is_fresh(latest=recent, now=now) is True


def test_stale_when_above_threshold():
    fc = FreshnessChecker(threshold=timedelta(days=7))
    now = datetime.now(tz=UTC)
    old = now - timedelta(days=30)
    assert fc.is_fresh(latest=old, now=now) is False


def test_stale_when_latest_is_none():
    fc = FreshnessChecker(threshold=timedelta(days=7))
    assert fc.is_fresh(latest=None, now=datetime.now(tz=UTC)) is False


@pytest.mark.asyncio
async def test_check_store_returns_stale_for_old_evidence(tmp_path: Path):
    from nuggetindex import NuggetStore
    from nuggetindex.core.enums import (
        EpistemicRank,
        LifecycleStatus,
        NuggetKind,
    )
    from nuggetindex.core.models import (
        EpistemicState,
        FactTriple,
        Nugget,
        ProvenanceRecord,
        ValidityInterval,
    )

    db = tmp_path / "fc.db"
    store = NuggetStore(db_path=db)
    old = datetime(2020, 1, 1, tzinfo=UTC)
    nugget = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Apple",
            predicate="chiefExecutiveOfficer",
            object="Tim Cook",
            text="Tim Cook is the CEO of Apple.",
        ),
        validity=ValidityInterval(start=old, end=None),
        epistemic=EpistemicState(
            status=LifecycleStatus.ACTIVE,
            rank=EpistemicRank.NORMAL,
            confidence=0.9,
        ),
        provenance=(
            ProvenanceRecord(
                source_id="s1",
                evidence_span="Tim Cook is CEO",
                char_start=0,
                char_end=16,
                created_at=old,
            ),
        ),
        extraction_confidence=0.9,
    )
    await store.aadd(nugget)

    fc = FreshnessChecker(threshold=timedelta(days=90))
    result = await fc.check_store(
        store, subject="Apple", predicate="chiefExecutiveOfficer",
    )
    assert result.is_fresh is False
    await store.backend.aclose()
