"""Tests for the resolve plumbing: ``amark_preferred``, ``asuppress``,
``acontested_keys``."""

from datetime import UTC, datetime

import pytest

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
from nuggetindex.store import NuggetStore


def _n(
    obj: str,
    *,
    status: LifecycleStatus = LifecycleStatus.CONTESTED,
    rank: EpistemicRank = EpistemicRank.NORMAL,
    source: str = "doc-1",
    start_year: int = 2016,
) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Microsoft",
            predicate="acquiredFor",
            object=obj,
            text=f"Microsoft paid {obj} for LinkedIn",
        ),
        validity=ValidityInterval(start=datetime(start_year, 6, 13, tzinfo=UTC)),
        epistemic=EpistemicState(status=status, rank=rank),
        provenance=(ProvenanceRecord(source_id=source, evidence_span=f"...for {obj}..."),),
    )


@pytest.mark.asyncio
async def test_mark_preferred_sets_active_and_preferred(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    a = _n("$26.2B", source="reuters")
    await store.aadd(a)
    updated = await store.amark_preferred(a.id)
    assert updated.epistemic.status is LifecycleStatus.ACTIVE
    assert updated.epistemic.rank is EpistemicRank.PREFERRED
    # Provenance + validity preserved.
    assert updated.provenance == a.provenance
    assert updated.validity == a.validity
    # Re-fetched from the backend matches the in-memory return value.
    refetched = await store.aget(a.id)
    assert refetched is not None
    assert refetched.epistemic.status is LifecycleStatus.ACTIVE
    assert refetched.epistemic.rank is EpistemicRank.PREFERRED
    await store.aclose()


@pytest.mark.asyncio
async def test_mark_preferred_no_set_active_keeps_status(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    a = _n("$26.2B", status=LifecycleStatus.DEPRECATED)
    await store.aadd(a)
    updated = await store.amark_preferred(a.id, set_active=False)
    assert updated.epistemic.status is LifecycleStatus.DEPRECATED
    assert updated.epistemic.rank is EpistemicRank.PREFERRED
    await store.aclose()


@pytest.mark.asyncio
async def test_suppress_marks_deprecated(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    a = _n("$26.4B", source="bloomberg")
    await store.aadd(a)
    updated = await store.asuppress(a.id)
    assert updated.epistemic.status is LifecycleStatus.DEPRECATED
    assert updated.epistemic.rank is EpistemicRank.DEPRECATED
    # Provenance preserved (no hard delete).
    assert updated.provenance == a.provenance
    await store.aclose()


@pytest.mark.asyncio
async def test_mark_preferred_unknown_id_raises(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    with pytest.raises(KeyError):
        await store.amark_preferred("does-not-exist")
    await store.aclose()


@pytest.mark.asyncio
async def test_suppress_unknown_id_raises(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    with pytest.raises(KeyError):
        await store.asuppress("does-not-exist")
    await store.aclose()


@pytest.mark.asyncio
async def test_contested_keys_orders_by_descending_count(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    # Two contested members on the LinkedIn key.
    await store.aadd(_n("$26.2B", source="reuters"))
    await store.aadd(_n("$26.4B", source="bloomberg"))
    # One contested member on a separate key (different subject).
    other = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Twitter",
            predicate="ceo",
            object="Jack Dorsey",
            text="...",
        ),
        validity=ValidityInterval(start=datetime(2015, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(status=LifecycleStatus.CONTESTED),
        provenance=(ProvenanceRecord(source_id="x", evidence_span="x"),),
    )
    await store.aadd(other)
    # An ACTIVE nugget should NOT show up in contested_keys.
    active = _n("ignored", status=LifecycleStatus.ACTIVE, start_year=2020)
    await store.aadd(active)

    keys = await store.acontested_keys()
    # MS/acquiredFor has 2 contested members; Twitter/ceo has 1.
    assert keys[0][:2] == ("Microsoft", "acquiredFor")
    assert keys[0][3] == 2
    assert ("Twitter", "ceo", "global", 1) in keys
    # ACTIVE nuggets must not surface a key even if they share a subject.
    assert all(s != "Microsoft" or p != "ceo" for s, p, _sc, _n in keys)
    await store.aclose()


@pytest.mark.asyncio
async def test_resolve_pins_winner_and_suppresses_losers(tmp_db_path):
    """End-to-end: two contested rivals -> pick one, suppress the other."""
    store = NuggetStore(db_path=tmp_db_path)
    winner = _n("$26.2B", source="reuters")
    loser = _n("$26.4B", source="bloomberg")
    await store.aadd(winner)
    await store.aadd(loser)

    await store.amark_preferred(winner.id)
    await store.asuppress(loser.id)

    w = await store.aget(winner.id)
    los = await store.aget(loser.id)
    assert w is not None and los is not None
    assert w.epistemic.status is LifecycleStatus.ACTIVE
    assert w.epistemic.rank is EpistemicRank.PREFERRED
    assert los.epistemic.status is LifecycleStatus.DEPRECATED
    assert los.epistemic.rank is EpistemicRank.DEPRECATED
    # No more contested keys for this pair.
    keys = await store.acontested_keys()
    assert all(s != "Microsoft" for s, _p, _sc, _n in keys)
    await store.aclose()
