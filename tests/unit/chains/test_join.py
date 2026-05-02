"""Tests for :meth:`NuggetStore.achain_join`."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex import NuggetStore
from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.errors import ChainAmbiguousError
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)


def _fact(
    subject: str,
    predicate: str,
    object_: str,
    start_year: int,
    end_year: int | None = None,
    *,
    status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> Nugget:
    vi = ValidityInterval(
        start=datetime(start_year, 1, 1, tzinfo=UTC),
        end=datetime(end_year, 1, 1, tzinfo=UTC) if end_year else None,
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=object_,
            text=f"{subject} {predicate} {object_}",
        ),
        validity=vi,
        epistemic=EpistemicState(status=status),
        provenance=(
            ProvenanceRecord(
                source_id=f"doc-{subject}-{object_}-{start_year}",
                evidence_span=f"{subject} {predicate} {object_}",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_join_one_hop_returns_functional_lookup(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_fact("Google", "parentCompany", "Alphabet", 2015))
        chain = await store.achain_join(
            start=("Google", "parentCompany"), then=[]
        )
        assert chain.chain_type == "joined"
        assert len(chain.nuggets) == 1
        assert chain.nuggets[0].fact.object == "Alphabet"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_join_two_hop_through_alphabet(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_fact("Google", "parentCompany", "Alphabet", 2015))
        await store.aadd(_fact("Alphabet", "chiefExecutiveOfficer", "Pichai", 2019))
        chain = await store.achain_join(
            start=("Google", "parentCompany"),
            then=["ceo"],
            as_of=datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert len(chain.nuggets) == 2
        assert chain.nuggets[0].fact.object == "Alphabet"
        assert chain.nuggets[1].fact.object == "Pichai"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_join_max_hops_guardrail(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        with pytest.raises(ValueError) as ei:
            await store.achain_join(
                start=("A", "p"), then=["a", "b", "c", "d"]
            )
        assert "max_hops" in str(ei.value)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_join_missing_binding_raises(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_fact("Google", "parentCompany", "Alphabet", 2015))
        # No ceo nugget for Alphabet -> zero candidates at step 1.
        with pytest.raises(ChainAmbiguousError):
            await store.achain_join(
                start=("Google", "parentCompany"),
                then=["ceo"],
                as_of=datetime(2020, 1, 1, tzinfo=UTC),
            )
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_join_ambiguous_at_start_raises(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        # Two nuggets valid at as_of -> ambiguous.
        await store.aadd(
            _fact("Google", "parentCompany", "A1", 2015, 2025)
        )
        await store.aadd(
            _fact("Google", "parentCompany", "A2", 2016, 2025)
        )
        with pytest.raises(ChainAmbiguousError) as ei:
            await store.achain_join(
                start=("Google", "parentCompany"),
                then=[],
                as_of=datetime(2020, 1, 1, tzinfo=UTC),
            )
        assert ei.value.subject == "Google"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_join_with_resolver_picks_candidate(tmp_db_path: Path) -> None:
    class StubResolver:
        async def adisambiguate(self, *, candidates: list, context: str) -> object:
            from types import SimpleNamespace

            return SimpleNamespace(
                picked=candidates[0], rationale="first"
            )

    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(
            _fact("Google", "parentCompany", "A1", 2015, 2025)
        )
        await store.aadd(
            _fact("Google", "parentCompany", "A2", 2016, 2025)
        )
        chain = await store.achain_join(
            start=("Google", "parentCompany"),
            then=[],
            as_of=datetime(2020, 1, 1, tzinfo=UTC),
            resolver=StubResolver(),
        )
        assert len(chain.nuggets) == 1
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_join_skips_deprecated(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(
            _fact(
                "Google",
                "parentCompany",
                "Stale",
                2010,
                2025,
                status=LifecycleStatus.DEPRECATED,
            )
        )
        await store.aadd(
            _fact("Google", "parentCompany", "Alphabet", 2015)
        )
        chain = await store.achain_join(
            start=("Google", "parentCompany"),
            then=[],
            as_of=datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert chain.nuggets[0].fact.object == "Alphabet"
    finally:
        await store.aclose()


def test_join_sync_wrapper(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        store.add(_fact("Google", "parentCompany", "Alphabet", 2015))
        chain = store.chain_join(
            start=("Google", "parentCompany"), then=[]
        )
        assert chain.nuggets[0].fact.object == "Alphabet"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_join_three_hop_allowed(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_fact("A", "p1", "B", 2015))
        await store.aadd(_fact("B", "p2", "C", 2015))
        await store.aadd(_fact("C", "p3", "D", 2015))
        chain = await store.achain_join(
            start=("A", "p1"), then=["p2", "p3"],
        )
        # len(then)==2 plus first hop => 3 nuggets
        assert len(chain.nuggets) == 3
    finally:
        await store.aclose()
