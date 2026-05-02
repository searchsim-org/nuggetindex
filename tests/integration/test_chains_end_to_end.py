"""End-to-end integration test for temporal provenance chains.

Seeds a 3-company corpus (Google CEO succession, Twitter -> X Corp rename,
Alphabet parent-of-Google) into a fresh NuggetStore, then exercises all
three chain methods against it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex import NuggetStore
from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)


def _n(
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
                source_id=f"doc-{subject}-{predicate}-{object_}",
                evidence_span=f"{subject} {predicate} {object_}",
            ),
        ),
    )


def _corpus() -> list[Nugget]:
    return [
        # --- Google CEO succession (Schmidt -> Page -> Pichai)
        _n("Google", "chiefExecutiveOfficer", "Schmidt", 2001, 2011),
        _n("Google", "chiefExecutiveOfficer", "Page", 2011, 2015),
        _n("Google", "chiefExecutiveOfficer", "Pichai", 2015, None),
        # --- Twitter -> X Corp rename chain + CEO succession
        _n("Twitter Inc", "renamedTo", "X Corp", 2023, None),
        _n("Twitter Inc", "chiefExecutiveOfficer", "Dorsey", 2006, 2015),
        _n("Twitter Inc", "chiefExecutiveOfficer", "Agrawal", 2021, 2022),
        _n("X Corp", "chiefExecutiveOfficer", "Musk", 2022, 2023),
        _n("X Corp", "chiefExecutiveOfficer", "Yaccarino", 2023, None),
        # --- Alphabet parent-of-Google (valid 2015-> )
        _n("Google", "parentCompany", "Alphabet", 2015, None),
        _n("Alphabet", "chiefExecutiveOfficer", "Page", 2015, 2019),
        _n("Alphabet", "chiefExecutiveOfficer", "Pichai", 2019, None),
    ]


@pytest.fixture
async def store(tmp_path: Path):
    s = NuggetStore(db_path=tmp_path / "chains.db")
    for n in _corpus():
        await s.aadd(n)
    try:
        yield s
    finally:
        await s.aclose()


@pytest.mark.asyncio
async def test_succession_over_google_ceos(store: NuggetStore) -> None:
    # Use the "ceo" alias to confirm canonicalisation at lookup time.
    chain = await store.achain_succession(
        subject="Google", predicate="ceo"
    )
    assert chain.chain_type == "succession"
    assert [n.fact.object for n in chain.nuggets] == [
        "Schmidt",
        "Page",
        "Pichai",
    ]
    assert len(chain.edges) == 2


@pytest.mark.asyncio
async def test_rename_forward_twitter_to_x(store: NuggetStore) -> None:
    chain = await store.achain_rename(subject="Twitter Inc")
    assert [n.fact.object for n in chain.nuggets] == ["X Corp"]


@pytest.mark.asyncio
async def test_rename_backward_from_x_corp(store: NuggetStore) -> None:
    chain = await store.achain_rename(
        subject="X Corp", direction="backward"
    )
    subjects = [n.fact.subject for n in chain.nuggets]
    assert subjects == ["Twitter Inc"]


@pytest.mark.asyncio
async def test_join_google_parent_ceo_2020(store: NuggetStore) -> None:
    chain = await store.achain_join(
        start=("Google", "parentCompany"),
        then=["ceo"],  # alias, canonicalises at lookup
        as_of=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert len(chain.nuggets) == 2
    # Hop 1: Google's parent = Alphabet
    assert chain.nuggets[0].fact.object == "Alphabet"
    # Hop 2: Alphabet's CEO at 2020 = Pichai (Pichai took over in 2019)
    assert chain.nuggets[1].fact.object == "Pichai"


@pytest.mark.asyncio
async def test_succession_as_of_filters_history(store: NuggetStore) -> None:
    chain = await store.achain_succession(
        subject="Google",
        predicate="ceo",  # alias, canonicalises at lookup
        as_of=datetime(2013, 1, 1, tzinfo=UTC),
    )
    assert [n.fact.object for n in chain.nuggets] == ["Schmidt", "Page"]


@pytest.mark.asyncio
async def test_rename_empty_for_non_rename_subject(
    store: NuggetStore,
) -> None:
    # Google has a parentCompany predicate, but not a renaming predicate.
    chain = await store.achain_rename(subject="Google")
    assert chain.nuggets == ()
