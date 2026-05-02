"""Predicate canonicalisation in chain methods (v0.2.1, findings-A3).

``achain_succession``, ``achain_rename``, and ``achain_join`` now pass the
supplied predicate through ``schema.canonicalize`` before SQL lookup so that
users typing an alias (e.g. ``ceo``) hit nuggets keyed under the canonical
predicate (e.g. ``chiefExecutiveOfficer``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex import NuggetStore
from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)


def _make(obj: str, year: int, predicate: str = "chiefExecutiveOfficer") -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate=predicate,
            object=obj,
            text=f"{obj} is CEO of Google",
        ),
        validity=ValidityInterval(start=datetime(year, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=f"d-{obj}", evidence_span=f"{obj} is CEO"),),
    )


def _rename(subject: str, obj: str, year: int, predicate: str = "renamedTo") -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=f"{subject} renamed to {obj}",
        ),
        validity=ValidityInterval(start=datetime(year, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id=f"rename-{subject}-{obj}",
                evidence_span=f"{subject} renamed to {obj}",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_chain_accepts_canonical_predicate(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_make("Schmidt", 2001))
        await store.aadd(_make("Page", 2011))
        await store.aadd(_make("Pichai", 2015))

        chain = await store.achain_succession(subject="Google", predicate="chiefExecutiveOfficer")
        assert [n.fact.object for n in chain.nuggets] == ["Schmidt", "Page", "Pichai"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_chain_accepts_alias_predicate(tmp_db_path):
    """v0.2.1 behaviour change: ``ceo`` canonicalises → matches."""
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_make("Schmidt", 2001))
        await store.aadd(_make("Page", 2011))
        await store.aadd(_make("Pichai", 2015))

        chain = await store.achain_succession(subject="Google", predicate="ceo")
        assert [n.fact.object for n in chain.nuggets] == ["Schmidt", "Page", "Pichai"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_chain_join_canonicalises_start_and_then(tmp_db_path):
    """Join canonicalises both ``start[1]`` and every entry of ``then``."""
    store = NuggetStore(db_path=tmp_db_path)
    try:
        # Google parentCompany -> Alphabet (valid 2015-)
        parent = Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject="Google",
                predicate="parentCompany",
                object="Alphabet",
                text="Google parent is Alphabet",
            ),
            validity=ValidityInterval(start=datetime(2015, 10, 2, tzinfo=UTC)),
            epistemic=EpistemicState(),
            provenance=(ProvenanceRecord(source_id="d-parent", evidence_span="alphabet"),),
        )
        # Alphabet ceo -> Pichai
        ceo = Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject="Alphabet",
                predicate="chiefExecutiveOfficer",
                object="Pichai",
                text="Pichai is CEO",
            ),
            validity=ValidityInterval(start=datetime(2019, 12, 3, tzinfo=UTC)),
            epistemic=EpistemicState(),
            provenance=(ProvenanceRecord(source_id="d-alphabet-ceo", evidence_span="pichai"),),
        )
        await store.aadd(parent)
        await store.aadd(ceo)

        # Alias predicate "parentOrganization" (alias of parentCompany?) —
        # actually use the alias "ceo" for the then hop, which is the
        # relevant behaviour change.
        chain = await store.achain_join(
            start=("Google", "parentCompany"),
            then=["ceo"],
            as_of=datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert [n.fact.object for n in chain.nuggets] == ["Alphabet", "Pichai"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_chain_rename_canonicalises_renaming_predicate(tmp_db_path):
    """Rename walk works the same whether stored nuggets use the canonical
    renaming predicate or a registered alias; the store's
    ``schema.renaming_predicates`` frozenset drives the SQL filter, and the
    backend's ``arename_candidates`` matches the canonical name set.
    """
    store = NuggetStore(db_path=tmp_db_path)
    try:
        # Canonical renaming predicate is ``renamedTo``. Seed a 2-step
        # rename chain: Facebook -> Meta Platforms -> Meta.
        await store.aadd(_rename("Facebook", "Meta Platforms", 2021))
        await store.aadd(_rename("Meta Platforms", "Meta", 2022))

        chain = await store.achain_rename(subject="Facebook")
        assert [n.fact.object for n in chain.nuggets] == ["Meta Platforms", "Meta"]
    finally:
        await store.aclose()
