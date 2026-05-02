"""Tests for ``NuggetStore.acandidate_keys`` (v0.2.1 discovery helper)."""
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


def _make(
    obj: str,
    year: int,
    *,
    subject: str = "Google",
    predicate: str = "chiefExecutiveOfficer",
) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=f"{obj} is {predicate} of {subject}",
        ),
        validity=ValidityInterval(start=datetime(year, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id=f"d-{subject}-{obj}",
                evidence_span=f"{obj} is {predicate}",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_candidate_keys_subject_substring(tmp_db_path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_make("Schmidt", 2001))
        await store.aadd(_make("Pichai", 2015))
        keys = await store.acandidate_keys(subject_contains="google")
        assert ("Google", "chiefExecutiveOfficer", "global") in keys
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_candidate_keys_predicate_substring(tmp_db_path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        await store.aadd(_make("Pichai", 2015))
        keys = await store.acandidate_keys(predicate_contains="chief")
        assert any("chiefExecutiveOfficer" in k[1] for k in keys)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_candidate_keys_limit_respected(tmp_db_path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        # Distinct (subject, predicate, scope) triples: keep predicate
        # varying so each row is a fresh key.
        for pred in ("chiefExecutiveOfficer", "chairperson", "founder"):
            await store.aadd(_make("X", 2020, predicate=pred))
        keys = await store.acandidate_keys(subject_contains="google", limit=1)
        assert len(keys) <= 1
    finally:
        await store.aclose()
