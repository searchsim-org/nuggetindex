"""Shared fixtures for the evaluation tests.

``populated_store`` mirrors the one under ``tests/integration/langchain`` —
we keep a local copy here so the evaluation tests aren't transitively
dependent on the LangChain integration tests (which are skipped when
langchain-core isn't installed).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.store.base import NuggetStore


def _make_nugget(
    *,
    subject: str,
    predicate: str,
    obj: str,
    sentence: str,
    source_id: str,
    status: LifecycleStatus = LifecycleStatus.ACTIVE,
    start: datetime | None = None,
) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text=sentence),
        validity=ValidityInterval(start=start or datetime(2019, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(status=status, confidence=0.9),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=sentence),),
    )


@pytest.fixture
def sample_nuggets() -> list[Nugget]:
    """Small nugget list used by the NLI tests.

    Mirrors the Google-succession fixture under ``tests/unit/chains`` but
    kept local so the evaluation tests don't depend on fixtures from
    another subtree.
    """
    return [
        _make_nugget(
            subject="Google",
            predicate="ceo",
            obj="Sundar Pichai",
            sentence="Sundar Pichai is CEO of Google.",
            source_id="d1",
            start=datetime(2015, 10, 2, tzinfo=UTC),
        ),
        _make_nugget(
            subject="Google",
            predicate="founder",
            obj="Larry Page",
            sentence="Larry Page was a founder of Google.",
            source_id="d2",
            start=datetime(1998, 9, 4, tzinfo=UTC),
        ),
    ]


@pytest.fixture
async def populated_store(tmp_path: Path):
    store = NuggetStore(tmp_path / "eval.db")
    await store.backend.aupsert_passage("d1", None, "Sundar Pichai is CEO of Google.")
    await store.backend.aupsert_passage("d2", None, "Larry Page was a founder of Google.")
    await store.backend.aupsert_passage("d3", None, "Foo is bar.")
    await store.aadd(
        _make_nugget(
            subject="Google",
            predicate="ceo",
            obj="Sundar Pichai",
            sentence="Sundar Pichai is CEO of Google.",
            source_id="d1",
        )
    )
    await store.aadd(
        _make_nugget(
            subject="Google",
            predicate="founder",
            obj="Larry Page",
            sentence="Larry Page was a founder of Google.",
            source_id="d2",
        )
    )
    await store.aadd(
        _make_nugget(
            subject="Foo",
            predicate="is",
            obj="bar",
            sentence="Foo is bar.",
            source_id="d3",
            status=LifecycleStatus.CONTESTED,
        )
    )
    try:
        yield store
    finally:
        await store.aclose()
