"""Shared fixtures for LlamaIndex integration tests.

``populated_store`` gives each test a fresh SQLite-backed NuggetStore
pre-seeded with a handful of deterministic nuggets — enough to exercise the
retriever and governance postprocessor without depending on a live LLM.
Mirrors the LangChain ``conftest.py`` fixture.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

# Skip the whole package when llama-index-core isn't installed — mirrors the
# ``pytest.importorskip`` in each test module but also keeps collection
# errors out of the way on minimal dev setups.
pytest.importorskip("llama_index.core")

from nuggetindex.core.enums import LifecycleStatus, NuggetKind  # noqa: E402
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.store.base import NuggetStore  # noqa: E402


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
async def populated_store(tmp_path: Path):
    """A NuggetStore pre-seeded with a few nuggets covering each lifecycle state."""
    store = NuggetStore(tmp_path / "li.db")
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
    # A contested one so we can exercise the [DISPUTED] prefix path.
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
