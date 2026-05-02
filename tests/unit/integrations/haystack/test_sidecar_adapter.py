"""Tests for ``nuggetindex.integrations.haystack.sidecar``.

The whole module is skipped if Haystack isn't installed. Tests confirm
the adapter:

* Inserts a synthetic ``nuggetindex-governance`` document when the sidecar
  router opts in (temporal / functional-predicate query).
* Passes through unchanged when the router declines (noise query).

Tests are deliberately sync: ``NuggetSidecarComponent.run`` calls the
sync :meth:`Sidecar.handle` wrapper which uses ``asyncio.run`` internally,
and nested loops raise. The store is seeded via a single ``asyncio.run``
helper before the ``run()`` call, matching the idiom used by the existing
Haystack retriever integration test.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("haystack")

from haystack import Document as HaystackDocument  # noqa: E402

from nuggetindex import NuggetStore  # noqa: E402
from nuggetindex.core.enums import (  # noqa: E402
    EpistemicRank,
    LifecycleStatus,
    NuggetKind,
)
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.integrations.haystack import NuggetSidecarComponent  # noqa: E402
from nuggetindex.sidecar import Sidecar  # noqa: E402


def _seed_store(db_path: Path) -> NuggetStore:
    async def _inner() -> NuggetStore:
        store = NuggetStore(db_path=db_path)
        now = datetime.now(tz=UTC)
        n = Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject="Google",
                predicate="chiefExecutiveOfficer",
                object="Larry Page",
                text="Larry Page served as Google CEO from 2011 through 2015",
            ),
            validity=ValidityInterval(
                start=datetime(2011, 4, 4, tzinfo=UTC),
                end=datetime(2015, 10, 2, tzinfo=UTC),
            ),
            epistemic=EpistemicState(
                status=LifecycleStatus.ACTIVE,
                rank=EpistemicRank.NORMAL,
                confidence=0.9,
            ),
            provenance=(
                ProvenanceRecord(
                    source_id="wiki",
                    evidence_span="Larry Page served as CEO from 2011 to 2015.",
                    char_start=0,
                    char_end=50,
                    created_at=now,
                ),
            ),
            extraction_confidence=0.9,
        )
        await store.aadd(n)
        return store

    return asyncio.run(_inner())


def test_haystack_sidecar_component_inserts_governance_doc(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "s.db")
    sidecar = Sidecar(store=store, mode="offline-curated")
    component = NuggetSidecarComponent(sidecar=sidecar)
    docs = [HaystackDocument(id="d1", content="some retrieved text")]
    result = component.run(query="who was Google's CEO in 2013?", documents=docs)
    assert "documents" in result
    gov = next(
        (d for d in result["documents"] if d.id == "nuggetindex-governance"), None
    )
    assert gov is not None
    assert "Larry Page" in gov.content
    ids = [d.id for d in result["documents"]]
    assert ids[0] == "nuggetindex-governance"
    assert "d1" in ids


def test_haystack_sidecar_component_passthrough(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "s.db")
    sidecar = Sidecar(store=store, mode="offline-curated")
    component = NuggetSidecarComponent(sidecar=sidecar)
    docs = [HaystackDocument(id="d1", content="some retrieved text")]
    result = component.run(query="the sky is blue", documents=docs)
    ids = [d.id for d in result["documents"]]
    assert "nuggetindex-governance" not in ids
    assert "d1" in ids
