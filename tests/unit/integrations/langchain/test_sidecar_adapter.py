"""Tests for ``nuggetindex.integrations.langchain.sidecar``.

The whole module is skipped if LangChain isn't installed. Tests confirm
the adapter:

* Emits a non-empty ``context_block`` when the sidecar router opts in
  (temporal / functional-predicate query).
* Passes through (empty ``context_block``, empty ``nuggets``) on a noise
  query while round-tripping the original ``documents`` unchanged.

Async tests are fine here — the adapter's ``ainvoke`` is a native coroutine
that awaits :meth:`Sidecar.ahandle` directly (no nested ``asyncio.run``).
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document as LangchainDocument  # noqa: E402

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
from nuggetindex.integrations.langchain import NuggetSidecarRunnable  # noqa: E402
from nuggetindex.sidecar import Sidecar  # noqa: E402


async def _seeded_store(tmp_path: Path) -> NuggetStore:
    store = NuggetStore(db_path=tmp_path / "s.db")
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


async def test_langchain_sidecar_runnable_emits_context(tmp_path: Path) -> None:
    store = await _seeded_store(tmp_path)
    sidecar = Sidecar(store=store, mode="offline-curated")
    runnable = NuggetSidecarRunnable(sidecar=sidecar)
    docs = [LangchainDocument(id="d1", page_content="some retrieved text")]
    result = await runnable.ainvoke(
        {"query": "who was Google's CEO in 2013?", "documents": docs}
    )
    assert result["query"] == "who was Google's CEO in 2013?"
    assert result["documents"] == docs
    assert result["context_block"] != ""
    assert "Larry Page" in result["context_block"]
    assert len(result["nuggets"]) >= 1
    await store.backend.aclose()


async def test_langchain_sidecar_runnable_passthrough(tmp_path: Path) -> None:
    store = await _seeded_store(tmp_path)
    sidecar = Sidecar(store=store, mode="offline-curated")
    runnable = NuggetSidecarRunnable(sidecar=sidecar)
    docs = [LangchainDocument(id="d1", page_content="some retrieved text")]
    result = await runnable.ainvoke(
        {"query": "the sky is blue", "documents": docs}
    )
    assert result["context_block"] == ""
    assert result["nuggets"] == []
    assert result["documents"] == docs
    await store.backend.aclose()
