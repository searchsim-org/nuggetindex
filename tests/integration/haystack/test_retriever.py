"""Tests for ``NuggetIndexRetriever`` (Haystack 2.x ``@component`` adapter).

Haystack components are inherently sync (their ``run()`` signature is a
regular function), and ``NuggetIndexRetriever.run`` calls ``asyncio.run``
internally. That collides with a running loop, so these tests are
deliberately sync and seed the ``NuggetStore`` via an ``asyncio.run``
helper rather than relying on an async pytest fixture.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("haystack")

from haystack import Document, Pipeline, component  # noqa: E402

from nuggetindex.core.enums import LifecycleStatus, NuggetKind  # noqa: E402
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.integrations.haystack import NuggetIndexRetriever  # noqa: E402
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


def _seed_store(db_path: Path) -> NuggetStore:
    """Build a pre-seeded store synchronously.

    Haystack's ``run()`` is sync and calls ``asyncio.run`` internally, so we
    cannot seed the store inside an async fixture (nested loops raise).
    This helper builds and seeds a store via a single ``asyncio.run`` so
    the subsequent ``run()`` calls have a fresh top-level loop.
    """

    async def _inner() -> NuggetStore:
        store = NuggetStore(db_path)
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
        return store

    return asyncio.run(_inner())


def _close_store(store: NuggetStore) -> None:
    asyncio.run(store.aclose())


def test_retriever_is_haystack_component(tmp_path: Path) -> None:
    """Haystack's ``@component`` decorator marks instances with ``__haystack_*__`` sockets."""
    store = _seed_store(tmp_path / "hs.db")
    try:
        inst = NuggetIndexRetriever(store=store, top_k=5)
        assert hasattr(inst, "__haystack_input__")
        assert hasattr(inst, "__haystack_output__")
    finally:
        _close_store(store)


def test_run_returns_documents(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs.db")
    try:
        retriever = NuggetIndexRetriever(store=store, top_k=5)
        out = retriever.run(
            query="Google CEO",
            query_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert set(out.keys()) == {"documents"}
        docs = out["documents"]
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)
        # Content is the nugget's fact text.
        assert any("Google" in (d.content or "") or "CEO" in (d.content or "") for d in docs)
        # Required meta fields are present.
        for d in docs:
            for key in (
                "nugget_id",
                "subject",
                "predicate",
                "object",
                "valid_from",
                "valid_until",
                "status",
                "confidence",
                "source",
                "evidence",
            ):
                assert key in d.meta
            # Fused score propagates onto the document.
            assert d.score is not None
    finally:
        _close_store(store)


def test_contested_docs_get_disputed_prefix(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs.db")
    try:
        retriever = NuggetIndexRetriever(
            store=store,
            view="active_contested",
            top_k=10,
            flag_contested=True,
        )
        out = retriever.run(
            query="Foo",
            query_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
        docs = out["documents"]
        contested = [d for d in docs if d.meta["status"] == "contested"]
        assert contested, "expected at least one contested doc in fixture"
        for d in contested:
            assert (d.content or "").startswith("[DISPUTED] ")
    finally:
        _close_store(store)


def test_flag_contested_false_omits_prefix(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs.db")
    try:
        retriever = NuggetIndexRetriever(
            store=store,
            view="active_contested",
            top_k=10,
            flag_contested=False,
        )
        out = retriever.run(
            query="Foo",
            query_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
        for d in out["documents"]:
            assert not (d.content or "").startswith("[DISPUTED] ")
    finally:
        _close_store(store)


# A tiny stub component at module scope — Haystack's ``@component`` decorator
# calls ``typing.get_type_hints(run)`` at decoration time, which evaluates
# annotations against this module's globals (we have ``from __future__ import
# annotations`` on, so they're strings). Defining the stub at module scope
# with ``Document`` already imported avoids NameError at decoration time.
@component
class _JoinContent:
    @component.output_types(joined=str)
    def run(self, documents: list[Document]) -> dict[str, str]:
        return {"joined": "\n".join(d.content or "" for d in documents)}


def test_retriever_composes_in_pipeline(tmp_path: Path) -> None:
    """End-to-end: NuggetIndexRetriever  downstream-stub inside a ``Pipeline``."""
    store = _seed_store(tmp_path / "hs.db")
    try:
        pipeline = Pipeline()
        pipeline.add_component("retriever", NuggetIndexRetriever(store=store, top_k=5))
        pipeline.add_component("joiner", _JoinContent())
        pipeline.connect("retriever.documents", "joiner.documents")

        out = pipeline.run(
            {
                "retriever": {
                    "query": "Google",
                    "query_time": datetime(2020, 1, 1, tzinfo=UTC),
                }
            }
        )
        joined = out["joiner"]["joined"]
        assert isinstance(joined, str)
        assert "Google" in joined
    finally:
        _close_store(store)
