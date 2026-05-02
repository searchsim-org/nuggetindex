"""Tests for ``NuggetConstructor`` (Haystack 2.x ingest component).

Haystack components are sync (``run()`` is a regular function), and
``NuggetConstructor.run`` calls ``asyncio.run`` internally. That collides
with a running loop, so these tests are sync and use ``asyncio.run`` to
set up / inspect the store.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("haystack")

from haystack import Document, Pipeline, component  # noqa: E402

from nuggetindex.integrations.haystack import NuggetConstructor  # noqa: E402
from nuggetindex.store.base import NuggetStore  # noqa: E402
from tests.fixtures import RuleBasedExtractor  # noqa: E402


def _make_store(db_path: Path) -> NuggetStore:
    """Build a store with a rule-based extractor, synchronously.

    Seeded via ``asyncio.run`` so the subsequent ``NuggetConstructor.run``
    call (which also uses ``asyncio.run``) has a fresh top-level loop.
    """
    return NuggetStore(db_path, extractor=RuleBasedExtractor())


def _close_store(store: NuggetStore) -> None:
    asyncio.run(store.aclose())


def test_constructor_is_haystack_component(tmp_path: Path) -> None:
    """Instance should expose Haystack's input/output socket metadata."""
    store = _make_store(tmp_path / "ingest.db")
    try:
        inst = NuggetConstructor(store=store)
        assert hasattr(inst, "__haystack_input__")
        assert hasattr(inst, "__haystack_output__")
    finally:
        _close_store(store)


def test_run_ingests_documents(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "ingest.db")
    try:
        constructor = NuggetConstructor(store=store)
        docs = [
            Document(content="Sundar Pichai is CEO of Google.", id="n1"),
            Document(content="Larry Page was a founder of Google.", id="n2"),
        ]
        out = constructor.run(documents=docs)
        # Docs pass through unchanged so the component composes in pipelines.
        assert out["documents"] == docs
        # The store now has rule-based-extracted nuggets persisted.
        count = asyncio.run(store.acount())
        assert count >= 1
    finally:
        _close_store(store)


def test_run_skips_blank_documents(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "ingest.db")
    try:
        constructor = NuggetConstructor(store=store)
        docs = [
            Document(content="   ", id="empty"),
            Document(content="Sundar Pichai is CEO of Google.", id="ok"),
        ]
        out = constructor.run(documents=docs)
        assert out["documents"] == docs
        # Only the non-blank id round-trips as a passage in the store.
        passages = asyncio.run(store.backend.aget_passages({"empty", "ok"}))
        assert "ok" in passages
        assert "empty" not in passages
    finally:
        _close_store(store)


def test_run_empty_list(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "ingest.db")
    try:
        constructor = NuggetConstructor(store=store)
        out = constructor.run(documents=[])
        assert out == {"documents": []}
    finally:
        _close_store(store)


def test_run_uses_document_id_as_source(tmp_path: Path) -> None:
    """Haystack documents auto-assign ids; we reuse them as ``source_id``."""
    store = _make_store(tmp_path / "ingest.db")
    try:
        constructor = NuggetConstructor(store=store)
        doc = Document(content="Sundar Pichai is CEO of Google.")
        # Capture the auto-generated id before ingest.
        source_id = doc.id
        constructor.run(documents=[doc])
        passages = asyncio.run(store.backend.aget_passages({source_id}))
        assert source_id in passages
    finally:
        _close_store(store)


# A stub upstream component at module scope; Haystack's ``@component``
# decorator calls ``typing.get_type_hints(run)`` at decoration time, and
# with ``from __future__ import annotations`` the annotations are strings
# that get evaluated against this module's globals. Defining the stub at
# module scope (with ``Document`` imported at the top) avoids NameError.
@component
class _DocSource:
    @component.output_types(documents=list[Document])
    def run(self) -> dict[str, list[Document]]:
        return {
            "documents": [
                Document(content="Sundar Pichai is CEO of Google.", id="p1"),
            ]
        }


def test_constructor_composes_in_pipeline(tmp_path: Path) -> None:
    """A loader-like source  NuggetConstructor  downstream works end-to-end."""
    store = _make_store(tmp_path / "pipeline.db")
    try:
        pipeline = Pipeline()
        pipeline.add_component("source", _DocSource())
        pipeline.add_component("constructor", NuggetConstructor(store=store))
        pipeline.connect("source.documents", "constructor.documents")

        out = pipeline.run({})
        assert len(out["constructor"]["documents"]) == 1
        # Ingest side-effect hit the store.
        assert asyncio.run(store.acount()) >= 1
    finally:
        _close_store(store)
