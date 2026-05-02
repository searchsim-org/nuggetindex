"""Tests for ``NuggetDocumentStore`` (Haystack 2.x ``DocumentStore``).

Custom scenario tests that exercise the v0.2 behaviour:

- Haystack ``Document``  source-passage round-trip through ``write_documents``.
- ``count_documents`` mirrors passage count.
- BM25 retrieval returns the source passages the matched nuggets belong to.
- ``filter_documents`` honours the v0.1 SQL allowlist — allowed fields pass
  through as ``extra_filters`` to :meth:`NuggetStore.afilter`, disallowed
  ones raise ``ValueError``.
- ``delete_documents`` drops passages + derived nuggets.

Haystack's ``DocumentStoreBaseTests`` in ``test_document_store_base.py``
cover the protocol-level conformance; this file covers the integration glue
specific to nuggetindex semantics.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("haystack")

from haystack import Document  # noqa: E402
from haystack.document_stores.errors import DuplicateDocumentError  # noqa: E402
from haystack.document_stores.types import DuplicatePolicy  # noqa: E402

from nuggetindex import NuggetStore  # noqa: E402
from nuggetindex.core.enums import LifecycleStatus, NuggetKind  # noqa: E402
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.integrations.haystack import NuggetDocumentStore  # noqa: E402


def _close(store: NuggetStore) -> None:
    asyncio.run(store.aclose())


def test_write_and_count(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        n = ds.write_documents(
            [Document(id="d1", content="Sundar Pichai is CEO of Google.")]
        )
        assert n == 1
        assert ds.count_documents() == 1
    finally:
        _close(store)


def test_write_documents_overwrite_is_idempotent(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        ds.write_documents(
            [Document(id="d1", content="first")],
            policy=DuplicatePolicy.OVERWRITE,
        )
        ds.write_documents(
            [Document(id="d1", content="second")],
            policy=DuplicatePolicy.OVERWRITE,
        )
        assert ds.count_documents() == 1
        [doc] = ds.filter_documents()
        assert doc.id == "d1"
        assert doc.content == "second"
    finally:
        _close(store)


def test_write_documents_duplicate_fail(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        doc = Document(content="test")
        assert ds.write_documents([doc], policy=DuplicatePolicy.FAIL) == 1
        with pytest.raises(DuplicateDocumentError):
            ds.write_documents([doc], policy=DuplicatePolicy.FAIL)
    finally:
        _close(store)


def test_write_documents_duplicate_skip(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        doc = Document(content="test")
        assert ds.write_documents([doc], policy=DuplicatePolicy.SKIP) == 1
        assert ds.write_documents([doc], policy=DuplicatePolicy.SKIP) == 0
    finally:
        _close(store)


def test_write_documents_invalid_input(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        with pytest.raises(ValueError):
            ds.write_documents(["not a document"])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            ds.write_documents("not a list actually")  # type: ignore[arg-type]
    finally:
        _close(store)


def _seed_nuggets(store: NuggetStore) -> None:
    """Seed nuggets + passages so bm25_retrieval has something to return."""

    async def _inner() -> None:
        await store.backend.aupsert_passage_with_meta(
            "d1", None, "Sundar Pichai is CEO of Google.", None
        )
        await store.backend.aupsert_passage_with_meta(
            "d2", None, "Apple was founded in 1976.", None
        )
        await store.aadd(
            Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject="Google",
                    predicate="ceo",
                    object="Sundar Pichai",
                    text="Sundar Pichai is CEO of Google.",
                ),
                validity=ValidityInterval(
                    start=datetime(2019, 1, 1, tzinfo=UTC)
                ),
                epistemic=EpistemicState(
                    status=LifecycleStatus.ACTIVE, confidence=0.9
                ),
                provenance=(
                    ProvenanceRecord(
                        source_id="d1",
                        evidence_span="Sundar Pichai is CEO of Google.",
                    ),
                ),
            )
        )
        await store.aadd(
            Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject="Apple",
                    predicate="founded",
                    object="1976",
                    text="Apple was founded in 1976.",
                ),
                validity=ValidityInterval(
                    start=datetime(1976, 1, 1, tzinfo=UTC)
                ),
                epistemic=EpistemicState(
                    status=LifecycleStatus.ACTIVE, confidence=0.9
                ),
                provenance=(
                    ProvenanceRecord(
                        source_id="d2",
                        evidence_span="Apple was founded in 1976.",
                    ),
                ),
            )
        )

    asyncio.run(_inner())


def test_bm25_retrieval(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        _seed_nuggets(store)
        ds = NuggetDocumentStore(store=store)
        results = ds.bm25_retrieval("CEO Google", top_k=5)
        assert results
        # We hand back source passages, not nugget facts: the d1 passage
        # should be the top match for the CEO query.
        assert results[0].id == "d1"
        assert "Sundar" in (results[0].content or "")
    finally:
        _close(store)


def test_filter_documents_no_filter_returns_all(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        ds.write_documents(
            [
                Document(id="d1", content="first"),
                Document(id="d2", content="second"),
            ]
        )
        out = ds.filter_documents()
        assert {d.id for d in out} == {"d1", "d2"}
        assert {d.content for d in out} == {"first", "second"}
    finally:
        _close(store)


def test_filter_documents_allowlist_allowed(tmp_path: Path) -> None:
    """Allowlisted column: passes to ``afilter`` without raising."""
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        _seed_nuggets(store)
        ds = NuggetDocumentStore(store=store)
        # ``status`` is in the v0.1 allowlist — call shouldn't raise.
        out = ds.filter_documents(
            {"field": "status", "operator": "==", "value": "active"}
        )
        assert isinstance(out, list)
    finally:
        _close(store)


def test_filter_documents_allowlist_disallowed(tmp_path: Path) -> None:
    """Disallowed column raises ``ValueError`` mentioning 'unknown filter'."""
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        ds.write_documents([Document(id="d1", content="Test")])
        with pytest.raises(ValueError, match="unknown filter"):
            ds.filter_documents(
                {"field": "secret_column", "operator": "==", "value": "x"}
            )
    finally:
        _close(store)


def test_filter_documents_only_eq_operator_supported(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        with pytest.raises(ValueError, match=r"only supports '=='"):
            ds.filter_documents(
                {"field": "status", "operator": ">", "value": "active"}
            )
    finally:
        _close(store)


def test_delete_documents(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        ds.write_documents(
            [
                Document(id="d1", content="alpha"),
                Document(id="d2", content="beta"),
            ]
        )
        ds.delete_documents(["d1"])
        assert ds.count_documents() == 1
        [doc] = ds.filter_documents()
        assert doc.id == "d2"
    finally:
        _close(store)


def test_delete_documents_empty_store_ok(tmp_path: Path) -> None:
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        ds = NuggetDocumentStore(store=store)
        # Should not raise even with no data.
        ds.delete_documents(["does_not_exist"])
        assert ds.count_documents() == 0
    finally:
        _close(store)


def test_delete_documents_cascades_to_nuggets(tmp_path: Path) -> None:
    """Deleting a passage source_id drops any nugget sourced from it."""
    store = NuggetStore(db_path=tmp_path / "hs.db")
    try:
        _seed_nuggets(store)
        # Sanity: both nuggets seeded.
        assert asyncio.run(store.backend.acount()) == 2
        ds = NuggetDocumentStore(store=store)
        ds.delete_documents(["d1"])
        # The ``d1``-sourced nugget is gone; ``d2``'s nugget remains.
        remaining = asyncio.run(store.backend.acount())
        assert remaining == 1
    finally:
        _close(store)
