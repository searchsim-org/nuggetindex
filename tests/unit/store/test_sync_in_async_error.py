"""Sync-in-async guard: sync wrappers called from a running event loop must
raise a clear ``RuntimeError`` that points to the async equivalent rather
than the cryptic default ``asyncio.run() cannot be called from a running
event loop`` message. (findings-A4)
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.pipeline.constructor import Document
from nuggetindex.store import NuggetStore


def _any_nugget(obj: str = "Pichai") -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google", predicate="ceo", object=obj, text=f"{obj} is CEO"
        ),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="doc-1", evidence_span="x"),),
    )


def _any_document() -> Document:
    return Document(source_id="doc-1", text="Google is a company.")


@pytest.mark.asyncio
async def test_count_sync_inside_event_loop_raises_clearly(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    with pytest.raises(RuntimeError, match="acount"):
        store.count()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name,async_name",
    [
        ("count", "acount"),
        ("close", "aclose"),
        ("add", "aadd"),
        ("ingest", "aingest"),
        ("retrieve", "aretrieve"),
        ("chain_succession", "achain_succession"),
        ("chain_rename", "achain_rename"),
        ("chain_join", "achain_join"),
        ("get_source_passages", "aget_source_passages"),
        ("candidate_keys", "acandidate_keys"),
    ],
)
async def test_sync_wrappers_raise_clear_error(
    tmp_db_path, method_name, async_name
):
    store = NuggetStore(db_path=tmp_db_path)
    method = getattr(store, method_name, None)
    if method is None:
        pytest.skip(f"method {method_name!r} not present")
    with pytest.raises(RuntimeError, match=async_name):
        # Give each sync method its minimum args; args can be irrelevant since
        # the guard must fire BEFORE any real work.
        if method_name == "add":
            method(_any_nugget())
        elif method_name == "ingest":
            method(_any_document())
        elif method_name == "retrieve":
            method("q")
        elif method_name == "chain_succession":
            method(subject="x", predicate="p")
        elif method_name == "chain_rename":
            method(subject="x")
        elif method_name == "chain_join":
            method(start=("x", "p"), then=["q"])
        elif method_name == "get_source_passages":
            method([])
        else:
            method()
