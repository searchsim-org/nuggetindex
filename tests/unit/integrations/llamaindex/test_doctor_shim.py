"""Tests for ``nuggetindex.integrations.llamaindex.doctor``.

The whole module is skipped if LlamaIndex isn't installed. Tests confirm
the shim:

* Maps a list of ``TextNode`` / ``Document`` into the duck-typed shape
  ``scan_index`` consumes.
* Maps a ``VectorStoreIndex``-like object (something exposing
  ``.docstore.docs``) into the same shape.
* Returns a ``DoctorReport`` with the canonical four scores.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import Document as LlamaDocument  # noqa: E402
from llama_index.core.schema import TextNode  # noqa: E402

from nuggetindex.audit.doctor import DoctorReport  # noqa: E402
from nuggetindex.integrations.llamaindex import doctor  # noqa: E402


def _seed_nodes() -> list[TextNode]:
    return [
        TextNode(
            id_="d1",
            text="Microsoft acquired LinkedIn for $26.2 billion.",
            metadata={
                "file_path": "/docs/d1.txt",
                "source_date": datetime(2016, 6, 14, tzinfo=UTC).isoformat(),
            },
        ),
        TextNode(
            id_="d2",
            text="Microsoft acquired LinkedIn for $26.4 billion.",
            metadata={
                "file_path": "/docs/d2.txt",
                "source_date": datetime(2016, 6, 14, tzinfo=UTC).isoformat(),
            },
        ),
        TextNode(
            id_="d3",
            text="Twitter Inc. was renamed to X Corp. in 2023.",
            metadata={"url": "https://example.com/twitter-rename"},
        ),
    ]


async def test_llamaindex_doctor_shim_translates_and_scans_from_list() -> None:
    nodes = _seed_nodes()
    report = await doctor(nodes, mode="fast", sample_size=10)
    assert isinstance(report, DoctorReport)
    assert report.sample_mode == "fast"
    assert len(report.scores) == 4
    assert {s.dimension for s in report.scores} == {
        "temporal_depth",
        "temporal_drift",
        "conflict_surface",
        "rename_events",
    }


async def test_llamaindex_doctor_shim_detects_rename() -> None:
    nodes = _seed_nodes()
    report = await doctor(nodes, mode="fast", sample_size=10)
    rename = next(s for s in report.scores if s.dimension == "rename_events")
    assert rename.percentage > 0.0


async def test_llamaindex_doctor_shim_accepts_documents() -> None:
    # ``Document`` is a ``BaseNode`` subclass; the shim should handle it
    # just as well as ``TextNode``.
    docs = [
        LlamaDocument(
            id_="d1",
            text="Microsoft acquired LinkedIn for $26.2 billion.",
            metadata={"source_date": datetime(2016, 6, 14, tzinfo=UTC).isoformat()},
        ),
        LlamaDocument(
            id_="d2",
            text="Twitter Inc. was renamed to X Corp. in 2023.",
            metadata={},
        ),
    ]
    report = await doctor(docs, mode="fast", sample_size=10)
    assert report.sample_mode == "fast"
    assert len(report.scores) == 4


async def test_llamaindex_doctor_shim_accepts_docstore_like_index() -> None:
    # Simulate a ``VectorStoreIndex``-shaped object: any object with
    # ``.docstore.docs`` mapping node_id -> BaseNode.
    nodes = _seed_nodes()

    class _FakeDocstore:
        def __init__(self, nodes: list[TextNode]) -> None:
            self.docs = {n.node_id: n for n in nodes}

    class _FakeIndex:
        def __init__(self, nodes: list[TextNode]) -> None:
            self.docstore = _FakeDocstore(nodes)

    index = _FakeIndex(nodes)
    report = await doctor(index, mode="fast", sample_size=10)
    assert isinstance(report, DoctorReport)
    assert report.sample_mode == "fast"
    assert len(report.scores) == 4


async def test_llamaindex_doctor_shim_delegates_to_scan_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_scan_index(**kwargs: Any) -> DoctorReport:
        captured.update(kwargs)
        return DoctorReport(
            sample_mode="fast",
            scores=[],
            verdict="low",
            rendered_markdown="stub",
        )

    import importlib

    doctor_mod = importlib.import_module(
        "nuggetindex.integrations.llamaindex.doctor"
    )
    monkeypatch.setattr(doctor_mod, "scan_index", fake_scan_index)
    nodes = _seed_nodes()
    await doctor_mod.doctor(nodes, mode="fast", sample_size=7, rng_seed=42)

    assert captured["mode"] == "fast"
    assert captured["sample_size"] == 7
    assert captured["rng_seed"] == 42
    docs = list(captured["docs"])
    assert len(docs) == 3
    assert {d.source_id for d in docs} == {"d1", "d2", "d3"}
    # file_path meta -> uri
    d1 = next(d for d in docs if d.source_id == "d1")
    assert d1.uri == "/docs/d1.txt"
    # url meta -> uri when file_path is absent
    d3 = next(d for d in docs if d.source_id == "d3")
    assert d3.uri == "https://example.com/twitter-rename"
    # source_date parsing
    assert isinstance(d1.source_date, datetime)
