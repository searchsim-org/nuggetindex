"""Tests for ``nuggetindex.integrations.langchain.doctor``.

The whole module is skipped if LangChain isn't installed. Tests confirm
the shim:

* Maps a concrete list of LangChain ``Document`` objects into the
  duck-typed shape ``scan_index`` consumes.
* Returns a ``DoctorReport`` with the canonical four scores.
* Refuses retriever / VectorStore sources with a helpful ``TypeError``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document as LangchainDocument  # noqa: E402

from nuggetindex.audit.doctor import DoctorReport  # noqa: E402
from nuggetindex.integrations.langchain import doctor  # noqa: E402


def _seed_corpus() -> list[LangchainDocument]:
    return [
        LangchainDocument(
            id="d1",
            page_content="Microsoft acquired LinkedIn for $26.2 billion.",
            metadata={
                "source": "news://d1",
                "source_date": datetime(2016, 6, 14, tzinfo=UTC).isoformat(),
            },
        ),
        LangchainDocument(
            id="d2",
            page_content="Microsoft acquired LinkedIn for $26.4 billion.",
            metadata={
                "source": "news://d2",
                "source_date": datetime(2016, 6, 14, tzinfo=UTC).isoformat(),
            },
        ),
        LangchainDocument(
            id="d3",
            page_content="Twitter Inc. was renamed to X Corp. in 2023.",
            metadata={"url": "https://example.com/twitter-rename"},
        ),
    ]


async def test_langchain_doctor_shim_translates_and_scans() -> None:
    docs = _seed_corpus()
    report = await doctor(docs, mode="fast", sample_size=10)
    assert isinstance(report, DoctorReport)
    assert report.sample_mode == "fast"
    assert len(report.scores) == 4
    assert {s.dimension for s in report.scores} == {
        "temporal_depth",
        "temporal_drift",
        "conflict_surface",
        "rename_events",
    }


async def test_langchain_doctor_shim_detects_rename() -> None:
    docs = _seed_corpus()
    report = await doctor(docs, mode="fast", sample_size=10)
    rename = next(s for s in report.scores if s.dimension == "rename_events")
    assert rename.percentage > 0.0


async def test_langchain_doctor_shim_delegates_to_scan_index(
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

    doctor_mod = importlib.import_module("nuggetindex.integrations.langchain.doctor")
    monkeypatch.setattr(doctor_mod, "scan_index", fake_scan_index)
    corpus = _seed_corpus()
    await doctor_mod.doctor(corpus, mode="fast", sample_size=7, rng_seed=42)

    assert captured["mode"] == "fast"
    assert captured["sample_size"] == 7
    assert captured["rng_seed"] == 42
    docs = list(captured["docs"])
    assert len(docs) == 3
    assert {d.source_id for d in docs} == {"d1", "d2", "d3"}
    # ``source`` meta -> uri
    d1 = next(d for d in docs if d.source_id == "d1")
    assert d1.uri == "news://d1"
    # ``url`` meta -> uri when ``source`` is absent
    d3 = next(d for d in docs if d.source_id == "d3")
    assert d3.uri == "https://example.com/twitter-rename"
    # source_date parsing
    assert isinstance(d1.source_date, datetime)


async def test_langchain_doctor_shim_rejects_retrievers() -> None:
    class FakeRetriever:
        def get_relevant_documents(self, query: str) -> list[Any]:  # noqa: ARG002
            return []

    with pytest.raises(TypeError, match="Iterable"):
        await doctor(FakeRetriever(), mode="fast")  # type: ignore[arg-type]


async def test_langchain_doctor_shim_rejects_vector_stores() -> None:
    class FakeVectorStore:
        def similarity_search(self, query: str, k: int = 4) -> list[Any]:  # noqa: ARG002
            return []

    with pytest.raises(TypeError, match="Iterable"):
        await doctor(FakeVectorStore(), mode="fast")  # type: ignore[arg-type]


async def test_langchain_doctor_shim_synthesises_id_when_missing() -> None:
    # LangChain Documents often ship without an explicit id. The shim
    # should hash the content so the stratified sampler's determinism
    # isn't broken by dropping in an empty string for every doc.
    corpus = [
        LangchainDocument(page_content="Microsoft acquired LinkedIn for $26.2 billion."),
        LangchainDocument(page_content="Microsoft acquired LinkedIn for $26.4 billion."),
    ]
    report = await doctor(corpus, mode="fast", sample_size=10)
    assert isinstance(report, DoctorReport)
    assert report.sample_mode == "fast"
