"""Tests for ``nuggetindex.integrations.haystack.doctor``.

The whole module is skipped if Haystack isn't installed. Tests confirm
the shim:

* Iterates a Haystack DocumentStore and translates each stored Document
  into the duck-typed shape ``scan_index`` consumes.
* Returns a ``DoctorReport`` with the canonical four scores.
* Propagates the ``mode`` argument through to ``scan_index``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

haystack = pytest.importorskip("haystack")

from haystack import Document as HaystackDocument  # noqa: E402
from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: E402

from nuggetindex.audit.doctor import DoctorReport  # noqa: E402
from nuggetindex.integrations.haystack import doctor  # noqa: E402


def _seed_store() -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    store.write_documents(
        [
            HaystackDocument(
                id="d1",
                content="Microsoft acquired LinkedIn for $26.2 billion.",
                meta={"source_date": datetime(2016, 6, 14, tzinfo=UTC).isoformat()},
            ),
            HaystackDocument(
                id="d2",
                content="Microsoft acquired LinkedIn for $26.4 billion.",
                meta={"source_date": datetime(2016, 6, 14, tzinfo=UTC).isoformat()},
            ),
            HaystackDocument(
                id="d3",
                content="Twitter Inc. was renamed to X Corp. in 2023.",
                meta={"url": "https://example.com/twitter-rename"},
            ),
        ]
    )
    return store


async def test_haystack_doctor_shim_translates_and_scans() -> None:
    store = _seed_store()
    report = await doctor(store, mode="fast", sample_size=10)
    assert isinstance(report, DoctorReport)
    assert report.sample_mode == "fast"
    assert len(report.scores) == 4
    assert {s.dimension for s in report.scores} == {
        "temporal_depth",
        "temporal_drift",
        "conflict_surface",
        "rename_events",
    }


async def test_haystack_doctor_shim_detects_rename() -> None:
    # Sanity: the translation populated ``text`` from ``content`` so the
    # trigger heuristics actually fire on the scanned corpus.
    store = _seed_store()
    report = await doctor(store, mode="fast", sample_size=10)
    rename = next(s for s in report.scores if s.dimension == "rename_events")
    assert rename.percentage > 0.0


async def test_haystack_doctor_shim_delegates_to_scan_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verify the shim passes its arguments through verbatim to scan_index.
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

    doctor_mod = importlib.import_module("nuggetindex.integrations.haystack.doctor")
    monkeypatch.setattr(doctor_mod, "scan_index", fake_scan_index)
    store = _seed_store()
    await doctor_mod.doctor(store, mode="fast", sample_size=7, rng_seed=42)

    assert captured["mode"] == "fast"
    assert captured["sample_size"] == 7
    assert captured["rng_seed"] == 42
    docs = list(captured["docs"])
    assert len(docs) == 3
    assert {d.source_id for d in docs} == {"d1", "d2", "d3"}
    # source_date parsing: ISO string -> datetime
    d1 = next(d for d in docs if d.source_id == "d1")
    assert isinstance(d1.source_date, datetime)
    # url meta -> uri
    d3 = next(d for d in docs if d.source_id == "d3")
    assert d3.uri == "https://example.com/twitter-rename"
