"""Unit tests for :class:`nuggetindex.adapters.jsonl.JsonlCorpus`."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from nuggetindex.adapters.jsonl import JsonlCorpus


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _synthetic_rows(n: int) -> list[dict[str, str]]:
    """50-doc synthetic corpus; a few contain the token 'pizza'."""
    rows: list[dict[str, str]] = []
    for i in range(n):
        if i % 17 == 0:
            text = f"doc {i}: a piece about pizza and toppings."
        else:
            text = f"doc {i}: some unrelated text about topic {i}."
        rows.append({"source_id": f"d{i}", "text": text})
    return rows


@pytest.mark.asyncio
async def test_sample_uniform_returns_requested_count(tmp_path: Path) -> None:
    """Uniform sampling over a 50-doc file returns exactly n docs."""
    path = tmp_path / "corpus.jsonl"
    _write_jsonl(path, _synthetic_rows(50))

    corpus = JsonlCorpus(path=path)
    docs = await corpus.sample(mode="uniform", n=10)
    assert len(docs) == 10
    # All returned items are Document instances with real source_ids.
    assert all(d.source_id.startswith("d") for d in docs)


@pytest.mark.asyncio
async def test_sample_topic_diverse_degrades_to_uniform_with_warning(
    tmp_path: Path,
) -> None:
    """topic_diverse on a JSONL file emits a UserWarning and samples uniformly."""
    path = tmp_path / "corpus.jsonl"
    _write_jsonl(path, _synthetic_rows(50))

    corpus = JsonlCorpus(path=path)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        docs = await corpus.sample(mode="topic_diverse", n=10)
    assert len(docs) == 10
    assert any(
        issubclass(w.category, UserWarning) and "topic_diverse" in str(w.message) for w in captured
    ), f"expected a UserWarning mentioning topic_diverse; got {captured!r}"


@pytest.mark.asyncio
async def test_search_matches_query_keyword(tmp_path: Path) -> None:
    """`search("pizza")` returns docs whose text contains 'pizza'."""
    path = tmp_path / "corpus.jsonl"
    _write_jsonl(path, _synthetic_rows(50))

    corpus = JsonlCorpus(path=path)
    hits = await corpus.search("pizza", limit=3)
    assert hits, "expected at least one hit for 'pizza'"
    assert all("pizza" in d.text.lower() for d in hits)


@pytest.mark.asyncio
async def test_sample_random_ids_alias(tmp_path: Path) -> None:
    """random_ids on JSONL uses the same deterministic-shuffle path."""
    path = tmp_path / "corpus.jsonl"
    _write_jsonl(path, _synthetic_rows(20))
    corpus = JsonlCorpus(path=path)
    docs = await corpus.sample(mode="random_ids", n=5)
    assert len(docs) == 5
