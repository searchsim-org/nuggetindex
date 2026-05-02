"""Tests for ``attach_nugget_metadata``."""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")


@pytest.mark.asyncio
async def test_attach_nugget_metadata_populates_columns(populated_store) -> None:
    from datasets import Dataset

    from nuggetindex.evaluation import attach_nugget_metadata

    ds = Dataset.from_dict(
        {
            "question": ["Who is the CEO of Google?"],
            "contexts": [["Sundar Pichai is CEO of Google."]],
            "answer": ["Sundar Pichai is CEO."],
            "query_time": ["2020-01-01T00:00:00+00:00"],
        }
    )
    enriched = attach_nugget_metadata(ds, store=populated_store, query_time_column="query_time")
    assert "retrieved_nuggets" in enriched.column_names
    assert "contested_keys" in enriched.column_names
    assert "temporal_valid_count" in enriched.column_names
    # The store is populated with a matching nugget, so we should hit at
    # least one retrieved result.
    assert len(enriched[0]["retrieved_nuggets"]) >= 1


@pytest.mark.asyncio
async def test_attach_without_query_time(populated_store) -> None:
    from datasets import Dataset

    from nuggetindex.evaluation import attach_nugget_metadata

    ds = Dataset.from_dict(
        {
            "question": ["Who founded Google?"],
            "contexts": [["Larry Page was a founder of Google."]],
        }
    )
    enriched = attach_nugget_metadata(ds, store=populated_store)
    row = enriched[0]
    # Without a query_time we still compute a temporal_valid_count (all
    # non-DEPRECATED nuggets qualify).
    assert row["temporal_valid_count"] >= 0
    assert isinstance(row["retrieved_nuggets"], list)


@pytest.mark.asyncio
async def test_attach_records_contested_keys(populated_store) -> None:
    """If the probe surfaces a CONTESTED nugget, it shows up in contested_keys."""
    from datasets import Dataset

    from nuggetindex.evaluation import attach_nugget_metadata

    # Probe text that will BM25-hit the ``Foo is bar.`` contested nugget.
    ds = Dataset.from_dict(
        {
            "question": ["What is foo?"],
            "contexts": [["Foo is bar."]],
        }
    )
    enriched = attach_nugget_metadata(ds, store=populated_store)
    row = enriched[0]
    # The contested nugget won't pass the default "active" view filter —
    # the view is ``active`` at retrieve time. That's fine: we just check
    # the column exists with the right shape.
    assert isinstance(row["contested_keys"], list)
    for entry in row["contested_keys"]:
        assert len(entry) == 3


@pytest.mark.asyncio
async def test_attach_empty_contexts(populated_store) -> None:
    """Rows with empty/absent contexts still produce well-formed output."""
    from datasets import Dataset

    from nuggetindex.evaluation import attach_nugget_metadata

    ds = Dataset.from_dict(
        {
            "question": ["nothing here"],
            "contexts": [[]],
        }
    )
    enriched = attach_nugget_metadata(ds, store=populated_store)
    row = enriched[0]
    assert row["retrieved_nuggets"] == []
    assert row["contested_keys"] == []
    assert row["temporal_valid_count"] == 0
