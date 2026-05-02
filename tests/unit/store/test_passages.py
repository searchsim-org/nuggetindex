import pytest

from nuggetindex.store.backends.sqlite import SQLiteBackend


@pytest.mark.asyncio
async def test_upsert_and_get_passage(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    await b.aupsert_passage("doc-1", "https://ex.com/a", "Sundar is CEO.")
    passages = await b.aget_passages(["doc-1"])
    assert passages == {"doc-1": "Sundar is CEO."}
    await b.aclose()


@pytest.mark.asyncio
async def test_get_passages_filters_missing(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    await b.aupsert_passage("doc-1", None, "x")
    got = await b.aget_passages(["doc-1", "doc-absent"])
    assert got == {"doc-1": "x"}  # missing silently dropped
    await b.aclose()


@pytest.mark.asyncio
async def test_passage_upsert_overwrites(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    await b.aupsert_passage("doc-1", None, "first")
    await b.aupsert_passage("doc-1", None, "second")
    got = await b.aget_passages(["doc-1"])
    assert got == {"doc-1": "second"}
    await b.aclose()
