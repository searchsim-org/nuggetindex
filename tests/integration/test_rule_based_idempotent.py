from datetime import UTC, datetime

import pytest

from nuggetindex import NuggetStore
from nuggetindex.pipeline.constructor import Document
from tests.fixtures import RuleBasedExtractor


@pytest.mark.asyncio
async def test_reingest_same_doc_rule_based_idempotent(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path, extractor=RuleBasedExtractor())
    doc = Document(
        source_id="d-1",
        text="Google is a company.",
        source_date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    await store.aingest(doc)
    c1 = await store.acount()
    r2 = await store.aingest(doc)
    c2 = await store.acount()
    assert c1 == c2  # idempotent
    assert r2.nuggets_added == 0  # everything was already there
    await store.aclose()
