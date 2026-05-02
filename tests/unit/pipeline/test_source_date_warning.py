import datetime
import warnings

import pytest

from nuggetindex.core.schema import RelationSchema
from nuggetindex.pipeline.conflict import ConflictDetector
from nuggetindex.pipeline.constructor import Document, DocumentConstructor
from nuggetindex.pipeline.dedup import Deduplicator
from tests.fixtures import RuleBasedExtractor


@pytest.mark.asyncio
async def test_rule_based_missing_source_date_warns():
    ctor = DocumentConstructor(
        extractor=RuleBasedExtractor(),
        schema=RelationSchema.default(),
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(RelationSchema.default()),
    )
    doc = Document(source_id="d-1", text="Google is a company.")  # no source_date!
    with pytest.warns(UserWarning, match="source_date"):
        await ctor.aprocess(doc)


@pytest.mark.asyncio
async def test_rule_based_with_source_date_no_warning():
    ctor = DocumentConstructor(
        extractor=RuleBasedExtractor(),
        schema=RelationSchema.default(),
        deduplicator=Deduplicator(encoder=None),
        conflict_detector=ConflictDetector(RelationSchema.default()),
    )
    doc = Document(
        source_id="d-1",
        text="Google is a company.",
        source_date=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        await ctor.aprocess(doc)  # must not raise
