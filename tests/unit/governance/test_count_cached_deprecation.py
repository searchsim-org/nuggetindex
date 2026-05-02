"""Tests for the async-safe + explicit-unsafe count_cached_nuggets split."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

import pytest

from nuggetindex.governance.postprocessor import (
    GovernancePostProcessor,
    RetrievedPassage,
)
from tests.fixtures import RuleBasedExtractor


@pytest.mark.asyncio
async def test_acount_is_async_and_safe(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    try:
        assert await pp.acount_cached_nuggets() == 0
        await pp.apostprocess([RetrievedPassage(source_id="d-1", text="Google is a company.")])
        assert await pp.acount_cached_nuggets() >= 1
    finally:
        await pp.aclose()


def test_unsafe_sync_still_works(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pp.count_cached_nuggets_unsafe() == 0


def test_old_name_deprecated(tmp_path):
    pp = GovernancePostProcessor(
        cache_path=tmp_path / "s.db",
        extractor=RuleBasedExtractor(),
        query_time=datetime(2024, 6, 1, tzinfo=UTC),
    )
    with pytest.warns(DeprecationWarning, match="count_cached_nuggets"):
        pp.count_cached_nuggets()
