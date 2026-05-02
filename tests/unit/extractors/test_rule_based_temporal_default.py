import pytest

from tests.fixtures import RuleBasedExtractor


@pytest.mark.asyncio
async def test_emits_placeholder_validity():
    ex = RuleBasedExtractor()
    results = await ex.aextract("Google is a company.")
    assert results, "rule-based should extract something from this sentence"
    for r in results:
        assert r.nugget.validity.is_placeholder()
