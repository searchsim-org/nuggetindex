"""Behavioural tests for :func:`nuggetindex.audit.seeds.propose_seeds`."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex.audit.seeds import SeedCandidate, SeedProposal, propose_seeds
from nuggetindex.pipeline.constructor import Document


@pytest.fixture
def synthetic_corpus() -> list[Document]:
    return [
        Document(
            source_id="d1",
            text="Larry Page became CEO of Google in 2011.",
            source_date=datetime(2011, 4, 5, tzinfo=UTC),
        ),
        Document(
            source_id="d2",
            text="Sundar Pichai became CEO of Google in 2015.",
            source_date=datetime(2015, 10, 3, tzinfo=UTC),
        ),
        Document(
            source_id="d3",
            text="Twitter Inc. was renamed to X Corp. in 2023.",
            source_date=datetime(2023, 4, 12, tzinfo=UTC),
        ),
        Document(
            source_id="d4",
            text="Microsoft acquired LinkedIn for 26 billion.",
            source_date=datetime(2016, 6, 14, tzinfo=UTC),
        ),
        Document(
            source_id="d5",
            text="Microsoft acquired GitHub in 2018.",
            source_date=datetime(2018, 6, 4, tzinfo=UTC),
        ),
        Document(
            source_id="d6",
            text="Apple founded by Steve Jobs in 1976.",
            source_date=datetime(1976, 4, 1, tzinfo=UTC),
        ),
        Document(
            source_id="d7",
            text="Facebook was renamed to Meta in 2021.",
            source_date=datetime(2021, 10, 28, tzinfo=UTC),
        ),
    ] * 3  # triple to get min_entity_frequency=3


async def test_shape(synthetic_corpus: list[Document]) -> None:
    p = await propose_seeds(
        docs=synthetic_corpus,
        budget=10,
        sample_size=50,
        min_entity_frequency=2,
    )
    assert isinstance(p, SeedProposal)
    assert len(p.seeds) <= 10
    assert p.seeds, "expected at least one seed"
    for s in p.seeds:
        assert isinstance(s, SeedCandidate)
        assert s.query
        assert s.kind in {
            "functional",
            "rename",
            "disputed_check",
            "entity_coverage",
        }


async def test_budget_respected(synthetic_corpus: list[Document]) -> None:
    p = await propose_seeds(
        docs=synthetic_corpus,
        budget=3,
        sample_size=50,
        min_entity_frequency=1,
    )
    assert len(p.seeds) <= 3


async def test_covers_entities(synthetic_corpus: list[Document]) -> None:
    p = await propose_seeds(
        docs=synthetic_corpus,
        budget=20,
        sample_size=50,
        min_entity_frequency=1,
    )
    entities = {s.entity.lower() for s in p.seeds}
    hits = {"google", "microsoft", "apple", "twitter", "facebook"} & entities
    assert len(hits) >= 3, (
        f"seeds didn't cover enough distinct entities: {entities}"
    )


async def test_empty_corpus() -> None:
    p = await propose_seeds(docs=[], budget=10, sample_size=10)
    assert p.seeds == []
    assert p.total_candidates_considered == 0


async def test_rendered_markdown_has_table(
    synthetic_corpus: list[Document],
) -> None:
    p = await propose_seeds(
        docs=synthetic_corpus,
        budget=5,
        sample_size=50,
        min_entity_frequency=1,
    )
    md = p.rendered_markdown.lower()
    assert "seed" in md
    assert any(
        kind in md for kind in ["functional", "rename", "entity_coverage"]
    )
