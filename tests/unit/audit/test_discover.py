"""Behavioural tests for :func:`nuggetindex.audit.discover.discover_schema`."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex.audit.discover import (
    PredicateProposal,  # noqa: F401 -- surface-test re-export
    SchemaProposal,  # noqa: F401 -- surface-test re-export
    discover_schema,
)
from nuggetindex.pipeline.constructor import Document


@pytest.fixture
def business_corpus() -> list[Document]:
    """10 seed texts x 3 copies each so ``min_frequency=3`` catches predicates."""
    return [
        Document(
            source_id=f"d{i}_{k}",
            text=text,
            source_date=datetime(2020, 1, 1, tzinfo=UTC),
        )
        for k in range(3)
        for i, text in enumerate(
            [
                "Larry Page became CEO of Google in 2011.",
                "Sundar Pichai became CEO of Google in 2015.",
                "Satya Nadella became CEO of Microsoft in 2014.",
                "Tim Cook became CEO of Apple.",
                "Microsoft acquired LinkedIn.",
                "Microsoft acquired GitHub.",
                "Google acquired YouTube.",
                "Twitter was renamed to X in 2023.",
                "Facebook was renamed to Meta.",
                "Meta announced new features.",
            ]
        )
    ]


@pytest.mark.asyncio
async def test_discovers_chief_executive_officer(business_corpus):
    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    names = {p.name for p in prop.predicates}
    assert "chiefExecutiveOfficer" in names


@pytest.mark.asyncio
async def test_event_log_cardinality_on_announced(business_corpus):
    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=1)
    ann = next((p for p in prop.predicates if "announce" in p.name.lower()), None)
    # Not strictly required that "announced" surfaces from triggers.py
    # (depends on doc count vs threshold); only check the shape if it does.
    if ann is not None:
        assert ann.cardinality == "event_log"


@pytest.mark.asyncio
async def test_functional_cardinality_on_ceo(business_corpus):
    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    ceo = next(
        (p for p in prop.predicates if p.name == "chiefExecutiveOfficer"),
        None,
    )
    assert ceo is not None
    assert ceo.cardinality == "functional"


@pytest.mark.asyncio
async def test_rendered_yaml_is_valid(business_corpus):
    import yaml

    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    parsed = yaml.safe_load(prop.rendered_yaml)
    assert "predicates" in parsed
    assert isinstance(parsed["predicates"], dict)


@pytest.mark.asyncio
async def test_min_frequency_filters_noise():
    # A handful of CEO docs + a single one-off doc with a rare predicate.
    docs = [
        Document(
            source_id="d1",
            text="Larry Page became CEO of Google.",
            source_date=datetime(2020, 1, 1, tzinfo=UTC),
        )
    ] * 5 + [
        Document(
            source_id="rare",
            text="Atoms exchange valence electrons.",
            source_date=datetime(2020, 1, 1, tzinfo=UTC),
        )
    ]
    prop = await discover_schema(docs=docs, sample_size=20, min_frequency=3)
    names = {p.name for p in prop.predicates}
    assert "chiefExecutiveOfficer" in names
    # "exchange" predicate shouldn't surface — only appears once, if at all.
    assert not any("exchange" in n.lower() for n in names)


@pytest.mark.asyncio
async def test_empty_corpus():
    prop = await discover_schema(docs=[], sample_size=10)
    assert prop.predicates == []


@pytest.mark.asyncio
async def test_merge_proposal_adds_new_predicates(business_corpus):
    from nuggetindex.audit.discover import merge_proposal
    from nuggetindex.core.schema import RelationSchema

    base = RelationSchema.default()
    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    merged = merge_proposal(base, prop, accept_all=True)

    # Every proposal predicate must either exist in base OR be representable in
    # the merged schema (canonicalize returns the same name back, or the name
    # is reachable via ``is_functional`` / ``cardinality``).
    for p in prop.predicates:
        assert p.name in base.renaming_predicates or merged.canonicalize(p.name) == p.name


@pytest.mark.asyncio
async def test_merge_proposal_respects_accepted_names(business_corpus):
    from nuggetindex.audit.discover import merge_proposal
    from nuggetindex.core.schema import RelationSchema

    base = RelationSchema.default()
    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    if not prop.predicates:
        pytest.skip("no predicates discovered in this test env")

    first_name = prop.predicates[0].name
    merged = merge_proposal(base, prop, accepted_names={first_name})
    # At minimum, merged schema should accept `first_name` as a valid
    # canonical form.
    assert merged.canonicalize(first_name) == first_name


@pytest.mark.asyncio
async def test_rendered_yaml_merges_back_into_schema(business_corpus):
    """The proposal YAML must round-trip through RelationSchema.from_yaml."""
    from nuggetindex.core.schema import RelationSchema

    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    if not prop.predicates:
        pytest.skip("no predicates discovered in this test env")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proposal.yaml"
        path.write_text(prop.rendered_yaml, encoding="utf-8")
        # Should parse cleanly as a RelationSchema.
        schema = RelationSchema.from_yaml(path)
        for p in prop.predicates:
            assert schema.canonicalize(p.name) == p.name


@pytest.mark.asyncio
async def test_predicate_proposal_has_examples(business_corpus):
    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    assert prop.predicates, "expected at least one predicate"
    for p in prop.predicates:
        # Up to 3 examples, each a parenthesised triple.
        assert 0 <= len(p.examples) <= 3
        for ex in p.examples:
            assert ex.startswith("(") and ex.endswith(")")


@pytest.mark.asyncio
async def test_frequency_and_sample_metadata(business_corpus):
    prop = await discover_schema(docs=business_corpus, sample_size=30, min_frequency=2)
    assert prop.n_docs_sampled > 0
    assert prop.n_docs_total == len(business_corpus)
    # Every proposal's frequency is a positive int.
    for p in prop.predicates:
        assert p.frequency >= 2
