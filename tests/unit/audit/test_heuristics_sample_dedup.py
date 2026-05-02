"""Tests for opt-in near-duplicate dedup in ``stratified_sample`` (Fix 4)."""

from __future__ import annotations

import pytest

from nuggetindex.audit.heuristics import stratified_sample
from nuggetindex.pipeline.constructor import Document


def _mk_dup_corpus() -> list[Document]:
    """50 unique docs + 50 exact duplicates of the first 50 (different ids)."""
    uniques = [
        Document(
            source_id=f"u-{i:03d}",
            text=(
                f"Article {i}: this is a unique piece of prose about topic "
                f"number {i}. It contains enough tokens for SimHash to be "
                f"meaningful, and the word 'alpha{i}' appears here uniquely."
            ),
            uri=f"https://site-{i % 5}.example.com/a/{i}",
            source_date=None,
        )
        for i in range(50)
    ]
    duplicates = [
        Document(
            source_id=f"d-{i:03d}",
            text=uniques[i].text,  # identical text
            uri=f"https://mirror-{i % 3}.example.com/a/{i}",
            source_date=None,
        )
        for i in range(50)
    ]
    return uniques + duplicates


@pytest.mark.asyncio
async def test_dedup_near_duplicates_removes_duplicates() -> None:
    """With dedup on, the sample should have <=50 unique-text docs (with some padding slack)."""
    docs = _mk_dup_corpus()
    sampled, n_total = await stratified_sample(
        docs,
        sample_size=80,
        stratify_by="none",
        rng_seed=0,
        dedup_near_duplicates=True,
    )
    assert n_total == 100
    # Ideally the sample contains exactly the 50 unique-text docs; top-up logic
    # won't find more non-duplicates so the final size is bounded above by 50.
    assert len(sampled) <= 50, (
        f"expected <=50 after dedup (population has only 50 unique texts), "
        f"got {len(sampled)}"
    )
    # And the texts really are unique.
    assert len({d.text for d in sampled}) == len(sampled)


@pytest.mark.asyncio
async def test_dedup_off_by_default() -> None:
    """Default call path leaves duplicates in — 80 docs returned unchanged."""
    docs = _mk_dup_corpus()
    sampled, n_total = await stratified_sample(
        docs, sample_size=80, stratify_by="none", rng_seed=0
    )
    assert n_total == 100
    assert len(sampled) == 80
