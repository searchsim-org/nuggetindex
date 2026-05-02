"""Tests for :mod:`nuggetindex.pipeline.aliases`."""

from __future__ import annotations

import pytest

from nuggetindex.pipeline.aliases import AliasResolver


def test_exact_match_returns_same() -> None:
    r = AliasResolver()
    r.resolve("SpaceX")
    res = r.resolve("SpaceX")
    assert res.canonical == "SpaceX"
    assert res.method == "exact"
    assert res.confidence == 1.0


def test_normalized_strips_legal_suffix() -> None:
    # "Microsoft" seen first wins as canonical. "Microsoft Corp" normalizes
    # to the same key and resolves back to "Microsoft".
    r = AliasResolver()
    r.resolve("Microsoft")
    res = r.resolve("Microsoft Corp")
    assert res.canonical == "Microsoft"
    assert res.method == "normalized"


def test_normalized_fuses_whitespace() -> None:
    # "SpaceX" and "Space X" both normalize to "spacex" once whitespace is
    # stripped; whichever is seen first wins as the canonical.
    r = AliasResolver()
    r.resolve("SpaceX")
    res = r.resolve("Space X")
    assert res.canonical == "SpaceX"
    assert res.method == "normalized"

    # Reverse order: "Space X" first, "SpaceX" resolves to it.
    r2 = AliasResolver()
    r2.resolve("Space X")
    res2 = r2.resolve("SpaceX")
    assert res2.canonical == "Space X"
    assert res2.method == "normalized"


def test_string_sim_catches_typo() -> None:
    # "Microsoft" + one other canonical to cross the >=2 threshold. A typo
    # "Microsft" should resolve back to "Microsoft" via char-ngram cosine.
    sklearn = pytest.importorskip("sklearn")  # noqa: F841 -- gate only
    r = AliasResolver(sim_threshold=0.7)
    r.resolve("Microsoft")
    r.resolve("Google")
    res = r.resolve("Microsft")
    assert res.canonical == "Microsoft"
    assert res.method == "string_sim"
    assert res.confidence >= 0.7


def test_new_mention_added_to_pool() -> None:
    r = AliasResolver()
    res = r.resolve("Anthropic")
    assert res.canonical == "Anthropic"
    assert res.method == "new"
    assert "Anthropic" in r.pool()


def test_empty_mention_returns_empty() -> None:
    r = AliasResolver()
    res = r.resolve("")
    assert res.canonical == ""
    assert res.method == "empty"
    res2 = r.resolve("   ")
    assert res2.canonical == ""
    assert res2.method == "empty"


def test_resolver_reused_across_multiple_calls_accumulates_pool() -> None:
    """Fix 10 (weak-but-valuable variant): a single resolver instance
    threaded across multiple "documents" accumulates canonicals and
    folds later mentions against earlier ones.

    This models the store-scoped lifetime without requiring the full
    NuggetStore + LLM-extractor stack. When the pool is shared, doc B's
    "Microsoft Corp" / "microsoft" collapse to doc A's "Microsoft".
    """
    r = AliasResolver()
    # --- Document A ---
    doc_a_subjects = ["Microsoft", "Satya Nadella"]
    for mention in doc_a_subjects:
        res = r.resolve(mention)
        assert res.method == "new"
    # --- Document B (new in a separate "doc", same resolver) ---
    # Legal-suffix normalization folds back to "Microsoft".
    res_b1 = r.resolve("Microsoft Corp")
    assert res_b1.canonical == "Microsoft"
    assert res_b1.method == "normalized"
    # Casefold + whitespace normalization folds "microsoft" to "Microsoft".
    res_b2 = r.resolve("microsoft")
    assert res_b2.canonical == "Microsoft"
    assert res_b2.method == "normalized"
    # Pool should contain only the first-seen canonicals -- no duplicates
    # have been introduced by doc B's mentions.
    assert r.pool() == doc_a_subjects


def test_resolver_seeded_from_prior_pool_catches_cross_doc_alias() -> None:
    """Direct model of fix 10's pre-load step: seed a resolver from the
    store's already-known canonicals, then resolving a variant form from
    a new doc returns the seeded canonical instead of adding a duplicate.
    """
    r = AliasResolver()
    # Pretend the backend returned these from ``adistinct_entities``.
    r.seed(["Microsoft", "Google", "Satya Nadella"])
    # New doc references the same org with a legal-suffix variant.
    res = r.resolve("Microsoft Corporation")
    assert res.canonical == "Microsoft"
    # Pool must not have grown: no duplicate "Microsoft Corporation".
    assert "Microsoft Corporation" not in r.pool()
    assert r.pool() == ["Microsoft", "Google", "Satya Nadella"]
