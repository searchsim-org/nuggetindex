"""Stratified / reservoir sampling for the doctor scan.

This module is stdlib-only for the default code paths; the ``"language"`` and
``"composite"`` stratifiers optionally delegate to ``langdetect`` via
:mod:`nuggetindex.audit.heuristics.language` (with a deterministic
dependency-free fallback). The near-duplicate dedup pass uses a pure-Python
SimHash -- no external deps.

Two core code paths, unchanged from the original design:

* **Concrete sequences** (``list``/``tuple`` of :class:`Document`). We know the
  population size up front, so we can bucket by ``source_date`` decile
  (default) or by domain / language / composite, and sample uniformly across
  buckets.

* **Streaming iterables** (sync or async). We fall back to Vitter's Algorithm R
  reservoir sampling. A streaming input does not let us compute per-bucket
  counts in a single pass, so every ``stratify_by`` mode degrades to uniform
  reservoir sampling in that case. ``n_total`` is returned as ``None``.

All randomness is sourced from ``random.Random(rng_seed)`` so every call is
reproducible for a given seed. The SimHash dedup pass is also deterministic.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import AsyncIterable, Iterable, Sequence
from datetime import datetime
from typing import Literal, cast
from urllib.parse import urlparse

from nuggetindex.audit.heuristics.language import _detect_language
from nuggetindex.pipeline.constructor import Document

StratifyBy = Literal["source_date", "none", "domain", "language", "composite"]

_KNOWN_STRATIFIERS: tuple[str, ...] = (
    "source_date",
    "none",
    "domain",
    "language",
    "composite",
)

_SIMHASH_BITS = 64
_SIMHASH_HAMMING_THRESHOLD = 3


async def stratified_sample(
    docs: AsyncIterable[Document] | Iterable[Document],
    *,
    sample_size: int,
    stratify_by: StratifyBy = "source_date",
    rng_seed: int = 0,
    dedup_near_duplicates: bool = False,
) -> tuple[list[Document], int | None]:
    """Return ``(sampled_docs, n_total)``.

    Parameters
    ----------
    docs:
        Either a concrete sequence / sync iterable, or an async iterable. When
        the input is a concrete :class:`~collections.abc.Sequence` we know
        ``n_total`` up front; streaming inputs return ``n_total=None``.
    sample_size:
        Target number of docs to return. If ``sample_size`` is greater than or
        equal to the population size (concrete inputs only), all docs are
        returned in a deterministic shuffle.
    stratify_by:
        One of:

        * ``"source_date"`` -- bucket concrete inputs into 10 deciles of the
          observed ``source_date`` range plus an "unknown" bucket for docs
          without a date.
        * ``"none"`` -- uniform random sample.
        * ``"domain"`` -- bucket by ``urlparse(doc.uri).netloc``; docs with
          ``uri is None`` fall into a synthetic ``"unknown_domain"`` bucket.
        * ``"language"`` -- detect language via
          :func:`nuggetindex.audit.heuristics.language._detect_language` and
          bucket by the returned code.
        * ``"composite"`` -- bucket by the ``(language, domain)`` cross
          product.

        Streaming inputs ignore ``stratify_by`` and always use reservoir
        sampling (bucket counts are unknown in a single pass).
    rng_seed:
        Seed for the internal :class:`random.Random` instance. Also seeds
        ``langdetect`` to keep language detection reproducible.
    dedup_near_duplicates:
        When ``True``, run a SimHash-based near-duplicate filter over the
        sampled docs, dropping docs whose 64-bit SimHash is within 3 bits
        Hamming distance of an already-kept doc. After dedup, the sample is
        topped back up to ``sample_size`` by drawing additional non-duplicate
        docs from the remaining population (respecting the chosen
        stratification where possible). Default ``False`` -- this is an
        opt-in quality pass, measured separately.

    Returns
    -------
    tuple
        ``(sampled_docs, n_total)`` where ``n_total`` is the known population
        size for concrete inputs or ``None`` for streaming inputs.
    """
    rng = random.Random(rng_seed)

    if stratify_by not in _KNOWN_STRATIFIERS:
        raise ValueError(f"stratify_by must be one of {_KNOWN_STRATIFIERS}, got {stratify_by!r}")

    if sample_size <= 0:
        # Drain streaming inputs to stay consistent; concrete inputs are cheap.
        n_total: int | None
        if isinstance(docs, Sequence):
            n_total = len(docs)
        elif isinstance(docs, AsyncIterable):
            count = 0
            async for _ in docs:
                count += 1
            n_total = count
        else:
            n_total = sum(1 for _ in docs)
        return [], n_total

    if isinstance(docs, Sequence):
        seq = cast(Sequence[Document], docs)
        sampled, n_total = _sample_concrete(
            seq, sample_size=sample_size, stratify_by=stratify_by, rng=rng
        )
        if dedup_near_duplicates:
            sampled = _dedup_and_top_up(
                sampled,
                population=seq,
                sample_size=sample_size,
                stratify_by=stratify_by,
                rng=rng,
            )
        return sampled, n_total

    if isinstance(docs, AsyncIterable):
        sampled = await _reservoir_async(docs, sample_size=sample_size, rng=rng)
    else:
        # Sync iterable that is NOT a Sequence -- treat as a one-shot stream.
        sampled = _reservoir_sync(docs, sample_size=sample_size, rng=rng)

    if dedup_near_duplicates:
        # Streaming input: population is exhausted, so we can only dedup
        # down -- no top-up. We still run the filter so the sample's
        # information content is honest.
        sampled = _dedup_only(sampled)
    return sampled, None


def _sample_concrete(
    docs: Sequence[Document],
    *,
    sample_size: int,
    stratify_by: str,
    rng: random.Random,
) -> tuple[list[Document], int]:
    n_total = len(docs)
    if n_total == 0:
        return [], 0

    # sample_size >= n_total: return a deterministic shuffle of everything.
    if sample_size >= n_total:
        out = list(docs)
        rng.shuffle(out)
        return out, n_total

    if stratify_by == "source_date":
        return _stratified_by_source_date(docs, sample_size=sample_size, rng=rng), n_total

    if stratify_by == "none":
        picked = rng.sample(range(n_total), sample_size)
        return [docs[i] for i in picked], n_total

    if stratify_by == "domain":
        buckets = _bucket_indices(docs, key=_domain_of)
        return _sample_from_buckets(docs, buckets, sample_size=sample_size, rng=rng), n_total

    if stratify_by == "language":
        buckets = _bucket_indices(docs, key=_language_of)
        return _sample_from_buckets(docs, buckets, sample_size=sample_size, rng=rng), n_total

    if stratify_by == "composite":
        buckets = _bucket_indices(docs, key=_composite_of)
        return _sample_from_buckets(docs, buckets, sample_size=sample_size, rng=rng), n_total

    raise ValueError(  # pragma: no cover -- validated up-front
        f"stratify_by must be one of {_KNOWN_STRATIFIERS}, got {stratify_by!r}"
    )


def _domain_of(doc: Document) -> str:
    """Return the host component of ``doc.uri`` or ``"unknown_domain"``."""
    if not doc.uri:
        return "unknown_domain"
    try:
        host = urlparse(doc.uri).netloc
    except ValueError:
        return "unknown_domain"
    return host or "unknown_domain"


def _language_of(doc: Document) -> str:
    """Return the detected language code of ``doc.text``."""
    return _detect_language(doc.text or "")


def _composite_of(doc: Document) -> tuple[str, str]:
    """Cross-product bucket key: ``(language, domain)``."""
    return (_language_of(doc), _domain_of(doc))


def _bucket_indices(
    docs: Sequence[Document],
    *,
    key,
) -> list[list[int]]:
    """Group document indices by ``key(doc)``; return the list of buckets.

    Bucket order is the insertion order of first occurrence of each key, which
    keeps downstream sampling deterministic given the fixed RNG seed.
    """
    buckets: dict[object, list[int]] = {}
    for i, d in enumerate(docs):
        buckets.setdefault(key(d), []).append(i)
    return [b for b in buckets.values() if b]


def _sample_from_buckets(
    docs: Sequence[Document],
    buckets: list[list[int]],
    *,
    sample_size: int,
    rng: random.Random,
) -> list[Document]:
    """Draw ``ceil(sample_size / n_buckets)`` per bucket, trim to ``sample_size``.

    Shared implementation used by ``domain`` / ``language`` / ``composite``.
    Also used as the fallback top-up for dedup padding.
    """
    if not buckets:
        return []

    per_bucket = math.ceil(sample_size / len(buckets))

    picked: list[int] = []
    for bucket in buckets:
        if len(bucket) <= per_bucket:
            picked.extend(bucket)
        else:
            picked.extend(rng.sample(bucket, per_bucket))

    rng.shuffle(picked)

    if len(picked) > sample_size:
        picked = picked[:sample_size]
    elif len(picked) < sample_size:
        chosen = set(picked)
        leftovers = [i for i in range(len(docs)) if i not in chosen]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: sample_size - len(picked)])

    return [docs[i] for i in picked]


def _stratified_by_source_date(
    docs: Sequence[Document],
    *,
    sample_size: int,
    rng: random.Random,
) -> list[Document]:
    """Bucket by source_date decile + unknown, sample uniformly across buckets."""
    dated: list[tuple[datetime, int]] = [
        (d.source_date, i) for i, d in enumerate(docs) if d.source_date is not None
    ]
    unknown_indices = [i for i, d in enumerate(docs) if d.source_date is None]

    buckets: list[list[int]] = []

    if dated:
        # Sort dates to compute decile thresholds. We compute 9 internal cut
        # points (10th, 20th, ... 90th percentile by position) and bucket each
        # dated doc by its rank within the sorted list. This is equivalent to
        # rank-based deciles and avoids timestamp arithmetic edge cases.
        dated_sorted = sorted(dated, key=lambda p: p[0])
        n_dated = len(dated_sorted)
        # Rank-based decile assignment: bucket = min(9, floor(rank * 10 / n_dated)).
        decile_buckets: list[list[int]] = [[] for _ in range(10)]
        for rank, (_, orig_idx) in enumerate(dated_sorted):
            b = min(9, (rank * 10) // n_dated)
            decile_buckets[b].append(orig_idx)
        buckets.extend([b for b in decile_buckets if b])

    if unknown_indices:
        buckets.append(unknown_indices)

    if not buckets:
        # Shouldn't be reachable (n_total > 0 checked by caller), but guard.
        return []

    return _sample_from_buckets(docs, buckets, sample_size=sample_size, rng=rng)


def _reservoir_sync(
    docs: Iterable[Document],
    *,
    sample_size: int,
    rng: random.Random,
) -> list[Document]:
    """Vitter's Algorithm R over a synchronous iterable."""
    reservoir: list[Document] = []
    for i, doc in enumerate(docs):
        if i < sample_size:
            reservoir.append(doc)
        else:
            j = rng.randint(0, i)
            if j < sample_size:
                reservoir[j] = doc
    return reservoir


async def _reservoir_async(
    docs: AsyncIterable[Document],
    *,
    sample_size: int,
    rng: random.Random,
) -> list[Document]:
    """Vitter's Algorithm R over an asynchronous iterable."""
    reservoir: list[Document] = []
    i = 0
    async for doc in docs:
        if i < sample_size:
            reservoir.append(doc)
        else:
            j = rng.randint(0, i)
            if j < sample_size:
                reservoir[j] = doc
        i += 1
    return reservoir


# --------------------------------------------------------------------------- #
# SimHash near-duplicate detection
# --------------------------------------------------------------------------- #


def _simhash(text: str, n: int = _SIMHASH_BITS) -> int:
    """Classic SimHash over whitespace tokens; returns an ``n``-bit integer.

    Deterministic by construction: uses ``hashlib.md5`` per token and
    aggregates per-bit sign counts. Empty input returns ``0``.
    """
    tokens = text.split()
    if not tokens:
        return 0
    v = [0] * n
    mask = (1 << n) - 1
    for tok in tokens:
        h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest(), "big") & mask
        for i in range(n):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum((1 << i) for i, c in enumerate(v) if c > 0)


def _hamming(a: int, b: int) -> int:
    """64-bit Hamming distance via XOR popcount."""
    return bin(a ^ b).count("1")


def _dedup_only(sampled: list[Document]) -> list[Document]:
    """Filter ``sampled`` in-order, dropping near-duplicates. No top-up."""
    kept: list[Document] = []
    kept_hashes: list[int] = []
    for doc in sampled:
        h = _simhash(doc.text or "")
        if any(_hamming(h, kh) <= _SIMHASH_HAMMING_THRESHOLD for kh in kept_hashes):
            continue
        kept.append(doc)
        kept_hashes.append(h)
    return kept


def _dedup_and_top_up(
    sampled: list[Document],
    *,
    population: Sequence[Document],
    sample_size: int,
    stratify_by: str,
    rng: random.Random,
) -> list[Document]:
    """Near-duplicate filter with population top-up.

    Drops near-duplicates from ``sampled``, then pads back up to
    ``sample_size`` by drawing additional non-duplicate docs from the
    unsampled remainder of ``population``. Draw order respects the active
    stratification where possible.
    """
    filtered = _dedup_only(sampled)
    if len(filtered) >= sample_size:
        return filtered[:sample_size]

    kept_ids = {id(d) for d in filtered}
    kept_hashes = [_simhash(d.text or "") for d in filtered]

    # Candidate pool: everything not already kept.
    leftover_indices = [i for i, d in enumerate(population) if id(d) not in kept_ids]
    if not leftover_indices:
        return filtered

    # Use the existing stratification to order the leftovers where possible.
    key_fn = _bucket_key_for(stratify_by)
    if key_fn is not None:
        leftover_buckets: dict[object, list[int]] = {}
        for i in leftover_indices:
            leftover_buckets.setdefault(key_fn(population[i]), []).append(i)
        # Shuffle within each bucket, then round-robin across buckets so
        # padding stays broadly representative.
        bucket_lists: list[list[int]] = []
        for idxs in leftover_buckets.values():
            rng.shuffle(idxs)
            bucket_lists.append(idxs)
        ordered_leftovers: list[int] = []
        while any(bucket_lists):
            for bucket in bucket_lists:
                if bucket:
                    ordered_leftovers.append(bucket.pop(0))
            bucket_lists = [b for b in bucket_lists if b]
    else:
        ordered_leftovers = list(leftover_indices)
        rng.shuffle(ordered_leftovers)

    need = sample_size - len(filtered)
    for i in ordered_leftovers:
        if need <= 0:
            break
        doc = population[i]
        h = _simhash(doc.text or "")
        if any(_hamming(h, kh) <= _SIMHASH_HAMMING_THRESHOLD for kh in kept_hashes):
            continue
        filtered.append(doc)
        kept_hashes.append(h)
        need -= 1

    return filtered


def _bucket_key_for(stratify_by: str):
    """Return the bucket-key function for ``stratify_by``, or ``None``."""
    if stratify_by == "domain":
        return _domain_of
    if stratify_by == "language":
        return _language_of
    if stratify_by == "composite":
        return _composite_of
    if stratify_by == "source_date":
        return _source_date_decile_bucket_key
    return None


def _source_date_decile_bucket_key(doc: Document) -> str:
    """Coarse date-bucket label for padding. Falls back to "unknown"."""
    if doc.source_date is None:
        return "source_date:unknown"
    # A year-level bucket is a cheap approximation to the rank-based decile
    # used for the primary sample; it is only used for top-up ordering so a
    # stable per-doc label is sufficient.
    return f"source_date:{doc.source_date.year}"
