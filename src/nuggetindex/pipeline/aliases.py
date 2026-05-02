"""Tiered entity-alias resolver.

Resolves a surface-form mention (e.g. ``"Space X"``, ``"Microsoft Corp"``,
``"microsoft"``) to a stable canonical form drawn from a running pool of
already-seen entity strings. No hardcoded alias tables -- the resolver learns
the pool as it sees mentions.

Tiers (cheapest first):

1. Exact match -- the surface form already appears in the canonical pool.
2. Normalized match -- casefold + strip punctuation/whitespace + strip common
   legal-entity suffixes (``Inc`` / ``Corp`` / ``Ltd`` / ``GmbH`` ...) before
   comparing. ``"Space X"`` and ``"SpaceX"`` both normalize to ``"spacex"``.
3. Character-n-gram TF-IDF cosine similarity >= ``sim_threshold`` -- catches
   typos and spelling variants once the pool has >= 2 canonicals. Lazy import
   of ``scikit-learn``.
4. (Optional) sentence-transformers embedding cosine >= ``emb_threshold`` --
   gated by the ``extras`` package. Left as a stub; the string-sim tier is
   sufficient for the current demo corpus.

On a match the resolver returns the pool's existing canonical (the first
form seen for that group). On no match it adds the mention to the pool as
a new canonical.

Store persistence is **out of scope** for this module: the alias map lives in
memory for the duration of an ingest run. A later task may persist it to
SQLite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Legal-entity suffixes: strip these before comparing. Conservative set, case-
# insensitive. Ordered from most-specific (with trailing dot) to least so the
# loop in ``_normalize`` consumes "inc." before "inc" when both would match.
_LEGAL_SUFFIXES: tuple[str, ...] = (
    "incorporated",
    "inc.",
    "inc",
    "corporation",
    "corp.",
    "corp",
    "limited",
    "ltd.",
    "ltd",
    "l.l.c.",
    "llc",
    "p.l.c.",
    "plc",
    "gmbh",
    "ag",
    "s.a.",
    "sa",
    "co.",
    "company",
    "holdings",
    "group",
)
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _strip_legal_suffixes(s: str) -> str:
    """Iteratively strip trailing legal-entity suffixes from ``s``.

    Works on the casefolded / whitespace-normalized form. Called from
    :func:`_normalize` and exposed separately only for clarity / testability.
    """
    changed = True
    while changed:
        changed = False
        for suf in _LEGAL_SUFFIXES:
            if s == suf:
                # Entire string is just the suffix -- leave it alone so we
                # don't return the empty string.
                continue
            if s.endswith(" " + suf):
                s = s[: -(len(suf) + 1)].rstrip()
                changed = True
                break
    return s


def _normalize(mention: str) -> str:
    """Casefold + strip punctuation + collapse whitespace + strip legal suffixes.

    ``"Space X"``, ``"SpaceX"``, ``"Space X Inc."`` -> ``"spacex"``.
    Whitespace inside the normalized form is removed so e.g. ``"Space X"`` and
    ``"SpaceX"`` collapse to the same key.
    """
    # Strip punctuation FIRST (so "Inc." becomes "Inc "), then casefold and
    # whitespace-normalize. Casefolding before punctuation-stripping is
    # equivalent but keeping this order matches the comments.
    s = _PUNCT_RE.sub(" ", mention.casefold())
    s = _WS_RE.sub(" ", s).strip()
    s = _strip_legal_suffixes(s)
    # Collapse remaining spaces so "space x" and "spacex" fuse.
    return s.replace(" ", "")


@dataclass(frozen=True)
class AliasResolution:
    """Outcome of one :meth:`AliasResolver.resolve` call."""

    canonical: str
    confidence: float  # 1.0 for exact / normalized / new; cosine for string_sim / embedding
    method: str  # "exact" | "normalized" | "string_sim" | "embedding" | "new" | "empty"
    resolved_from: str  # the raw mention handed to resolve()


@dataclass
class AliasResolver:
    """Stateful resolver; maintains a running pool of canonical forms.

    The resolver is deliberately single-threaded and per-ingest-run: the
    canonical pool is an in-memory list of first-seen surface forms. Callers
    that need cross-run persistence must serialize / restore the pool via
    :meth:`pool` + :meth:`seed` (the latter is equivalent to calling
    :meth:`resolve` on each stored form in insertion order).
    """

    sim_threshold: float = 0.88  # char-ngram cosine
    emb_threshold: float = 0.90  # sentence-transformers cosine
    embedding_model: str | None = None
    ngram_range: tuple[int, int] = (3, 4)  # char-wb ngrams
    # Length-ratio guard for the string-sim tier. Char-ngram cosine over a
    # substring hit can falsely bind very different lengths (e.g. "Apple" vs
    # "Следующие CEO Apple?") because of shared n-grams inside the longer
    # string. Rejecting matches when the shorter/longer length ratio drops
    # below this threshold keeps the resolver algorithmic without resorting
    # to a lookup table.
    length_ratio_floor: float = 0.5

    # Internal: normalized -> canonical (first-seen wins).
    _norm_to_canonical: dict[str, str] = field(default_factory=dict)
    # Internal: list of canonicals in insertion order (for TF-IDF vocabulary reuse).
    _canonicals: list[str] = field(default_factory=list)
    # Lazy: TF-IDF vectorizer + matrix; rebuilt when the canonical pool grows.
    _vectorizer: Any = None
    _matrix: Any = None
    _vectorizer_dirty: bool = True

    def seed(self, mentions: list[str]) -> None:
        """Pre-populate the canonical pool from an iterable of known forms.

        Equivalent to calling :meth:`resolve` on each mention in order and
        discarding the result. Useful for priming the resolver from the
        existing store's known subjects/objects at ingest-run start.
        """
        for m in mentions:
            self.resolve(m)

    def resolve(self, mention: str) -> AliasResolution:
        raw = mention.strip()
        if not raw:
            return AliasResolution(
                canonical="",
                confidence=0.0,
                method="empty",
                resolved_from=raw,
            )
        # Tier 1: exact.
        if raw in self._canonicals:
            return AliasResolution(
                canonical=raw,
                confidence=1.0,
                method="exact",
                resolved_from=raw,
            )
        # Tier 2: normalized.
        norm = _normalize(raw)
        if norm and norm in self._norm_to_canonical:
            return AliasResolution(
                canonical=self._norm_to_canonical[norm],
                confidence=1.0,
                method="normalized",
                resolved_from=raw,
            )
        # Tier 3: string-sim (char n-gram cosine via TF-IDF). Only worthwhile
        # once the pool has >= 2 canonicals.
        if len(self._canonicals) >= 2:
            hit = self._string_sim_lookup(raw)
            if hit is not None:
                canonical, score = hit
                # Memoize so future exact-alias queries short-circuit.
                if norm:
                    self._norm_to_canonical[norm] = canonical
                return AliasResolution(
                    canonical=canonical,
                    confidence=score,
                    method="string_sim",
                    resolved_from=raw,
                )
        # Tier 4: embedding similarity (optional / stubbed).
        if self.embedding_model and len(self._canonicals) >= 2:
            hit = self._embedding_lookup(raw)
            if hit is not None:
                canonical, score = hit
                if norm:
                    self._norm_to_canonical[norm] = canonical
                return AliasResolution(
                    canonical=canonical,
                    confidence=score,
                    method="embedding",
                    resolved_from=raw,
                )
        # No match -- add to pool as a new canonical.
        self._canonicals.append(raw)
        if norm:
            self._norm_to_canonical[norm] = raw
        self._vectorizer_dirty = True
        return AliasResolution(
            canonical=raw,
            confidence=1.0,
            method="new",
            resolved_from=raw,
        )

    def _string_sim_lookup(self, raw: str) -> tuple[str, float] | None:
        """Char-ngram TF-IDF cosine over the canonical pool.

        Lazy-imports ``scikit-learn``; returns ``None`` (i.e. "tier skipped")
        if the import fails, preserving the module-level "no extras required"
        guarantee.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return None
        if self._vectorizer_dirty or self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                min_df=1,
                norm="l2",
                lowercase=True,
            )
            # ``fit_transform`` raises ValueError on empty vocabulary; we
            # guard on ``len(self._canonicals) >= 2`` before calling, so
            # this path is safe.
            self._matrix = self._vectorizer.fit_transform(self._canonicals)
            self._vectorizer_dirty = False
        q = self._vectorizer.transform([raw])
        sims = (self._matrix @ q.T).toarray().ravel()
        idx = int(sims.argmax())
        score = float(sims[idx])
        if score < self.sim_threshold:
            return None
        # Length-ratio guard: a char-ngram cosine hit between strings of
        # wildly different lengths is almost always a substring artefact, not
        # an alias. Reject unless shorter/longer >= ``length_ratio_floor``.
        candidate = self._canonicals[idx]
        a = len(raw)
        b = len(candidate)
        if a == 0 or b == 0:
            return None
        ratio = min(a, b) / max(a, b)
        if ratio < self.length_ratio_floor:
            return None
        return candidate, score

    def _embedding_lookup(self, raw: str) -> tuple[str, float] | None:
        """Sentence-transformers cosine (stub).

        Lazy-imports ``sentence-transformers``; implementation is intentionally
        left as a no-op because the string-sim tier is sufficient for the
        current demo corpus. A later task can flesh this out if needed.
        """
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            return None
        return None

    def pool(self) -> list[str]:
        """Inspect the current canonical pool (debugging / tests)."""
        return list(self._canonicals)
