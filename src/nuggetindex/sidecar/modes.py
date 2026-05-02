"""Resolution strategies for the sidecar runtime.

Two modes are supported:

* :class:`OfflineCurated` — queries a pre-built
  :class:`~nuggetindex.NuggetStore` alongside the original retriever. Good for
  frozen corpora where the store has been indexed ahead of time.
* :class:`JustInTime` — extracts facts live from the retriever's top-K
  passages without mutating the store. Good for live corpora where
  re-indexing is impractical; the extracted nuggets are ephemeral.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from nuggetindex.core.models import Nugget
from nuggetindex.sidecar.router import RouterDecision

# Strip temporal phrases from the BM25 query text so FTS5 doesn't AND in
# year-tokens or stopwords that aren't in the indexed text. The temporal
# intent is already captured in ``decision.query_time`` and applied as a
# validity filter — we don't need it echoed in the keyword query.
_TEMPORAL_PHRASE_RE = re.compile(
    r"\b(?:"
    r"in\s+\d{4}"
    r"|as\s+of(?:\s+\w+){0,3}"
    r"|when(?:\s+\w+){0,2}"
    r"|(?:before|prior\s+to|after|since)\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)

# Small, conservative stopword set for BM25 pre-cleaning. The store's FTS5
# backend AND-s all remaining tokens, so leaving common interrogatives /
# articles in the query can drop matching rows even when the substantive
# keywords are present. We strip only words that are extremely unlikely to
# appear in the indexed ``text|subject|object`` columns.
_BM25_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "who",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "of",
        "to",
        "for",
        "on",
        "at",
        "by",
        "from",
        "and",
        "or",
    }
)

# FTS5 reserved / quirky punctuation: ``?`` is a parser error, ``"`` is a
# phrase delimiter, ``'`` closes identifier quoting in some tokenisers, and
# column-filter ``:`` needs to be stripped from free-form queries.
_FTS_PUNCT_RE = re.compile(r"[?\"':]")


def _strip_temporal_phrases(query: str) -> str:
    """Remove matched temporal phrases so BM25 isn't AND'd against year tokens."""
    cleaned = _TEMPORAL_PHRASE_RE.sub(" ", query)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_bm25_query(query: str) -> str:
    """Prepare a free-form query for FTS5 MATCH.

    Strips temporal phrases (already encoded in ``decision.query_time``),
    FTS5-hostile punctuation, and a small stopword set. Returns the
    remaining substantive keywords joined by single spaces. Falls back to
    the untouched ``query`` if cleaning reduces it to nothing.
    """
    cleaned = _strip_temporal_phrases(query)
    cleaned = _FTS_PUNCT_RE.sub(" ", cleaned)
    # Drop possessive ``'s`` fragments left dangling after apostrophe removal.
    cleaned = re.sub(r"\bs\b", " ", cleaned, flags=re.IGNORECASE)
    tokens = [t for t in cleaned.split() if t.lower() not in _BM25_STOPWORDS]
    result = " ".join(tokens).strip()
    return result or query.strip()


class SidecarMode(ABC):
    """Strategy protocol for resolving nugget context from a router decision."""

    @abstractmethod
    async def aresolve(
        self,
        store: Any,
        decision: RouterDecision,
        query: str,
        top_k: int,
        *,
        original_hits: list[Any] | None = None,
        extractor: Any | None = None,
        jit_cache: Any | None = None,
    ) -> list[Nugget]:
        """Return the set of nuggets to feed the :class:`ContextFormatter`."""


class OfflineCurated(SidecarMode):
    """Query the pre-built nugget store alongside the original retriever."""

    async def aresolve(
        self,
        store: Any,
        decision: RouterDecision,
        query: str,
        top_k: int,
        *,
        original_hits: list[Any] | None = None,
        extractor: Any | None = None,
        jit_cache: Any | None = None,  # noqa: ARG002 -- accepted for API symmetry
    ) -> list[Nugget]:
        # Use the store's native aretrieve with the router's query_time.
        # view="active_contested" so the formatter can surface disputes.
        # Temporal phrases in the raw query are stripped because the store
        # FTS5 backend AND-s tokens — year-tokens not present in the indexed
        # text would drop matching rows. The temporal intent is already
        # captured in ``decision.query_time`` and applied as a validity filter.
        bm25_query = _clean_bm25_query(query)
        results = await store.aretrieve(
            query=bm25_query,
            query_time=decision.query_time,
            view="active_contested",
            top_k=top_k,
        )
        nuggets: list[Nugget] = []
        for r in results:
            n = getattr(r, "nugget", None)
            if n is not None:
                nuggets.append(n)
        return nuggets


class JustInTime(SidecarMode):
    """Extract facts from the original retriever's top-K live.

    Does NOT mutate the store — facts are extracted on the fly and returned
    as ephemeral nuggets. The caller (``Sidecar.ahandle``) should not persist
    them; offline-curated mode is the path that builds up the store over time.

    When the invoking :class:`Sidecar` exposes a ``jit_cache`` attribute (a
    :class:`~nuggetindex.sidecar.jit_cache.JITPassageCache`), each passage's
    extraction result is memoised by passage-text hash so overlapping top-K
    across similar queries don't re-pay the extractor cost. The cache is
    looked up via duck-typed attribute access (``getattr(..., "jit_cache",
    None)``) so the strategy stays usable on its own, outside the sidecar.
    """

    async def aresolve(
        self,
        store: Any,
        decision: RouterDecision,
        query: str,
        top_k: int,
        *,
        original_hits: list[Any] | None = None,
        extractor: Any | None = None,
        jit_cache: Any | None = None,
    ) -> list[Nugget]:
        if extractor is None:
            raise ValueError(
                "JustInTime.aresolve requires extractor= (the `Sidecar.extractor` attribute)."
            )
        if not original_hits:
            return []

        nuggets: list[Nugget] = []
        for hit in original_hits[:top_k]:
            text = _hit_text(hit)
            if not text:
                continue
            try:
                if jit_cache is not None:
                    results = await jit_cache.get_or_extract(text, extractor)
                else:
                    results = await extractor.aextract(text=text)
            except Exception:  # noqa: BLE001 — just-in-time is best-effort
                continue
            for r in results:
                n = getattr(r, "nugget", None)
                if n is not None:
                    nuggets.append(n)
        return nuggets


def _hit_text(hit: Any) -> str:
    """Duck-type on the caller's hit object to extract text.

    Accepts a plain :class:`str`, an object with ``.content``, ``.text`` or
    ``.page_content`` attributes, or a dict with any of those keys. Returns
    an empty string when no text can be located; callers treat that as "skip".
    """
    if isinstance(hit, str):
        return hit
    for attr in ("content", "text", "page_content"):
        val = getattr(hit, attr, None)
        if isinstance(val, str):
            return val
    if isinstance(hit, dict):
        for key in ("content", "text", "page_content"):
            v = hit.get(key)
            if isinstance(v, str):
                return v
    return ""
