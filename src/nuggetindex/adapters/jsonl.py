"""JSONL-backed :class:`~nuggetindex.adapters.base.CorpusSource` adapter.

Thin wrapper around a ``.jsonl`` / ``.ndjson`` file. Used when the caller
has a flat export of their corpus rather than a live retrieval backend.

Note: ``topic_diverse`` sampling can't actually run queries against a flat
file, so on :class:`JsonlCorpus` it degrades to uniform sampling (with a
warning). Users who want *real* topic-diverse coverage should wire
``auto()`` to a live adapter like :class:`~nuggetindex.adapters.vespa.VespaCorpus`.
"""

from __future__ import annotations

import json
import random
import re
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.pipeline.constructor import Document


def _parse_source_date(raw: Any) -> datetime | None:
    """Lenient ISO-8601 parser (mirrors ``auto.py``'s behaviour)."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokeniser used for the fallback BM25 ranker."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


@dataclass
class JsonlCorpus:
    """A :class:`CorpusSource` backed by a flat JSONL file.

    The file must have one JSON object per line with at least ``source_id``
    and ``text`` keys. Optional fields: ``uri``, ``source_date``.

    Sampling modes:

    * ``uniform`` / ``random_ids`` - deterministic random shuffle (seed ``0``)
      followed by a head-slice of ``n``.
    * ``topic_diverse`` - degrades to uniform (with a :class:`UserWarning`)
      because we can't issue queries against a flat file. Use
      :class:`~nuggetindex.adapters.vespa.VespaCorpus` for real
      topic-diverse sampling.

    ``search()`` runs BM25 (via ``rank_bm25``) if available and falls back
    to token-overlap scoring otherwise.
    """

    path: Path

    def _iter(self) -> list[Document]:
        from nuggetindex.pipeline.constructor import Document

        docs: list[Document] = []
        with Path(self.path).open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                source_id = row.get("source_id")
                text = row.get("text")
                if not source_id or not text:
                    raise ValueError(
                        f"{self.path}:{lineno}: missing required 'source_id' or 'text'"
                    )
                docs.append(
                    Document(
                        source_id=str(source_id),
                        text=str(text),
                        uri=row.get("uri"),
                        source_date=_parse_source_date(row.get("source_date")),
                    )
                )
        return docs

    async def sample(
        self,
        *,
        mode: Literal["topic_diverse", "uniform", "random_ids"],
        n: int,
    ) -> list[Document]:
        docs = self._iter()
        if mode == "topic_diverse":
            warnings.warn(
                "topic_diverse mode is approximated by uniform sampling "
                "over a JSONL file; use a live corpus adapter for true "
                "topic-diverse sampling.",
                UserWarning,
                stacklevel=2,
            )
        rng = random.Random(0)
        rng.shuffle(docs)
        return docs[:n]

    async def search(self, query: str, *, limit: int) -> list[Document]:
        docs = self._iter()
        if not docs:
            return []
        q_tokens = _tokenise(query)
        if not q_tokens:
            return docs[:limit]
        # Prefer rank_bm25 when available (it's a listed dep, but guard
        # defensively so tests that monkey-patch it out still succeed).
        try:
            from rank_bm25 import BM25Okapi

            tokenised_corpus = [_tokenise(d.text) for d in docs]
            bm25 = BM25Okapi(tokenised_corpus)
            scores = bm25.get_scores(q_tokens)
            ranked = sorted(
                zip(docs, scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            return [d for d, s in ranked[:limit] if s > 0.0] or docs[:limit]
        except Exception:  # pragma: no cover - fallback path
            # Cheap token-overlap scoring.
            q_set = set(q_tokens)
            scored = [(d, sum(1 for t in _tokenise(d.text) if t in q_set)) for d in docs]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            return [d for d, s in scored[:limit] if s > 0] or docs[:limit]


__all__ = ["JsonlCorpus"]
