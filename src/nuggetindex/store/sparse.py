"""Sparse retrieval interface.

Thin protocol placeholder. For v0.1 the SQLite backend implements BM25 via
FTS5 directly on the `SQLiteBackend` class (see `abm25_search`). Future
non-SQLite sparse backends can implement this protocol.
"""

from __future__ import annotations

from typing import Protocol


class SparseBackend(Protocol):
    """Sparse (lexical / BM25) retrieval backend."""

    async def abm25_search(
        self,
        query: str,
        *,
        candidate_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[str, float]]: ...
