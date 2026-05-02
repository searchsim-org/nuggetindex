"""Query-time retrieval pipeline.

Pipeline stages:
    1. View filter (validity-at-query-time + lifecycle status)
    2. Sparse retrieval (BM25 over the candidate set)
    3. Dense retrieval (optional; skipped when ``dense_backend`` is ``None``)
    4. Fusion (RRF default; weighted min-max available for reproducibility)
    5. Result assembly (hydrate nuggets, attach component scores/ranks)

The ``Retriever`` is not instantiated directly by users; it is a thin
collaborator created lazily by ``NuggetStore.aretrieve``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nuggetindex.core.models import Nugget
from nuggetindex.retrieve.fusion import (
    reciprocal_rank_fusion,
    weighted_minmax_fusion,
)

if TYPE_CHECKING:
    from nuggetindex.store.backends.sqlite import SQLiteBackend

FusionMode = Literal["rrf", "weighted_minmax"]


class RetrievalResult(BaseModel):
    """A single ranked retrieval hit with component scores and ranks.

    ``score`` is the fused score (RRF or weighted min-max). ``sparse_score``
    and ``dense_score`` carry the raw per-component values; either may be
    ``None`` if that component did not surface this nugget (or was disabled).
    ``component_ranks`` maps component name to its 1-indexed rank for
    downstream explainability.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nugget: Nugget
    score: float
    rank: int
    sparse_score: float | None = None
    dense_score: float | None = None
    component_ranks: dict[str, int] = Field(default_factory=dict)


class Retriever:
    """Query-time retrieval pipeline.

    Usually accessed via ``NuggetStore.aretrieve``. Created with a metadata
    backend (providing ``afilter`` / ``abm25_search`` / ``aget``) and an
    optional dense backend. When ``dense_backend`` is ``None`` the pipeline
    degrades to sparse-only, which still covers roughly 92% of hybrid recall
    per the spec.
    """

    def __init__(
        self,
        backend: SQLiteBackend,
        dense_backend: Any | None = None,
    ) -> None:
        self.backend = backend
        self.dense_backend = dense_backend

    async def aretrieve(
        self,
        query: str,
        *,
        query_time: datetime | None = None,
        view: str = "active",
        top_k: int = 20,
        fusion: FusionMode = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        qt = query_time or datetime.now(UTC)

        # 1. View filter: validity + lifecycle status.
        candidate_ids = await self.backend.afilter(
            query_time=qt,
            view=view,
            extra_filters=filters,  # type: ignore[arg-type]
        )
        if not candidate_ids:
            return []

        # 2. Sparse (BM25) within candidate set. Over-fetch by 3x for
        #    fusion-headroom so RRF has enough to work with.
        sparse_hits = await self.backend.abm25_search(
            query,
            candidate_ids=candidate_ids,
            top_k=top_k * 3,
        )

        # 3. Dense (optional).
        dense_hits: list[tuple[str, float]] = []
        if self.dense_backend is not None:
            dense_hits = await self.dense_backend.asearch(
                query,
                candidate_ids=candidate_ids,
                top_k=top_k * 3,
            )

        # 4. Fusion.
        if fusion == "rrf":
            rankings = [r for r in (sparse_hits, dense_hits) if r]
            fused = reciprocal_rank_fusion(rankings, k=60)
        else:
            fused = weighted_minmax_fusion(sparse_hits, dense_hits)

        # 5. Assemble results with hydrated nuggets.
        sparse_map = {d: (s, r) for r, (d, s) in enumerate(sparse_hits, start=1)}
        dense_map = {d: (s, r) for r, (d, s) in enumerate(dense_hits, start=1)}

        results: list[RetrievalResult] = []
        for rank, (nid, fscore) in enumerate(fused[:top_k], start=1):
            n = await self.backend.aget(nid)
            if n is None:
                # Candidate vanished between filter and hydrate (unlikely but
                # possible under concurrent writes) - skip without failing.
                continue
            s_entry = sparse_map.get(nid)
            d_entry = dense_map.get(nid)
            component_ranks: dict[str, int] = {}
            if s_entry is not None:
                component_ranks["sparse"] = s_entry[1]
            if d_entry is not None:
                component_ranks["dense"] = d_entry[1]
            results.append(
                RetrievalResult(
                    nugget=n,
                    score=fscore,
                    rank=rank,
                    sparse_score=s_entry[0] if s_entry is not None else None,
                    dense_score=d_entry[0] if d_entry is not None else None,
                    component_ranks=component_ranks,
                )
            )
        return results
