"""LlamaIndex integration: ``NuggetIndexRetriever``.

``NuggetIndexRetriever`` is a ``BaseRetriever`` subclass that maps a
LlamaIndex ``QueryBundle`` to a ``list[NodeWithScore]`` via the underlying
``NuggetStore``.

Imports follow the same two-layer pattern as the dense-backend guards:
``TYPE_CHECKING`` gives mypy the real classes, and the module-level
``_require_llamaindex()`` call is the runtime guard so callers missing the
``[llamaindex]`` extra see a useful ``pip install`` hint rather than a bare
``ModuleNotFoundError``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nuggetindex import NuggetStore

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore, QueryBundle


def _require_llamaindex() -> tuple[Any, Any, Any, Any]:
    try:
        from llama_index.core.retrievers import BaseRetriever as _BaseRetriever
        from llama_index.core.schema import (
            NodeWithScore as _NodeWithScore,
        )
        from llama_index.core.schema import (
            QueryBundle as _QueryBundle,
        )
        from llama_index.core.schema import (
            TextNode as _TextNode,
        )
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[llamaindex] not installed. "
            "Run: pip install 'nuggetindex[llamaindex]'"
        ) from e
    return _BaseRetriever, _NodeWithScore, _QueryBundle, _TextNode


_BaseRetriever, _NodeWithScore, _QueryBundle, _TextNode = _require_llamaindex()


class NuggetIndexRetriever(_BaseRetriever):  # type: ignore[misc,valid-type]
    """BaseRetriever adapter over ``NuggetStore``.

    Each returned ``NodeWithScore`` wraps a ``TextNode`` whose ``text`` is
    the nugget's fact sentence (prefixed with ``[DISPUTED]`` when the
    nugget is CONTESTED and ``flag_contested=True``), and whose
    ``metadata`` carries the key governance fields so downstream
    postprocessors can key off of them without another backend round-trip.
    """

    def __init__(
        self,
        store: NuggetStore,
        *,
        query_time: datetime | None = None,
        view: str = "active",
        top_k: int = 20,
        fusion: str = "rrf",
        flag_contested: bool = True,
    ) -> None:
        super().__init__()
        self._store = store
        self._query_time = query_time
        self._view = view
        self._top_k = top_k
        self._fusion = fusion
        self._flag_contested = flag_contested

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = await self._store.aretrieve(
            query_bundle.query_str,
            query_time=self._query_time,
            view=self._view,
            top_k=self._top_k,
            fusion=self._fusion,
        )
        return [self._to_node(r) for r in results]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return asyncio.run(self._aretrieve(query_bundle))

    # --- internals --------------------------------------------------------

    def _to_node(self, result: Any) -> NodeWithScore:
        n = result.nugget
        text = n.fact.text
        status = str(n.epistemic.status)
        if self._flag_contested and status == "contested":
            text = f"[DISPUTED] {text}"

        first_prov = n.provenance[0] if n.provenance else None
        metadata: dict[str, Any] = {
            "nugget_id": n.id,
            "subject": n.fact.subject,
            "predicate": n.fact.predicate,
            "object": n.fact.object,
            "valid_from": n.validity.start.isoformat(),
            "valid_until": (
                n.validity.end.isoformat() if n.validity.end is not None else "ongoing"
            ),
            "status": status,
            "confidence": n.epistemic.confidence,
            "source": first_prov.source_id if first_prov is not None else None,
            "evidence": first_prov.evidence_span if first_prov is not None else "",
            "retrieval_score": result.score,
            "sparse_score": result.sparse_score,
            "dense_score": result.dense_score,
        }
        node = _TextNode(text=text, metadata=metadata)
        out: NodeWithScore = _NodeWithScore(node=node, score=result.score)
        return out
