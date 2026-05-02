"""LlamaIndex integration: ``NuggetChainRetriever``.

``NuggetChainRetriever`` is a :class:`BaseRetriever` adapter over the three
chain methods on :class:`~nuggetindex.NuggetStore`. Each hop becomes a
:class:`NodeWithScore` whose ``metadata`` carries both the governance
fields the regular retriever emits and four chain-specific extras
(``chain_position``, ``chain_type``, ``gap_seconds_to_prev``,
``edge_type_to_prev``).

LlamaIndex's ``BaseRetriever`` contract expects ``_aretrieve`` to take a
:class:`QueryBundle` with a free-text ``query_str`` -- not a natural fit
for a chain spec. This adapter supports **two** entry points:

1. **Primary: passthrough methods** -- call
   :meth:`achain_succession`, :meth:`achain_rename`, or :meth:`achain_join`
   directly on the retriever. Their kwargs mirror
   :meth:`NuggetStore.achain_*` one-to-one.
2. **Fallback: JSON-in-query-str** -- ``_aretrieve(QueryBundle(query_str=spec))``
   where ``spec`` is a JSON string with a ``"type"`` discriminator and the
   same kwargs the passthrough methods accept. Useful in LlamaIndex query
   engines that only surface a ``QueryBundle`` to their retriever.

Imports follow the same two-layer guard pattern as the sibling
``retriever.py``.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nuggetindex import NuggetChain, NuggetStore

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


class NuggetChainRetriever(_BaseRetriever):  # type: ignore[misc,valid-type]
    """BaseRetriever adapter for the three :class:`NuggetStore` chain APIs."""

    def __init__(
        self,
        store: NuggetStore,
        *,
        flag_contested: bool = True,
    ) -> None:
        super().__init__()
        self._store = store
        self._flag_contested = flag_contested

    # --- primary passthrough API -----------------------------------------

    async def achain_succession(
        self,
        *,
        subject: str,
        predicate: str,
        scope: str = "global",
        as_of: datetime | None = None,
        include_contested: bool = False,
        max_depth: int = 50,
    ) -> list[NodeWithScore]:
        chain = await self._store.achain_succession(
            subject=subject,
            predicate=predicate,
            scope=scope,
            as_of=as_of,
            include_contested=include_contested,
            max_depth=max_depth,
        )
        return self._chain_to_nodes(chain)

    async def achain_rename(
        self,
        *,
        subject: str,
        as_of: datetime | None = None,
        direction: str = "forward",
        max_depth: int = 10,
        include_contested: bool = False,
        resolver: Any | None = None,
    ) -> list[NodeWithScore]:
        chain = await self._store.achain_rename(
            subject=subject,
            as_of=as_of,
            direction=direction,  # type: ignore[arg-type]
            max_depth=max_depth,
            include_contested=include_contested,
            resolver=resolver,
        )
        return self._chain_to_nodes(chain)

    async def achain_join(
        self,
        *,
        start: tuple[str, str],
        then: list[str],
        scope: str = "global",
        as_of: datetime | None = None,
        resolver: Any | None = None,
    ) -> list[NodeWithScore]:
        chain = await self._store.achain_join(
            start=start,
            then=then,
            scope=scope,
            as_of=as_of,
            resolver=resolver,
        )
        return self._chain_to_nodes(chain)

    # --- QueryBundle fallback --------------------------------------------

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Parse ``query_bundle.query_str`` as a JSON chain spec and dispatch."""
        spec = json.loads(query_bundle.query_str)
        chain_type = spec.get("type")
        if chain_type == "succession":
            return await self.achain_succession(
                **self._parse_kwargs(spec, _SUCCESSION_KEYS)
            )
        if chain_type == "rename":
            return await self.achain_rename(
                **self._parse_kwargs(spec, _RENAME_KEYS)
            )
        if chain_type == "joined":
            return await self.achain_join(
                **self._parse_kwargs(spec, _JOIN_KEYS)
            )
        raise ValueError(f"unknown chain type: {chain_type!r}")

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return asyncio.run(self._aretrieve(query_bundle))

    # --- internals --------------------------------------------------------

    @staticmethod
    def _parse_kwargs(
        spec: dict[str, Any], allowed: set[str]
    ) -> dict[str, Any]:
        """Pick + coerce JSON-decoded kwargs for a chain-method call.

        Coerces ``as_of`` (if present) from ISO-8601 string to ``datetime``,
        and ``start`` (for joins) from list to tuple.
        """
        out: dict[str, Any] = {k: v for k, v in spec.items() if k in allowed}
        if "as_of" in out and isinstance(out["as_of"], str):
            out["as_of"] = datetime.fromisoformat(out["as_of"])
        if "start" in out and isinstance(out["start"], list):
            out["start"] = tuple(out["start"])
        return out

    def _chain_to_nodes(self, chain: NuggetChain) -> list[NodeWithScore]:
        """Emit one ``NodeWithScore`` per nugget, threading edge metadata."""
        edge_by_to: dict[int, Any] = {e.to_idx: e for e in chain.edges}
        nodes: list[NodeWithScore] = []
        for i, n in enumerate(chain.nuggets):
            text = n.fact.text
            status = str(n.epistemic.status)
            if self._flag_contested and status == "contested":
                text = f"[DISPUTED] {text}"

            first_prov = n.provenance[0] if n.provenance else None
            edge = edge_by_to.get(i)
            metadata: dict[str, Any] = {
                "nugget_id": n.id,
                "subject": n.fact.subject,
                "predicate": n.fact.predicate,
                "object": n.fact.object,
                "valid_from": n.validity.start.isoformat(),
                "valid_until": (
                    n.validity.end.isoformat()
                    if n.validity.end is not None
                    else "ongoing"
                ),
                "status": status,
                "confidence": n.epistemic.confidence,
                "source": first_prov.source_id if first_prov is not None else None,
                "evidence": (
                    first_prov.evidence_span if first_prov is not None else ""
                ),
                "chain_position": i,
                "chain_type": chain.chain_type,
                "gap_seconds_to_prev": (
                    edge.gap.total_seconds()
                    if edge is not None and edge.gap is not None
                    else None
                ),
                "edge_type_to_prev": (
                    str(edge.edge_type) if edge is not None else None
                ),
            }
            text_node = _TextNode(text=text, metadata=metadata)
            nodes.append(_NodeWithScore(node=text_node, score=None))
        return nodes


_SUCCESSION_KEYS = {
    "subject",
    "predicate",
    "scope",
    "as_of",
    "include_contested",
    "max_depth",
}
_RENAME_KEYS = {
    "subject",
    "as_of",
    "direction",
    "max_depth",
    "include_contested",
    "resolver",
}
_JOIN_KEYS = {"start", "then", "scope", "as_of", "resolver"}
