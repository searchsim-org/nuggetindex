"""Haystack 2.x integration: ``@component NuggetChainRetriever``.

``NuggetChainRetriever`` is a Haystack ``@component`` that maps a
``chain_spec`` dict (with ``"type"`` in ``{"succession", "rename", "joined"}``
plus the chain-method kwargs) to a ``list[Document]`` plus an aggregate
``chain_metadata`` payload. Runs the underlying async
:meth:`NuggetStore.achain_*` call through ``asyncio.run`` since Haystack
components are inherently sync.

Imports follow the same two-layer pattern as the sibling
``retriever.py``: ``TYPE_CHECKING`` gives mypy the real classes, and a
runtime import guard raises a useful ``pip install`` hint when the
``[haystack]`` extra is missing.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from nuggetindex import NuggetChain, NuggetStore

if TYPE_CHECKING:
    from haystack import Document as HaystackDocument


def _require_haystack() -> tuple[Any, type]:
    try:
        from haystack import Document as _Document
        from haystack import component as _component
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[haystack] not installed. "
            "Run: pip install 'nuggetindex[haystack]'"
        ) from e
    return _component, _Document


_component, _HaystackDocument = _require_haystack()

# Runtime binding under the same name: Haystack's ``@component`` decorator
# calls ``typing.get_type_hints`` on ``run`` at decoration time, which
# evaluates string annotations (``from __future__ import annotations``) in
# this module's globals. ``HaystackDocument`` therefore has to be a real
# binding at import time (not just a ``TYPE_CHECKING`` hint).
HaystackDocument = _HaystackDocument  # type: ignore[assignment,misc]


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


@_component
class NuggetChainRetriever:
    """Haystack retriever component over the three :class:`NuggetStore` chain APIs.

    ``run(chain_spec=...)`` returns ``{"documents": list[Document],
    "chain_metadata": dict}``. Each ``Document`` carries the governance
    fields the regular retriever emits plus four chain-specific extras
    (``chain_position``, ``chain_type``, ``gap_seconds_to_prev``,
    ``edge_type_to_prev``); ``chain_metadata`` is a small aggregate payload
    downstream components can key off of without re-computing it.
    """

    def __init__(
        self,
        store: NuggetStore,
        *,
        flag_contested: bool = True,
    ) -> None:
        self.store = store
        self.flag_contested = flag_contested

    @_component.output_types(  # type: ignore[untyped-decorator]
        documents=list[HaystackDocument], chain_metadata=dict
    )
    def run(
        self,
        chain_spec: dict[str, Any],
    ) -> dict[str, Any]:
        chain = asyncio.run(self._dispatch(chain_spec))
        documents: list[HaystackDocument] = self._chain_to_documents(chain)
        chain_metadata: dict[str, Any] = {
            "chain_type": chain.chain_type,
            "length": len(chain.nuggets),
            "truncated": chain.truncated,
            "as_of": chain.as_of.isoformat() if chain.as_of is not None else None,
        }
        return {"documents": documents, "chain_metadata": chain_metadata}

    # --- internals --------------------------------------------------------

    async def _dispatch(self, spec: dict[str, Any]) -> NuggetChain:
        chain_type = spec.get("type")
        if chain_type == "succession":
            return await self.store.achain_succession(
                **self._extract_kwargs(spec, _SUCCESSION_KEYS)
            )
        if chain_type == "rename":
            return await self.store.achain_rename(
                **self._extract_kwargs(spec, _RENAME_KEYS)
            )
        if chain_type == "joined":
            return await self.store.achain_join(
                **self._extract_kwargs(spec, _JOIN_KEYS)
            )
        raise ValueError(f"unknown chain type: {chain_type!r}")

    @staticmethod
    def _extract_kwargs(
        spec: dict[str, Any], allowed: set[str]
    ) -> dict[str, Any]:
        return {k: v for k, v in spec.items() if k in allowed}

    def _chain_to_documents(self, chain: NuggetChain) -> list[HaystackDocument]:
        edge_by_to: dict[int, Any] = {e.to_idx: e for e in chain.edges}
        docs: list[HaystackDocument] = []
        for i, n in enumerate(chain.nuggets):
            content = n.fact.text
            status = str(n.epistemic.status)
            if self.flag_contested and status == "contested":
                content = f"[DISPUTED] {content}"

            first_prov = n.provenance[0] if n.provenance else None
            edge = edge_by_to.get(i)
            meta: dict[str, Any] = {
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
            docs.append(_HaystackDocument(content=content, meta=meta))
        return docs
