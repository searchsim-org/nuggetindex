"""LangChain integration: Runnable-based temporal chain retriever.

``NuggetChainRetriever`` exposes :meth:`NuggetStore.achain_succession`,
:meth:`NuggetStore.achain_rename`, and :meth:`NuggetStore.achain_join` via
a single ``Runnable`` whose input is a ``dict`` discriminated on ``"type"``.
Each hop in the chain becomes a ``Document`` whose ``metadata`` carries
both the governance fields :class:`NuggetIndexRetriever` emits and four
chain-specific extras:

* ``chain_position`` -- 0-indexed position in the chain.
* ``chain_type`` -- ``"succession"`` / ``"rename"`` / ``"joined"``.
* ``gap_seconds_to_prev`` -- :class:`~datetime.timedelta` gap from the
  previous hop's validity, in seconds (``None`` for head / rename / join
  hops where there is no succession gap).
* ``edge_type_to_prev`` -- semantic edge label (``"succeeds"`` /
  ``"renames_to"`` / ``"object_is_subject"``) or ``None`` for the head.

Imports follow the same two-layer pattern as the sibling
``retriever.py``: ``TYPE_CHECKING`` imports give mypy the real
``Document`` / ``RunnableSerializable`` types, and the module-level
``_require_langchain()`` call is the runtime guard so callers missing the
``[langchain]`` extra see a useful ``pip install`` hint.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from nuggetindex import NuggetChain, NuggetStore

if TYPE_CHECKING:
    from langchain_core.documents import Document


def _require_langchain() -> tuple[Any, Any, Any]:
    try:
        from langchain_core.documents import Document as _Document
        from langchain_core.runnables import RunnableSerializable as _Runnable
        from langchain_core.runnables.config import RunnableConfig as _Config
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[langchain] not installed. Run: pip install 'nuggetindex[langchain]'"
        ) from e
    return _Document, _Runnable, _Config


_Document, _RunnableSerializable, _RunnableConfig = _require_langchain()


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


class NuggetChainRetriever(_RunnableSerializable):  # type: ignore[misc,valid-type]
    """Runnable adapter that maps a chain spec to ``list[Document]``.

    ``ainvoke`` accepts a ``dict`` with a ``"type"`` key (one of
    ``"succession"``, ``"rename"``, ``"joined"``) plus the chain-specific
    arguments you would otherwise pass to :meth:`NuggetStore.achain_*`
    directly. See the module docstring for the metadata shape.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: NuggetStore
    flag_contested: bool = True

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        if not isinstance(input, dict) or "type" not in input:
            raise TypeError(
                "NuggetChainRetriever expected a dict with a 'type' key; got "
                f"{type(input).__name__}"
            )
        chain_type = input["type"]
        if chain_type == "succession":
            chain = await self.store.achain_succession(
                **self._extract_kwargs(input, _SUCCESSION_KEYS)
            )
        elif chain_type == "rename":
            chain = await self.store.achain_rename(**self._extract_kwargs(input, _RENAME_KEYS))
        elif chain_type == "joined":
            chain = await self.store.achain_join(**self._extract_kwargs(input, _JOIN_KEYS))
        else:
            raise ValueError(f"unknown chain type: {chain_type!r}")
        return self._chain_to_documents(chain)

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    # --- internals --------------------------------------------------------

    @staticmethod
    def _extract_kwargs(input: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
        """Pick the chain-method kwargs out of ``input``, ignoring ``type``."""
        return {k: v for k, v in input.items() if k in allowed}

    def _chain_to_documents(self, chain: NuggetChain) -> list[Document]:
        """Emit one Document per nugget, threading chain edge metadata."""
        # edges are directed from_idx -> to_idx; build a lookup keyed by to_idx
        # so the hop that "receives" the edge knows its predecessor metadata.
        edge_by_to: dict[int, Any] = {e.to_idx: e for e in chain.edges}
        docs: list[Document] = []
        for i, n in enumerate(chain.nuggets):
            content = n.fact.text
            status = str(n.epistemic.status)
            if self.flag_contested and status == "contested":
                content = f"[DISPUTED] {content}"

            first_prov = n.provenance[0] if n.provenance else None
            edge = edge_by_to.get(i)
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
                "evidence": (first_prov.evidence_span if first_prov is not None else None),
                "chain_position": i,
                "chain_type": chain.chain_type,
                "gap_seconds_to_prev": (
                    edge.gap.total_seconds() if edge is not None and edge.gap is not None else None
                ),
                "edge_type_to_prev": (str(edge.edge_type) if edge is not None else None),
            }
            docs.append(_Document(page_content=content, metadata=metadata))
        return docs
