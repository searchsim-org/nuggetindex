"""Haystack 2.x ``@component`` wrapping :class:`~nuggetindex.sidecar.Sidecar`.

Sits in a Haystack ``Pipeline`` after a retriever; outputs the retriever's
original documents with a synthetic governance :class:`Document` prepended
whose ``content`` is the formatted context block the LLM can consume
directly.

Usage::

    sidecar_core = Sidecar(store=my_store, mode="offline-curated")
    pipeline.add_component("sidecar", NuggetSidecarComponent(sidecar=sidecar_core))
    pipeline.connect("retriever.documents", "sidecar.documents")
    pipeline.connect("sidecar.documents", "prompt_builder.documents")

Imports follow the two-layer pattern used by the sibling ``retriever.py``:
``TYPE_CHECKING`` gives mypy the real :class:`Document`, and the
module-level :func:`_require_haystack` call raises a useful ``pip install``
hint when the ``[haystack]`` extra isn't installed.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from nuggetindex.sidecar import Sidecar as _Sidecar

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

# Runtime binding under the same name the ``run`` signature references —
# Haystack's ``@component`` decorator calls ``typing.get_type_hints`` on
# ``run`` at decoration time and therefore needs a real binding here.
HaystackDocument = _HaystackDocument  # type: ignore[assignment,misc]


@_component
class NuggetSidecarComponent:
    """Haystack component wrapping a :class:`~nuggetindex.sidecar.Sidecar`.

    ``run(query, documents, query_time)`` returns the same documents list
    with a synthetic ``id="nuggetindex-governance"`` document prepended
    when the sidecar's router opts in. When the router passes through
    (e.g. noise query), the returned list is the input list unchanged.
    """

    def __init__(self, sidecar: _Sidecar) -> None:
        self._sidecar = sidecar

    @_component.output_types(documents=list[HaystackDocument])  # type: ignore[untyped-decorator]
    def run(
        self,
        query: str,
        documents: list[HaystackDocument],
        query_time: datetime | None = None,
    ) -> dict[str, list[HaystackDocument]]:
        response = self._sidecar.handle(
            query=query,
            query_time=query_time,
            top_k=len(documents) if documents else 10,
            original_hits=documents,
        )
        out_docs: list[HaystackDocument] = list(documents)
        if response.context_block:
            reason = response.decision.reason if response.decision is not None else ""
            gov_doc = _HaystackDocument(
                id="nuggetindex-governance",
                content=response.context_block,
                meta={"source": "nuggetindex-sidecar", "decision": reason},
            )
            out_docs.insert(0, gov_doc)
        return {"documents": out_docs}
