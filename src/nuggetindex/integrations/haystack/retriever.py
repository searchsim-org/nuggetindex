"""Haystack 2.x integration: ``@component NuggetIndexRetriever``.

``NuggetIndexRetriever`` is a Haystack 2.x ``@component`` that maps a query
string (and optional ``query_time``) to a ``list[Document]`` via the
underlying :class:`~nuggetindex.NuggetStore`. It drops into a
``haystack.Pipeline`` in the usual retriever slot so downstream components
(a ``PromptBuilder``, a generator, etc.) can consume the returned
``Document`` list directly.

Imports follow the same two-layer pattern as the other framework
integrations: ``TYPE_CHECKING`` gives mypy the real Haystack classes, and
the module-level ``_require_haystack()`` call is the runtime guard so
callers missing the ``[haystack]`` extra see a useful ``pip install`` hint
rather than a bare ``ModuleNotFoundError``.

Haystack components are inherently sync (``run()``), so ``run`` wraps the
async :meth:`~nuggetindex.NuggetStore.aretrieve` call in ``asyncio.run``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nuggetindex import NuggetStore

if TYPE_CHECKING:
    # mypy-only: use the real class so annotations like
    # ``list[HaystackDocument]`` type-check properly.
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
# calls ``typing.get_type_hints`` on the ``run`` method at decoration time,
# which evaluates string annotations in this module's globals. ``HaystackDocument``
# therefore has to be a real binding at import time (not just a
# ``TYPE_CHECKING`` hint) so annotations like ``list[HaystackDocument]``
# resolve. The ``TYPE_CHECKING`` import above wins at type-check time; the
# runtime assignment below wins at import time.
HaystackDocument = _HaystackDocument  # type: ignore[assignment,misc]


@_component
class NuggetIndexRetriever:
    """Haystack 2.x retriever component over a ``NuggetStore``.

    Each returned ``Document`` has ``content`` set to the nugget's fact
    sentence (prefixed with ``[DISPUTED]`` when the nugget is CONTESTED and
    ``flag_contested=True``), ``meta`` carrying the governance fields
    downstream components may want to key off of, and ``score`` set to the
    fused retrieval score.
    """

    def __init__(
        self,
        store: NuggetStore,
        *,
        top_k: int = 20,
        view: str = "active",
        fusion: str = "rrf",
        flag_contested: bool = True,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.view = view
        self.fusion = fusion
        self.flag_contested = flag_contested

    @_component.output_types(documents=list[HaystackDocument])  # type: ignore[untyped-decorator]
    def run(
        self,
        query: str,
        query_time: datetime | None = None,
    ) -> dict[str, list[HaystackDocument]]:
        results = asyncio.run(
            self.store.aretrieve(
                query,
                query_time=query_time,
                view=self.view,
                top_k=self.top_k,
                fusion=self.fusion,
            )
        )
        documents: list[HaystackDocument] = [self._result_to_document(r) for r in results]
        return {"documents": documents}

    # --- internals --------------------------------------------------------

    def _result_to_document(self, result: Any) -> HaystackDocument:
        n = result.nugget
        content = n.fact.text
        status = str(n.epistemic.status)
        if self.flag_contested and status == "contested":
            content = f"[DISPUTED] {content}"

        first_prov = n.provenance[0] if n.provenance else None
        meta: dict[str, Any] = {
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
            "sparse_score": result.sparse_score,
            "dense_score": result.dense_score,
        }
        doc: HaystackDocument = _HaystackDocument(
            content=content,
            meta=meta,
            score=result.score,
        )
        return doc
