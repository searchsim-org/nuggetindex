"""LangChain integration: Runnable-based retriever.

``NuggetIndexRetriever`` is a ``RunnableSerializable`` over ``NuggetStore`` so
it composes naturally into a LangChain chain via ``|``. We deliberately use
the modern ``RunnableSerializable`` surface rather than the legacy
``BaseRetriever`` — see spec §8.1 / Improvement #4.

Imports are split into two layers: ``TYPE_CHECKING`` imports give mypy the
real ``Document`` / ``RunnableSerializable`` types, while the module-level
``_require_langchain()`` call is the runtime import guard — if the
``[langchain]`` extra isn't installed the caller sees a useful
``pip install nuggetindex[langchain]`` hint instead of a bare
``ModuleNotFoundError`` (same pattern as the dense backends in Phase 6).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from nuggetindex import NuggetStore

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


class NuggetIndexRetriever(_RunnableSerializable):  # type: ignore[misc,valid-type]
    """Runnable adapter that maps a query string to ``list[Document]``.

    Fields are pydantic-validated. ``store`` is an arbitrary
    ``NuggetStore`` — we set ``arbitrary_types_allowed=True`` so pydantic
    doesn't try to introspect its (non-pydantic) internals.

    ``ainvoke`` accepts either a raw ``str`` query or a ``dict`` with a
    ``"query"`` key so the retriever drops into prompt templates that
    forward ``{"query": ...}`` or ``{"question": ...}`` payloads.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: NuggetStore
    query_time: datetime | None = None
    view: str = "active"
    top_k: int = 20
    fusion: str = "rrf"
    flag_contested: bool = True

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        query = self._extract_query(input)
        results = await self.store.aretrieve(
            query,
            query_time=self.query_time,
            view=self.view,
            top_k=self.top_k,
            fusion=self.fusion,
        )
        return [self._result_to_document(r) for r in results]

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    # --- internals --------------------------------------------------------

    @staticmethod
    def _extract_query(input: Any) -> str:
        """Accept a plain string, or a dict with ``query`` / ``question``."""
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            for key in ("query", "question", "input"):
                if key in input and isinstance(input[key], str):
                    value: str = input[key]
                    return value
        raise TypeError(
            f"NuggetIndexRetriever expected str or dict with 'query', got {type(input).__name__}"
        )

    def _result_to_document(self, result: Any) -> Document:
        n = result.nugget
        content = n.fact.text
        status = str(n.epistemic.status)
        if self.flag_contested and status == "contested":
            content = f"[DISPUTED] {content}"

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
            "evidence": first_prov.evidence_span if first_prov is not None else None,
            "retrieval_score": result.score,
            "sparse_score": result.sparse_score,
            "dense_score": result.dense_score,
        }
        doc: Document = _Document(page_content=content, metadata=metadata)
        return doc
