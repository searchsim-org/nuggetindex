"""LangChain integration: ``GovernanceFilter`` wrapping any base retriever.

This is the Tier-1 adoption wedge for LangChain users: take any retriever
they already have, run its output through the framework-agnostic
``nuggetindex.governance.GovernancePostProcessor``, and return governed
``Document``s (DEPRECATED filtered, CONTESTED flagged ``[DISPUTED]``).

Governance keys off ``Document.metadata["source"]`` so that the passages it
sees round-trip cleanly to the caller. When a base retriever doesn't stamp a
``source``, we synthesize a stable one from the document's position so
per-passage governance still works — but the caller won't be able to tell
which upstream doc produced a kept passage unless they were already tracking
their own ID.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from nuggetindex.governance import GovernancePostProcessor, RetrievedPassage

if TYPE_CHECKING:
    from langchain_core.documents import Document


def _require_langchain() -> tuple[Any, Any]:
    try:
        from langchain_core.documents import Document as _Document
        from langchain_core.runnables import RunnableSerializable as _Runnable
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[langchain] not installed. Run: pip install 'nuggetindex[langchain]'"
        ) from e
    return _Document, _Runnable


_Document, _RunnableSerializable = _require_langchain()


_SYNTHETIC_SOURCE_PREFIX = "__nuggetindex_synthetic__"


class GovernanceFilter(_RunnableSerializable):  # type: ignore[misc,valid-type]
    """Wrap any LangChain retriever with nuggetindex governance.

    Composes naturally: ``base_retriever | GovernanceFilter(...)`` — but the
    simpler idiom is to construct ``GovernanceFilter(base_retriever=...,
    postprocessor=...)`` and invoke it directly, since the filter needs a
    single string query (not ``list[Document]``) on its input side.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any
    postprocessor: GovernancePostProcessor

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        base_docs = await self.base_retriever.ainvoke(input, config, **kwargs)
        return await self._filter(list(base_docs))

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        import asyncio

        return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def _filter(self, base_docs: list[Any]) -> list[Document]:
        if not base_docs:
            return []

        # Build framework-agnostic passages. Use a synthetic source_id when
        # the upstream retriever didn't stamp one so governance can still
        # track per-passage state; the synthetic id carries the doc index
        # so we can map kept passages back to original docs below.
        passages: list[RetrievedPassage] = []
        for i, d in enumerate(base_docs):
            sid = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            if not sid:
                sid = f"{_SYNTHETIC_SOURCE_PREFIX}{i}"
            passages.append(RetrievedPassage(source_id=sid, text=d.page_content))

        kept = await self.postprocessor.apostprocess(passages)
        kept_by_source: dict[str, RetrievedPassage] = {p.source_id: p for p in kept}

        # Map kept source ids back to the original Document objects, applying
        # any ``[DISPUTED]`` prefix the postprocessor added. Docs whose
        # source_id is not in ``kept_by_source`` were filtered by governance
        # (e.g. all-DEPRECATED) and are dropped.
        out: list[Document] = []
        for i, d in enumerate(base_docs):
            sid = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            if not sid:
                sid = f"{_SYNTHETIC_SOURCE_PREFIX}{i}"
            if sid not in kept_by_source:
                continue
            kept_passage = kept_by_source[sid]
            if kept_passage.text != d.page_content:
                out.append(_Document(page_content=kept_passage.text, metadata=dict(d.metadata)))
            else:
                out.append(d)
        return out
