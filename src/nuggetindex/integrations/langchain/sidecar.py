"""LangChain ``RunnableSerializable`` wrapping :class:`~nuggetindex.sidecar.Sidecar`.

Composes between a retriever and a prompt template::

    from langchain_core.runnables import RunnablePassthrough
    chain = (
        {"documents": retriever, "query": RunnablePassthrough()}
        | NuggetSidecarRunnable(sidecar=my_sidecar)
        | prompt_template
        | llm
    )

``invoke({"query": ..., "documents": [...], "query_time": ...})`` returns
``{"query": query, "documents": [...], "context_block": str, "nuggets":
[...]}`` — the prompt template downstream picks whichever of those keys it
cares about.

Imports follow the two-layer guard used by the sibling ``retriever.py``:
``TYPE_CHECKING`` gives mypy the real :class:`Document` /
:class:`RunnableSerializable`, and :func:`_require_langchain` raises a
useful ``pip install`` hint when the ``[langchain]`` extra isn't installed.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from nuggetindex.sidecar import Sidecar as _Sidecar

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


def _require_langchain() -> tuple[Any, Any]:
    try:
        from langchain_core.runnables import RunnableSerializable as _Runnable
        from langchain_core.runnables.config import RunnableConfig as _Config
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[langchain] not installed. "
            "Run: pip install 'nuggetindex[langchain]'"
        ) from e
    return _Runnable, _Config


_RunnableSerializable, _RunnableConfig = _require_langchain()


class NuggetSidecarRunnable(_RunnableSerializable):  # type: ignore[misc,valid-type]
    """Runnable that augments retriever output with nuggetindex context.

    ``sidecar`` is an arbitrary :class:`~nuggetindex.sidecar.Sidecar`;
    ``arbitrary_types_allowed`` lets pydantic store it without trying to
    validate its (non-pydantic) internals, matching the idiom used by the
    sibling :class:`NuggetIndexRetriever`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sidecar: _Sidecar

    async def ainvoke(
        self,
        input: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict:
        query: str = input["query"]
        documents: list[Any] = list(input.get("documents") or [])
        query_time: datetime | None = input.get("query_time")
        response = await self.sidecar.ahandle(
            query=query,
            query_time=query_time,
            top_k=len(documents) if documents else 10,
            original_hits=documents,
        )
        return {
            "query": query,
            "documents": documents,
            "context_block": response.context_block,
            "nuggets": list(response.nuggets),
        }

    def invoke(
        self,
        input: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict:
        import asyncio

        return asyncio.run(self.ainvoke(input, config, **kwargs))
