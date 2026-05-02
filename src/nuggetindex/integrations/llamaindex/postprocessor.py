"""LlamaIndex integration: ``GovernancePostProcessor`` — Tier-1 adoption wedge.

Thin adapter over the framework-agnostic
``nuggetindex.governance.GovernancePostProcessor``. Any LlamaIndex
``QueryEngine`` can accept one of these via
``node_postprocessors=[GovernancePostProcessor()]`` and inherit DEPRECATED
filtering + ``[DISPUTED]`` flagging on its retrieved nodes.

The integration class here shadows the core class name. To keep things
readable we alias the import to ``_CoreGPP`` so the wrapping class can keep
the unqualified name consumers expect from LlamaIndex postprocessors.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nuggetindex.governance import (
    GovernancePostProcessor as _CoreGPP,
)
from nuggetindex.governance import (
    RetrievedPassage,
)

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore, QueryBundle


def _require_llamaindex() -> tuple[Any, Any, Any]:
    try:
        from llama_index.core.postprocessor.types import (
            BaseNodePostprocessor as _BNP,
        )
        from llama_index.core.schema import (
            NodeWithScore as _NodeWithScore,
        )
        from llama_index.core.schema import (
            QueryBundle as _QueryBundle,
        )
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[llamaindex] not installed. Run: pip install 'nuggetindex[llamaindex]'"
        ) from e
    return _BNP, _NodeWithScore, _QueryBundle


_BaseNodePostprocessor, _NodeWithScore, _QueryBundle = _require_llamaindex()


class GovernancePostProcessor(_BaseNodePostprocessor):  # type: ignore[misc,valid-type]
    """LlamaIndex adapter over the shared governance core (Improvement A).

    Two-line integration with any existing LlamaIndex query engine::

        index.as_query_engine(
            node_postprocessors=[GovernancePostProcessor()],
        )

    Construction is cheap and loop-safe — the heavy work (extraction,
    ingestion) happens lazily inside :meth:`_apostprocess_nodes`.
    """

    def __init__(
        self,
        *,
        cache_path: Path | str | None = None,
        extractor: Any = "gpt-4o-mini",
        query_time: datetime | None = None,
        filter_deprecated: bool = True,
        flag_contested: bool = True,
    ) -> None:
        super().__init__()
        # Store on a private attribute that pydantic (BaseNodePostprocessor
        # extends a pydantic BaseComponent) won't try to validate.
        object.__setattr__(
            self,
            "_core",
            _CoreGPP(
                cache_path=cache_path,
                extractor=extractor,
                query_time=query_time,
                filter_deprecated=filter_deprecated,
                flag_contested=flag_contested,
            ),
        )

    @classmethod
    def class_name(cls) -> str:
        return "NuggetIndexGovernancePostProcessor"

    async def _apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        passages = [
            RetrievedPassage(
                source_id=n.node.node_id,
                text=n.node.get_content(),
                score=n.score if n.score is not None else 0.0,
            )
            for n in nodes
        ]
        kept = await self._core.apostprocess(passages)
        kept_by_source: dict[str, RetrievedPassage] = {p.source_id: p for p in kept}

        out: list[NodeWithScore] = []
        for n in nodes:
            sid = n.node.node_id
            if sid not in kept_by_source:
                continue
            matching = kept_by_source[sid]
            # If governance rewrote the text (e.g. added ``[DISPUTED]``),
            # stamp the new text onto the node before emitting.
            if matching.text != n.node.get_content():
                n.node.set_content(matching.text)
            out.append(n)
        return out

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        return asyncio.run(self._apostprocess_nodes(nodes, query_bundle))

    # --- access to the underlying core (handy in tests) -------------------

    @property
    def core(self) -> _CoreGPP:
        core: _CoreGPP = object.__getattribute__(self, "_core")
        return core

    async def aclose(self) -> None:
        await self.core.aclose()
