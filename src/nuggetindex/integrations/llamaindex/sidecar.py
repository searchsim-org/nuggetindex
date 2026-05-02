"""LlamaIndex ``NodePostprocessor`` wrapping :class:`~nuggetindex.sidecar.Sidecar`.

Inserted after the retriever, before response synthesis::

    query_engine = index.as_query_engine(
        node_postprocessors=[NuggetSidecarNodePostprocessor(sidecar=my_sidecar)],
    )

When the sidecar router opts in, the postprocessor prepends a synthetic
``TextNode`` (``id_="nuggetindex-governance"``) whose text is the formatted
context block. Otherwise the input node list is returned unchanged.

Imports follow the two-layer guard used by the sibling ``retriever.py``:
``TYPE_CHECKING`` gives mypy the real classes, and
:func:`_require_llamaindex` raises a useful ``pip install`` hint when the
``[llamaindex]`` extra isn't installed.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from nuggetindex.sidecar import Sidecar as _Sidecar

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore, QueryBundle


def _require_llamaindex() -> tuple[Any, Any, Any, Any]:
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
        from llama_index.core.schema import (
            TextNode as _TextNode,
        )
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[llamaindex] not installed. "
            "Run: pip install 'nuggetindex[llamaindex]'"
        ) from e
    return _BNP, _NodeWithScore, _QueryBundle, _TextNode


_BaseNodePostprocessor, _NodeWithScore, _QueryBundle, _TextNode = _require_llamaindex()


class NuggetSidecarNodePostprocessor(_BaseNodePostprocessor):  # type: ignore[misc,valid-type]
    """LlamaIndex ``NodePostprocessor`` that prepends a governance node.

    Construction is cheap — the heavy work runs inside
    :meth:`_postprocess_nodes`. The :class:`~nuggetindex.sidecar.Sidecar`
    is stashed on a private attribute via ``object.__setattr__`` so
    pydantic (``BaseNodePostprocessor`` inherits a pydantic base) doesn't
    try to validate it, matching the idiom used by the sibling
    :class:`GovernancePostProcessor`.
    """

    def __init__(self, *, sidecar: _Sidecar) -> None:
        super().__init__()
        object.__setattr__(self, "_sidecar", sidecar)

    @classmethod
    def class_name(cls) -> str:
        return "NuggetSidecarNodePostprocessor"

    @property
    def sidecar(self) -> _Sidecar:
        sc: _Sidecar = object.__getattribute__(self, "_sidecar")
        return sc

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
        query_time: datetime | None = None,
    ) -> list[NodeWithScore]:
        query = query_bundle.query_str if query_bundle is not None else ""
        response = self.sidecar.handle(
            query=query,
            query_time=query_time,
            top_k=len(nodes) if nodes else 10,
            original_hits=[n.node for n in nodes],
        )
        out: list[NodeWithScore] = list(nodes)
        if response.context_block:
            gov_node = _TextNode(
                id_="nuggetindex-governance",
                text=response.context_block,
                metadata={"source": "nuggetindex-sidecar"},
            )
            out.insert(0, _NodeWithScore(node=gov_node, score=1.0))
        return out
