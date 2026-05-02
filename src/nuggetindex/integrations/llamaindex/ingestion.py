"""LlamaIndex integration: ``NuggetTransformation`` for ``IngestionPipeline``.

Drops into a ``IngestionPipeline(transformations=[...])`` composition so
users already building LlamaIndex ingestion graphs can add a nuggetindex
ingest step with a single line. For each input node, the transformation
calls ``store.aingest(...)`` and returns the nodes unchanged — the point is
side-effect ingest, not node rewriting.

The ``Document`` payload passed to ``store.aingest`` is a minimal local
dataclass rather than the internal ``nuggetindex.pipeline.constructor.Document``
so this module keeps to the public surface (see import-hygiene test).
``NuggetStore.aingest`` duck-types on ``source_id``, ``text``, ``uri``,
``source_date``.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nuggetindex import NuggetStore

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode


def _require_llamaindex() -> tuple[Any, Any]:
    try:
        from llama_index.core.schema import (
            BaseNode as _BaseNode,
        )
        from llama_index.core.schema import (
            TransformComponent as _TransformComponent,
        )
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[llamaindex] not installed. "
            "Run: pip install 'nuggetindex[llamaindex]'"
        ) from e
    return _TransformComponent, _BaseNode


_TransformComponent, _BaseNode = _require_llamaindex()


@dataclass
class _IngestDoc:
    """Minimal shape expected by ``NuggetStore.aingest``.

    Kept local to this module so we don't reach into
    ``nuggetindex.pipeline.constructor`` (which is explicitly off-limits to
    integration glue per the import-hygiene test).
    """

    source_id: str
    text: str
    uri: str | None = None
    source_date: datetime | None = None


class NuggetTransformation(_TransformComponent):  # type: ignore[misc,valid-type]
    """IngestionPipeline component that ingests nodes into a ``NuggetStore``.

    The transformation is a pure side effect from LlamaIndex's point of
    view: every input node is returned unchanged so downstream
    transformations can keep composing. The nuggetindex-side effect is that
    each node's content is extracted and persisted into the configured
    ``NuggetStore``.
    """

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        store: NuggetStore,
        *,
        source_date: datetime | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_source_date", source_date)

    @classmethod
    def class_name(cls) -> str:
        return "NuggetTransformation"

    # --- LlamaIndex transformation API ------------------------------------

    async def acall(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        store: NuggetStore = object.__getattribute__(self, "_store")
        source_date: datetime | None = object.__getattribute__(self, "_source_date")
        for node in nodes:
            text = node.get_content()
            if not text.strip():
                continue
            doc = _IngestDoc(
                source_id=node.node_id,
                text=text,
                source_date=source_date,
            )
            await store.aingest(doc)
        return nodes

    def __call__(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        return asyncio.run(self.acall(nodes, **kwargs))
