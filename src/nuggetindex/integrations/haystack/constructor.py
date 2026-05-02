"""Haystack 2.x integration: ``@component NuggetConstructor``.

``NuggetConstructor`` is a Haystack 2.x ``@component`` that ingests a list
of Haystack ``Document`` objects into a :class:`~nuggetindex.NuggetStore`.
Each input document's ``content`` is handed to
:meth:`~nuggetindex.NuggetStore.aingest`; the component then returns the
same document list unchanged so it composes cleanly in a pipeline
(upstream loader  NuggetConstructor  something that consumes the docs).

Name collision note: Haystack's own ``Document`` class collides with the
internal ``nuggetindex.pipeline.constructor.Document`` dataclass. The
integration layer is explicitly forbidden from reaching into
``nuggetindex.pipeline.constructor`` (see the import-hygiene test), so we
define a small local dataclass with the shape that ``NuggetStore.aingest``
duck-types on (``source_id``, ``text``, ``uri``, ``source_date``) and use
``HaystackDocument`` for the Haystack type to make the aliasing explicit.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


@_component
class NuggetConstructor:
    """Haystack 2.x constructor component that ingests docs into a ``NuggetStore``.

    Run shape:
        input:  ``documents: list[haystack.Document]``
        output: ``{"documents": list[haystack.Document]}`` (unchanged)

    The ingest side-effect happens inside ``store.aingest`` which handles
    extraction, deduplication, conflict detection, and persistence. We
    return the input documents unchanged so the component composes in
    pipelines (e.g., loader  NuggetConstructor  writer).
    """

    def __init__(
        self,
        store: NuggetStore,
        *,
        source_date: datetime | None = None,
    ) -> None:
        self.store = store
        self.source_date = source_date

    @_component.output_types(documents=list[HaystackDocument])  # type: ignore[untyped-decorator]
    def run(
        self,
        documents: list[HaystackDocument],
    ) -> dict[str, list[HaystackDocument]]:
        asyncio.run(self._aingest_all(documents))
        return {"documents": documents}

    # --- internals --------------------------------------------------------

    async def _aingest_all(self, documents: list[HaystackDocument]) -> None:
        for doc in documents:
            content = doc.content or ""
            if not content.strip():
                continue
            ingest_doc = _IngestDoc(
                source_id=doc.id,
                text=content,
                source_date=self.source_date,
            )
            await self.store.aingest(ingest_doc)
