"""LangChain integration: ``NuggetConstructionLoader``.

A lightweight adapter that wraps any LangChain ``BaseLoader`` and yields
the same documents with a ``nuggetindex_ingested=True`` metadata flag so
downstream code can see which docs have already been pushed through the
nuggetindex ingest path.

For v0.1 this is intentionally minimal: the heavy lifting (extraction,
dedup, conflict detection) stays behind ``NuggetStore.aingest``. Users wire
the loader to a store themselves; the loader just marks docs.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document


def _require_langchain() -> Any:
    try:
        from langchain_core.documents import Document as _Document
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[langchain] not installed. "
            "Run: pip install 'nuggetindex[langchain]'"
        ) from e
    return _Document


_Document = _require_langchain()


class NuggetConstructionLoader:
    """Wraps any LangChain ``BaseLoader`` and stamps a nuggetindex flag.

    Provides both synchronous (``lazy_load``) and asynchronous
    (``alazy_load``) iteration. The underlying loader's own ``lazy_load`` is
    iterated as-is; if it only implements the legacy ``load()``, we fall
    back to that.
    """

    def __init__(self, base_loader: Any) -> None:
        self.base_loader = base_loader

    # --- iteration --------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        if hasattr(self.base_loader, "lazy_load"):
            iterator = self.base_loader.lazy_load()
        else:
            iterator = iter(self.base_loader.load())
        for doc in iterator:
            yield self._mark(doc)

    async def alazy_load(self) -> AsyncIterator[Document]:
        # Prefer native async if the wrapped loader provides it.
        if hasattr(self.base_loader, "alazy_load"):
            async for doc in self.base_loader.alazy_load():
                yield self._mark(doc)
            return
        # Fall back to sync iteration.
        for doc in self.lazy_load():
            yield doc

    def load(self) -> list[Document]:
        return list(self.lazy_load())

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _mark(doc: Any) -> Document:
        """Return a Document with a ``nuggetindex_ingested=True`` flag.

        Creates a new Document rather than mutating in place so the caller's
        own references stay clean.
        """
        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata.setdefault("nuggetindex_ingested", True)
        out: Document = _Document(page_content=doc.page_content, metadata=metadata)
        return out
