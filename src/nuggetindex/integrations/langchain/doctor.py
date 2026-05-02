"""LangChain integration: ``doctor()`` convenience shim.

Thin adapter on top of :func:`nuggetindex.scan_index` that accepts a
concrete iterable of LangChain ``Document`` objects and translates them
into the duck-typed shape :func:`scan_index` consumes (``source_id`` /
``text`` / ``uri`` / ``source_date``). Callers therefore don't have to
write a jsonl or open a SQLite by hand just to get a :class:`DoctorReport`
over their existing LangChain corpus.

Scope: this shim intentionally does NOT accept ``BaseRetriever`` or
``VectorStore`` — those surfaces either require a live query or vary too
much across backends for a one-size-fits-all scan. If you have a
``VectorStore``, call ``list(store.docstore._dict.values())`` (or the
equivalent for your concrete store) and pass the resulting list in.

Imports follow the same two-layer pattern as the other submodules in this
package: ``TYPE_CHECKING`` gives mypy the real LangChain types, and the
module-level ``_require_langchain()`` call is the runtime guard so callers
missing the ``[langchain]`` extra see a useful ``pip install`` hint rather
than a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex import DoctorReport, scan_index

if TYPE_CHECKING:
    from langchain_core.documents import Document


def _require_langchain() -> Any:
    try:
        from langchain_core.documents import Document as _Document
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[langchain] not installed. Run: pip install 'nuggetindex[langchain]'"
        ) from e
    return _Document


_Document = _require_langchain()


@dataclass
class _DoctorDoc:
    """Minimal shape duck-typed by :func:`scan_index`.

    Kept local to this module so we don't reach into
    ``nuggetindex.pipeline.constructor`` (blocked by the integration
    import-hygiene test). :func:`scan_index` reads ``source_id``, ``text``,
    ``uri``, and ``source_date`` off each item — those are the only four
    fields we populate.
    """

    source_id: str
    text: str
    uri: str | None = None
    source_date: datetime | None = None


def _coerce_source_date(value: Any) -> datetime | None:
    """Best-effort coerce a LangChain metadata value into a ``datetime``.

    Accepts ``None``, an existing ``datetime`` (passed through), or an ISO
    8601 string (``datetime.fromisoformat``). Anything else returns
    ``None`` — we'd rather drop a malformed source_date than raise and
    sink the whole scan.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _stable_id_from_content(content: str) -> str:
    """Return a deterministic id for LangChain Documents that don't carry one.

    LangChain's ``Document.id`` is optional (it's a recent addition). When
    it's missing we fall back to a short blake2b digest of the
    ``page_content`` so the same content maps to the same ``source_id``
    across runs — matters for the stratified sampler's determinism.
    """
    return hashlib.blake2b(content.encode("utf-8"), digest_size=16).hexdigest()


def _langchain_to_doctor_doc(doc: Document) -> _DoctorDoc:
    content = doc.page_content or ""
    doc_id = getattr(doc, "id", None) or _stable_id_from_content(content)
    metadata: dict[str, Any] = dict(getattr(doc, "metadata", {}) or {})
    uri: str | None = metadata.get("source") or metadata.get("url")
    return _DoctorDoc(
        source_id=str(doc_id),
        text=content,
        uri=uri,
        source_date=_coerce_source_date(metadata.get("source_date")),
    )


async def doctor(
    documents: Iterable[Document],
    *,
    mode: Literal["fast", "deep"] = "fast",
    sample_size: int = 500,
    stratify_by: Literal["source_date", "none"] = "source_date",
    extractor: Any | None = None,
    rng_seed: int = 0,
) -> DoctorReport:
    """Run :func:`nuggetindex.scan_index` over LangChain ``Document`` objects.

    Parameters
    ----------
    documents:
        An iterable of LangChain ``Document`` (e.g., the output of a
        ``BaseLoader.load()`` call, or the values of a concrete in-memory
        vector store's docstore). Retrievers and ``VectorStore`` instances
        are deliberately NOT accepted — see the module docstring.
    mode, sample_size, stratify_by, extractor, rng_seed:
        Forwarded verbatim to :func:`scan_index`. ``extractor`` is required
        when ``mode="deep"``.

    Returns
    -------
    DoctorReport
        Whatever :func:`scan_index` produces for the translated corpus.

    Raises
    ------
    TypeError
        If ``documents`` is a ``BaseRetriever``-like object (has a
        ``.get_relevant_documents`` method) or a ``VectorStore``-like
        object (has ``.similarity_search`` but no ``__iter__``) — the
        shim's contract is a concrete iterable of ``Document``.
    """
    if hasattr(documents, "get_relevant_documents") or hasattr(documents, "similarity_search"):
        raise TypeError(
            "nuggetindex.integrations.langchain.doctor() accepts an "
            "Iterable[langchain_core.documents.Document], not a retriever "
            "or VectorStore. Materialise the documents first (e.g., "
            "`docs = list(store.docstore._dict.values())`) and pass the "
            "list in."
        )
    docs = [_langchain_to_doctor_doc(d) for d in documents]
    return await scan_index(
        docs=docs,
        mode=mode,
        sample_size=sample_size,
        stratify_by=stratify_by,
        extractor=extractor,
        rng_seed=rng_seed,
    )
