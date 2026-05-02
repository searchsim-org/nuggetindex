"""Haystack 2.x integration: ``doctor()`` convenience shim.

Thin adapter on top of :func:`nuggetindex.scan_index` that accepts a
Haystack ``DocumentStore`` (anything exposing ``.filter_documents()``) and
maps the stored Haystack ``Document`` objects into the duck-typed shape
:func:`scan_index` consumes (``source_id`` / ``text`` / ``uri`` /
``source_date``). Callers therefore don't have to write a jsonl or open a
SQLite by hand just to get a :class:`DoctorReport` over their existing
Haystack index.

Imports follow the same two-layer pattern as the other submodules in this
package: ``TYPE_CHECKING`` gives mypy the real Haystack types, and the
module-level ``_require_haystack()`` call is the runtime guard so callers
missing the ``[haystack]`` extra see a useful ``pip install`` hint rather
than a bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex import DoctorReport, scan_index

if TYPE_CHECKING:
    from haystack import Document as HaystackDocument
    from haystack.document_stores.types import DocumentStore


def _require_haystack() -> Any:
    try:
        from haystack import Document as _Document
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[haystack] not installed. "
            "Run: pip install 'nuggetindex[haystack]'"
        ) from e
    return _Document


_HaystackDocument = _require_haystack()


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
    """Best-effort coerce a Haystack meta value into a ``datetime``.

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


def _haystack_to_doctor_doc(doc: HaystackDocument) -> _DoctorDoc:
    meta: dict[str, Any] = dict(getattr(doc, "meta", {}) or {})
    uri: str | None = meta.get("url") or meta.get("uri")
    return _DoctorDoc(
        source_id=doc.id,
        text=doc.content or "",
        uri=uri,
        source_date=_coerce_source_date(meta.get("source_date")),
    )


async def doctor(
    document_store: DocumentStore,
    *,
    mode: Literal["fast", "deep"] = "fast",
    sample_size: int = 500,
    stratify_by: Literal["source_date", "none"] = "source_date",
    extractor: Any | None = None,
    rng_seed: int = 0,
) -> DoctorReport:
    """Run :func:`nuggetindex.scan_index` over a Haystack ``DocumentStore``.

    Parameters
    ----------
    document_store:
        Any Haystack-style store exposing ``filter_documents()`` (the
        canonical iteration method in Haystack 2.x). Each stored document's
        ``.id`` / ``.content`` / ``.meta`` is translated into the shape
        :func:`scan_index` consumes.
    mode, sample_size, stratify_by, extractor, rng_seed:
        Forwarded verbatim to :func:`scan_index`. ``extractor`` is required
        when ``mode="deep"``.

    Returns
    -------
    DoctorReport
        Whatever :func:`scan_index` produces for the translated corpus.
    """
    haystack_docs = document_store.filter_documents()
    docs = [_haystack_to_doctor_doc(d) for d in haystack_docs]
    return await scan_index(
        docs=docs,
        mode=mode,
        sample_size=sample_size,
        stratify_by=stratify_by,
        extractor=extractor,
        rng_seed=rng_seed,
    )
