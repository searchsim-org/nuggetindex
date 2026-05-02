"""LlamaIndex integration: ``doctor()`` convenience shim.

Thin adapter on top of :func:`nuggetindex.scan_index` that accepts either
a LlamaIndex ``VectorStoreIndex`` (anything exposing a ``.docstore``) or a
concrete iterable of ``BaseNode``/``Document`` objects and translates them
into the duck-typed shape :func:`scan_index` consumes (``source_id`` /
``text`` / ``uri`` / ``source_date``). Callers therefore don't have to
write a jsonl or open a SQLite by hand just to get a :class:`DoctorReport`
over their existing LlamaIndex corpus.

Imports follow the same two-layer pattern as the other submodules in this
package: ``TYPE_CHECKING`` gives mypy the real LlamaIndex types, and the
module-level ``_require_llamaindex()`` call is the runtime guard so
callers missing the ``[llamaindex]`` extra see a useful ``pip install``
hint rather than a bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex import DoctorReport, scan_index

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode


def _require_llamaindex() -> Any:
    try:
        from llama_index.core.schema import BaseNode as _BaseNode
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[llamaindex] not installed. "
            "Run: pip install 'nuggetindex[llamaindex]'"
        ) from e
    return _BaseNode


_BaseNode = _require_llamaindex()


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
    """Best-effort coerce a LlamaIndex metadata value into a ``datetime``.

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


def _node_to_doctor_doc(node: BaseNode) -> _DoctorDoc:
    # ``node_id`` is the canonical public accessor; ``id_`` is the raw
    # attribute used on older LlamaIndex releases. Prefer the public one.
    node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or ""
    # ``get_content()`` takes an optional MetadataMode; the default
    # renders the text content without the metadata block, which is what
    # we want for heuristic scanning.
    text = node.get_content() if hasattr(node, "get_content") else ""
    metadata: dict[str, Any] = dict(getattr(node, "metadata", {}) or {})
    uri: str | None = metadata.get("file_path") or metadata.get("url")
    return _DoctorDoc(
        source_id=str(node_id),
        text=text,
        uri=uri,
        source_date=_coerce_source_date(metadata.get("source_date")),
    )


def _iter_nodes(source: Any) -> Iterable[BaseNode]:
    """Normalise ``source`` into an iterable of ``BaseNode``.

    Three shapes are accepted:

    * A ``VectorStoreIndex``-like object with a ``.docstore`` — we iterate
      ``source.docstore.docs.values()`` (the canonical in-memory docstore
      exposes a ``.docs`` dict).
    * An object with a ``.docstore.get_all_ref_doc_info()`` fallback (used
      by some persisted stores) — we prefer ``.docs`` but fall back to
      ``.get_nodes()`` if it exists.
    * Any plain iterable of ``BaseNode`` / ``Document``.
    """
    docstore = getattr(source, "docstore", None)
    if docstore is not None:
        docs = getattr(docstore, "docs", None)
        if isinstance(docs, dict):
            return list(docs.values())
        # Fallback: some persistent docstores expose ``get_nodes`` but not ``.docs``.
        get_nodes = getattr(docstore, "get_nodes", None)
        if callable(get_nodes):
            try:
                return list(get_nodes(node_ids=None))
            except TypeError:
                # Some impls require node_ids; in that case we can't
                # enumerate without help. Fall through to the generic
                # iterable path below.
                pass
    # Generic iterable path.
    return list(source)


async def doctor(
    source: Any,
    *,
    mode: Literal["fast", "deep"] = "fast",
    sample_size: int = 500,
    stratify_by: Literal["source_date", "none"] = "source_date",
    extractor: Any | None = None,
    rng_seed: int = 0,
) -> DoctorReport:
    """Run :func:`nuggetindex.scan_index` over a LlamaIndex corpus.

    Parameters
    ----------
    source:
        Either a LlamaIndex ``VectorStoreIndex`` (anything exposing a
        ``.docstore`` with a ``.docs`` dict) or a concrete iterable of
        ``BaseNode`` / ``Document`` / ``TextNode`` objects.
    mode, sample_size, stratify_by, extractor, rng_seed:
        Forwarded verbatim to :func:`scan_index`. ``extractor`` is required
        when ``mode="deep"``.

    Returns
    -------
    DoctorReport
        Whatever :func:`scan_index` produces for the translated corpus.
    """
    nodes = _iter_nodes(source)
    docs = [_node_to_doctor_doc(n) for n in nodes]
    return await scan_index(
        docs=docs,
        mode=mode,
        sample_size=sample_size,
        stratify_by=stratify_by,
        extractor=extractor,
        rng_seed=rng_seed,
    )
