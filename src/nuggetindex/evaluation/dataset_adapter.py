"""Dataset adapter that enriches a Hugging Face ``Dataset`` with nugget metadata.

The adapter fills the ``retrieved_nuggets`` / ``contested_keys`` /
``temporal_valid_count`` columns that the Ragas metrics in
``nuggetindex.evaluation.ragas`` depend on. It lets users take an existing
RAG evaluation dataset (``question`` / ``contexts`` / ``answer``) and turn it
into one that exercises the governance features without asking them to
re-instrument their pipeline.

The adapter only imports ``datasets`` (which is a much smaller dependency
than Ragas itself), so a user who wants to pre-compute nugget metadata for
offline inspection never has to install Ragas.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nuggetindex.core.enums import LifecycleStatus
from nuggetindex.core.models import Nugget
from nuggetindex.store import NuggetStore

if TYPE_CHECKING:
    from datasets import Dataset


# FTS5 MATCH barfs on bare punctuation in queries, so strip everything that
# isn't a word character or whitespace before handing the probe to
# ``aretrieve``. This matches the behaviour users expect from free-text
# keyword search.
_QUERY_TOKEN_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _sanitise_probe(text: str) -> str:
    cleaned = _QUERY_TOKEN_RE.sub(" ", text)
    return " ".join(cleaned.split())


def _require_datasets() -> Any:
    try:
        import datasets  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised indirectly
        raise ImportError(
            "nuggetindex[eval] not installed. Run: pip install nuggetindex[eval]"
        ) from e
    return datasets


def attach_nugget_metadata(
    dataset: Dataset,
    store: NuggetStore,
    *,
    query_time_column: str | None = None,
    context_column: str = "contexts",
    top_k: int = 5,
) -> Dataset:
    """Enrich a Ragas-shaped dataset with nugget governance metadata.

    For each row we call ``store.aretrieve`` once per provided context
    (truncated to 200 chars — the first sentence is usually enough to seed a
    BM25/dense lookup), collect the returned nuggets, dedupe by id, and
    write three new columns:

    - ``retrieved_nuggets`` — list of ``Nugget.model_dump_json()`` strings
    - ``contested_keys`` — list of ``[subject, predicate, scope]`` triples
      for nuggets whose status is ``CONTESTED``
    - ``temporal_valid_count`` — count of retrieved nuggets whose validity
      covers ``query_time`` (if provided) and whose status is not
      ``DEPRECATED``

    Parameters
    ----------
    dataset:
        The HF dataset to enrich. Must contain ``context_column``.
    store:
        A ``NuggetStore`` that will answer the ``aretrieve`` probes.
    query_time_column:
        If given, each row's ``query_time`` field (ISO-8601 string) is
        parsed and passed as ``query_time`` to ``aretrieve``. If absent,
        retrieval uses the "active" view at ingestion time (no temporal
        filter).
    context_column:
        Column name holding the list of contexts. Defaults to ``"contexts"``
        to match the Ragas convention.
    top_k:
        Number of candidates per context to pull from the store. The union
        across contexts is deduped afterwards.
    """
    _require_datasets()

    async def _enrich(row: dict[str, Any]) -> dict[str, Any]:
        qt_raw = row.get(query_time_column) if query_time_column else None
        qt: datetime | None = None
        if qt_raw:
            qt = datetime.fromisoformat(qt_raw)

        retrieved: list[Nugget] = []
        for ctx in row.get(context_column) or []:
            # Truncate the probe — retrieval only needs keywords, not the
            # full document. Keeps the cost bounded on long contexts. We
            # also strip punctuation (FTS5 MATCH rejects bare "."/","/etc.).
            probe = _sanitise_probe((ctx or "")[:200])
            if not probe:
                continue
            results = await store.aretrieve(probe, query_time=qt, top_k=top_k)
            for r in results:
                # ``RetrievalResult`` exposes the source nugget at ``.nugget``.
                retrieved.append(r.nugget)

        # Dedupe — the same nugget frequently surfaces for multiple contexts.
        seen: set[str] = set()
        unique: list[Nugget] = []
        for n in retrieved:
            if n.id in seen:
                continue
            seen.add(n.id)
            unique.append(n)

        contested_keys: list[list[str]] = [
            [n.fact.subject, n.fact.predicate, n.validity.scope]
            for n in unique
            if n.epistemic.status == LifecycleStatus.CONTESTED
        ]

        if qt is None:
            temporal_valid = sum(
                1 for n in unique if n.epistemic.status != LifecycleStatus.DEPRECATED
            )
        else:
            temporal_valid = sum(
                1
                for n in unique
                if n.validity.contains(qt)
                and n.epistemic.status != LifecycleStatus.DEPRECATED
            )

        return {
            "retrieved_nuggets": [n.model_dump_json() for n in unique],
            "contested_keys": contested_keys,
            "temporal_valid_count": temporal_valid,
        }

    def _sync_enrich(row: dict[str, Any]) -> dict[str, Any]:
        # ``Dataset.map`` is a sync API, so we bridge back to async. The
        # simple path (``asyncio.run``) spins a fresh loop; however, inside
        # pytest-asyncio or a Jupyter kernel we'll already be in a running
        # loop and ``asyncio.run`` raises. We dispatch the coroutine to a
        # dedicated worker thread with its own loop in that case — it's a
        # few ms of overhead per row and avoids hard-wiring ``nest_asyncio``.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_enrich(row))

        import threading

        result: dict[str, Any] = {}
        error: list[BaseException] = []

        def _worker() -> None:
            try:
                result.update(asyncio.run(_enrich(row)))
            except BaseException as e:  # noqa: BLE001 — relay to caller
                error.append(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if error:
            raise error[0]
        return result

    return dataset.map(_sync_enrich)
