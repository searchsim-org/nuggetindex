"""Qdrant dense backend (optional).

Satisfies the ``DenseBackend`` protocol via ``qdrant_client``. Nugget IDs are
hex strings (16 chars) — Qdrant point IDs must be either an unsigned integer
or a UUID string, so we map each nugget ID to a deterministic UUIDv5 and keep
the original in the point's ``payload['nugget_id']`` for filter/recovery.

The ``qdrant_client`` SDK is imported lazily inside ``__init__`` so importing
this module does not require the ``[qdrant]`` extra.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from typing import Any

# Stable namespace for mapping a nugget-id string to a UUIDv5 point id. Any
# fixed UUID works; we use one that's clearly nuggetindex-scoped.
_POINT_NS = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")


def _require_qdrant_sdk() -> Any:
    try:
        import qdrant_client
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "qdrant-client is not installed. Run: pip install 'nuggetindex[qdrant]'"
        ) from e
    return qdrant_client


def _nid_to_point_id(nugget_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, nugget_id))


class QdrantBackend:
    """Qdrant-backed implementation of ``DenseBackend``.

    Parameters
    ----------
    url:
        URL passed to ``QdrantClient`` (``":memory:"`` for in-process tests).
        Accepts ``None`` to use Qdrant's default host:port.
    collection_name:
        Name of the Qdrant collection. Created on first use with
        inner-product distance and the configured ``dim``.
    encoder:
        Callable ``(list[str]) -> ndarray`` used to embed queries. Defaults
        to the cached bge-small encoder.
    dim:
        Vector dimensionality. Defaults to 384.
    """

    def __init__(
        self,
        url: str | None = ":memory:",
        *,
        collection_name: str = "nuggetindex",
        encoder: Any | None = None,
        dim: int = 384,
    ) -> None:
        qdrant = _require_qdrant_sdk()
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qmodels = qmodels
        self.collection_name = collection_name
        self.dim = dim
        if encoder is None:
            from nuggetindex.store.dense import default_encoder

            encoder = default_encoder()
        self.encoder = encoder

        if url in (None, ":memory:"):
            self._client: QdrantClient = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=url)

        # Keep the ``qdrant`` module reference for error introspection.
        self._qdrant = qdrant

        # Create collection if missing. Use inner-product since encoders are
        # already L2-normalized for cosine.
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.DOT),
            )

    # --- upsert ---------------------------------------------------------

    async def aupsert(self, id: str, vector: list[float]) -> None:
        await self.aupsert_batch([(id, vector)])

    async def aupsert_batch(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        await asyncio.get_running_loop().run_in_executor(None, self._upsert_sync, items)

    def _upsert_sync(self, items: list[tuple[str, list[float]]]) -> None:
        qmodels = self._qmodels
        points = [
            qmodels.PointStruct(
                id=_nid_to_point_id(nid),
                vector=list(vec),
                payload={"nugget_id": nid},
            )
            for nid, vec in items
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    # --- search ---------------------------------------------------------

    async def asearch(
        self,
        query: str,
        *,
        candidate_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._search_sync, query, candidate_ids, top_k
        )

    def _search_sync(
        self,
        query: str,
        candidate_ids: list[str] | None,
        top_k: int,
    ) -> list[tuple[str, float]]:
        qmodels = self._qmodels
        import numpy as np

        q_vec = np.asarray(self.encoder([query]), dtype="float32")
        if q_vec.ndim == 2:
            q_vec = q_vec[0]
        query_filter = None
        if candidate_ids is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="nugget_id",
                        match=qmodels.MatchAny(any=list(candidate_ids)),
                    )
                ]
            )
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=list(q_vec.tolist()),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        out: list[tuple[str, float]] = []
        for hit in response.points:
            payload = hit.payload or {}
            nid = payload.get("nugget_id")
            if nid is None:
                continue
            out.append((str(nid), float(hit.score)))
        return out

    # --- delete / close -------------------------------------------------

    async def adelete(self, ids: Iterable[str]) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        qmodels = self._qmodels
        point_ids = [_nid_to_point_id(nid) for nid in ids_list]
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
            ),
        )

    async def aclose(self) -> None:
        # Qdrant's Python client has no explicit close() for the in-memory
        # path; local HTTP connections are closed by the client's __del__.
        close = getattr(self._client, "close", None)
        if callable(close):
            await asyncio.get_running_loop().run_in_executor(None, close)
