"""Chroma dense backend (optional).

Satisfies the ``DenseBackend`` protocol via ``chromadb``. Uses
``PersistentClient(path=...)`` when a path is supplied, or the in-memory
``EphemeralClient`` for tests. Nugget IDs are written through as Chroma IDs
directly (Chroma accepts arbitrary strings).

The ``chromadb`` SDK is imported lazily inside ``__init__``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _require_chromadb_sdk() -> Any:
    try:
        import chromadb
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "chromadb is not installed. Run: pip install 'nuggetindex[chroma]'"
        ) from e
    return chromadb


class ChromaBackend:
    """Chroma-backed implementation of ``DenseBackend``.

    Parameters
    ----------
    persist_directory:
        Directory on disk for a ``PersistentClient``. Pass ``None`` for an
        in-memory ``EphemeralClient`` (handy in tests).
    collection_name:
        Name of the Chroma collection.
    encoder:
        Callable ``(list[str]) -> ndarray`` used to embed queries. Defaults
        to the cached bge-small encoder.
    """

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        *,
        collection_name: str = "nuggetindex",
        encoder: Any | None = None,
    ) -> None:
        chromadb = _require_chromadb_sdk()
        self.collection_name = collection_name
        if encoder is None:
            from nuggetindex.store.dense import default_encoder

            encoder = default_encoder()
        self.encoder = encoder

        if persist_directory is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=str(persist_directory))

        # Inner-product ("ip") keeps us consistent with FAISS/Qdrant when the
        # encoder yields normalized vectors (cosine equivalence).
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "ip"},
        )

    # --- upsert ---------------------------------------------------------

    async def aupsert(self, id: str, vector: list[float]) -> None:
        await self.aupsert_batch([(id, vector)])

    async def aupsert_batch(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        await asyncio.get_running_loop().run_in_executor(None, self._upsert_sync, items)

    def _upsert_sync(self, items: list[tuple[str, list[float]]]) -> None:
        import numpy as np

        ids = [nid for nid, _ in items]
        # Chroma normalization chokes on lists-of-np.float32; stack into a
        # proper 2D float32 array which Chroma accepts directly.
        embeddings = np.asarray([list(vec) for _, vec in items], dtype="float32")
        self._collection.upsert(ids=ids, embeddings=embeddings)

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
        import numpy as np

        q_vec = np.asarray(self.encoder([query]), dtype="float32")
        if q_vec.ndim == 2:
            q_vec = q_vec[0]

        kwargs: dict[str, Any] = {
            # Pass as a proper numpy array to avoid Chroma's np.float32-in-list
            # normalization error.
            "query_embeddings": q_vec.reshape(1, -1).astype("float32"),
            "n_results": top_k,
        }
        if candidate_ids is not None:
            kwargs["ids"] = list(candidate_ids)

        result = self._collection.query(**kwargs)
        # Chroma returns dicts with per-query lists; we only send one query.
        ids_batch = result.get("ids") or [[]]
        dist_batch = result.get("distances") or [[]]
        ids = ids_batch[0] if ids_batch else []
        distances = dist_batch[0] if dist_batch else []

        # For inner-product space in Chroma, "distance" == -similarity (the
        # smaller the distance, the better). Flip sign so downstream fusion
        # treats a larger score as a better match — consistent with FAISS
        # and Qdrant backends.
        out: list[tuple[str, float]] = []
        for nid, dist in zip(ids, distances, strict=False):
            out.append((str(nid), -float(dist)))
        return out

    # --- delete / close -------------------------------------------------

    async def adelete(self, ids: Iterable[str]) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._collection.delete(ids=ids_list)
        )

    async def aclose(self) -> None:
        # PersistentClient flushes on each write; no explicit close is needed.
        return None
