"""FAISS sidecar-file dense backend.

An ``IndexIVFFlat`` (inner-product metric) is persisted as a FAISS binary
file; the nugget-id → FAISS-row mapping lives next to it as a JSON sidecar.
Deletion is soft (IVF cannot remove rows without a full rebuild) — we drop
the id→row mapping and filter orphaned rows at search time.

The FAISS SDK is imported lazily inside ``__init__`` so users without
``pip install nuggetindex[dense]`` can still import this module without
triggering an ``ImportError``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def _require_faiss_sdk() -> Any:
    """Import and return the ``faiss`` module, with a helpful error message."""
    try:
        import faiss
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError("FAISS is not installed. Run: pip install 'nuggetindex[dense]'") from e
    return faiss


def _lazy_default_encoder() -> Any:
    from nuggetindex.store.dense import default_encoder

    return default_encoder()


class FAISSBackend:
    """File-based FAISS ``IndexIVFFlat`` with an inner-product metric.

    Parameters
    ----------
    path:
        Where the FAISS binary index is written. The id map lives at
        ``path.with_suffix(".ids.json")``.
    encoder:
        Callable ``(list[str]) -> np.ndarray`` mapping strings to
        (optionally L2-normalized) vectors. Defaults to the cached
        ``bge-small-en-v1.5`` encoder.
    dim:
        Embedding dimensionality. Defaults to 384 (bge-small output).
    nlist:
        Number of IVF clusters. Must be <= number of training points
        (we train on ``max(nlist * 40, 1024)`` random vectors at init).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        encoder: Any | None = None,
        dim: int = 384,
        nlist: int = 100,
    ) -> None:
        faiss = _require_faiss_sdk()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder if encoder is not None else _lazy_default_encoder()
        self.dim = dim
        self.nlist = nlist
        self._lock = asyncio.Lock()
        self._id_map_path = self.path.with_suffix(".ids.json")

        if self.path.exists():
            self._index = faiss.read_index(str(self.path))
        else:
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            # IVF must be trained before .add(); train on random vectors now
            # so first ingest doesn't need to collect a full training batch.
            n_train = max(nlist * 40, 1024)
            training = np.random.randn(n_train, dim).astype("float32")
            # Normalize so training distribution roughly matches inference vectors.
            norms = np.linalg.norm(training, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            training = training / norms
            self._index.train(training)
        # Probe multiple clusters at search time so recall doesn't depend on
        # the query landing in the same centroid as the docs. We cap at nlist
        # to stay within the trained layout.
        self._index.nprobe = min(max(nlist // 4, 8), nlist)

        if self._id_map_path.exists():
            raw = json.loads(self._id_map_path.read_text())
            # JSON keys are strings; rows are ints.
            self._id_to_row: dict[str, int] = {k: int(v) for k, v in raw.items()}
        else:
            self._id_to_row = {}
        self._row_to_id: dict[int, str] = {v: k for k, v in self._id_to_row.items()}

    # --- upsert ---------------------------------------------------------

    async def aupsert(self, id: str, vector: list[float]) -> None:
        await self.aupsert_batch([(id, vector)])

    async def aupsert_batch(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        async with self._lock:
            await asyncio.get_running_loop().run_in_executor(None, self._upsert_sync, items)

    def _upsert_sync(self, items: list[tuple[str, list[float]]]) -> None:
        vectors = np.asarray([v for _, v in items], dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of dim {self.dim}, got shape {vectors.shape}")
        start_row = self._index.ntotal
        self._index.add(vectors)
        for i, (nid, _) in enumerate(items):
            row = start_row + i
            self._id_to_row[nid] = row
            self._row_to_id[row] = nid
        self._persist()

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
        if self._index.ntotal == 0:
            return []
        q_vec = np.asarray(self.encoder([query]), dtype="float32")
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        # Over-retrieve to account for (a) deleted rows that still occupy
        # FAISS slots and (b) candidate-id filtering. Bound by ntotal.
        over_k = top_k * 4 if candidate_ids else top_k * 2
        over_k = min(max(over_k, top_k), self._index.ntotal)
        distances, indices = self._index.search(q_vec, over_k)

        cand_set = set(candidate_ids) if candidate_ids is not None else None
        results: list[tuple[str, float]] = []
        for row, dist in zip(indices[0], distances[0], strict=False):
            if row == -1:
                continue
            nid = self._row_to_id.get(int(row))
            if nid is None:
                # Row was soft-deleted or belongs to untracked data.
                continue
            if cand_set is not None and nid not in cand_set:
                continue
            results.append((nid, float(dist)))
            if len(results) >= top_k:
                break
        return results

    # --- delete / close -------------------------------------------------

    async def adelete(self, ids: Iterable[str]) -> None:
        async with self._lock:
            ids_list = list(ids)
            for nid in ids_list:
                row = self._id_to_row.pop(nid, None)
                if row is not None:
                    self._row_to_id.pop(row, None)
            self._persist_id_map()

    async def aclose(self) -> None:
        async with self._lock:
            await asyncio.get_running_loop().run_in_executor(None, self._persist)

    # --- persistence helpers --------------------------------------------

    def _persist(self) -> None:
        faiss = _require_faiss_sdk()
        faiss.write_index(self._index, str(self.path))
        self._persist_id_map()

    def _persist_id_map(self) -> None:
        self._id_map_path.write_text(json.dumps(self._id_to_row))
