"""Postgres + pgvector dense backend (optional).

Satisfies the :class:`~nuggetindex.store.dense.DenseBackend` protocol via
`asyncpg <https://magicstack.github.io/asyncpg/>`_ and the
`pgvector <https://github.com/pgvector/pgvector>`_ Postgres extension.

The ``asyncpg`` and ``pgvector`` packages are imported lazily inside
``_ensure_pool`` so merely importing this module does not require the
``[pgvector]`` extra to be installed. The import guard fires the first time
a coroutine actually needs a database connection.

Schema (created on first use)::

    CREATE TABLE nugget_vectors (
        id        TEXT PRIMARY KEY,
        embedding vector(<dim>)
    );
    CREATE INDEX nugget_vectors_ivfflat
        ON nugget_vectors USING ivfflat (embedding vector_cosine_ops);

Cosine similarity is computed as ``1 - (embedding <=> query)``; pgvector's
``<=>`` operator is cosine *distance* (smaller is better), so the subtraction
flips the sign so downstream hybrid-fusion logic sees "bigger is better"
scores, consistent with FAISS/Qdrant/Chroma backends.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def _require_pgvector_deps() -> tuple[Any, Any]:
    """Import ``asyncpg`` and ``pgvector.asyncpg.register_vector`` lazily.

    Raises :class:`ImportError` with an actionable message pointing at the
    ``[pgvector]`` extra when the dependencies are missing.
    """
    try:
        import asyncpg
        from pgvector.asyncpg import register_vector
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[pgvector] not installed. Run: pip install 'nuggetindex[pgvector]'"
        ) from e
    return asyncpg, register_vector


class PgvectorBackend:
    """Postgres + pgvector implementation of ``DenseBackend``.

    Parameters
    ----------
    dsn:
        libpq-style connection string passed to
        ``asyncpg.create_pool(dsn=...)``. Examples:
        ``postgresql://user:pass@localhost:5432/mydb``.
    table:
        Table name. Defaults to ``nugget_vectors``. Created on first use.
    dim:
        Vector dimensionality. Must match the encoder output. Defaults to
        384 (bge-small).
    encoder:
        Callable ``(list[str]) -> ndarray`` used to embed queries. Defaults
        to the cached bge-small encoder from :mod:`nuggetindex.store.dense`.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "nugget_vectors",
        dim: int = 384,
        encoder: Callable[[list[str]], Any] | None = None,
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._dim = dim
        self._encoder = encoder  # resolved lazily if None
        self._pool: Any | None = None

    # --- lifecycle ------------------------------------------------------

    async def _ensure_pool(self) -> None:
        """Create the connection pool and provision the schema on first use."""
        if self._pool is not None:
            return
        asyncpg, register_vector = _require_pgvector_deps()

        async def _init(conn: Any) -> None:
            # ``register_vector`` is what teaches asyncpg to marshal Python
            # lists <-> ``vector(N)`` columns. It must be called on every
            # connection the pool hands out, hence ``init=``.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector(conn)

        self._pool = await asyncpg.create_pool(dsn=self._dsn, init=_init, min_size=1, max_size=4)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"  id TEXT PRIMARY KEY,"
                f"  embedding vector({self._dim})"
                f")"
            )
            # IVF-Flat is pgvector's approximate-nearest-neighbour index.
            # Postgres falls back to a sequential scan on very small tables,
            # which is fine for tests.
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_ivfflat "
                f"ON {self._table} USING ivfflat (embedding vector_cosine_ops)"
            )

    def _resolve_encoder(self) -> Callable[[list[str]], Any]:
        if self._encoder is None:
            from nuggetindex.store.dense import default_encoder

            self._encoder = default_encoder()
        return self._encoder

    # --- upsert ---------------------------------------------------------

    async def aupsert(self, id: str, vector: list[float]) -> None:
        await self.aupsert_batch([(id, vector)])

    async def aupsert_batch(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        await self._ensure_pool()
        assert self._pool is not None
        # pgvector's asyncpg adapter accepts a Python list[float] directly
        # once ``register_vector`` has run on the connection.
        payload = [(nid, list(vec)) for nid, vec in items]
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                f"INSERT INTO {self._table} (id, embedding) VALUES ($1, $2) "
                f"ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
                payload,
            )

    # --- search ---------------------------------------------------------

    async def asearch(
        self,
        query: str,
        *,
        candidate_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        await self._ensure_pool()
        assert self._pool is not None
        import numpy as np

        encoder = self._resolve_encoder()
        q_vec = np.asarray(encoder([query]), dtype="float32")
        if q_vec.ndim == 2:
            q_vec = q_vec[0]
        q_list = list(q_vec.tolist())

        async with self._pool.acquire() as conn:
            if candidate_ids is not None:
                sql = (
                    f"SELECT id, 1 - (embedding <=> $1) AS similarity "
                    f"FROM {self._table} WHERE id = ANY($2) "
                    f"ORDER BY similarity DESC LIMIT $3"
                )
                rows = await conn.fetch(sql, q_list, list(candidate_ids), top_k)
            else:
                sql = (
                    f"SELECT id, 1 - (embedding <=> $1) AS similarity "
                    f"FROM {self._table} "
                    f"ORDER BY embedding <=> $1 LIMIT $2"
                )
                rows = await conn.fetch(sql, q_list, top_k)
        return [(str(r["id"]), float(r["similarity"])) for r in rows]

    # --- delete / close -------------------------------------------------

    async def adelete(self, ids: Iterable[str]) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        await self._ensure_pool()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table} WHERE id = ANY($1)", ids_list)

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
