"""DenseBackend protocol + default encoder (bge-small).

The ``DenseBackend`` Protocol is the contract every pluggable dense backend
(FAISS, Qdrant, Chroma, ...) must satisfy. ``NuggetStore`` treats all dense
backends structurally — there is no inheritance requirement.

``default_encoder`` loads ``BAAI/bge-small-en-v1.5`` via sentence-transformers
and caches the instance with ``functools.lru_cache`` so the heavy model load
only happens once per process.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any, Protocol


class DenseBackend(Protocol):
    """Structural protocol implemented by every dense backend.

    Backends are plugged into ``NuggetStore(dense=...)`` and consumed by
    ``Retriever``. Implementations must be safe to call from ``asyncio`` —
    CPU-bound work should be dispatched to an executor internally.
    """

    async def aupsert(self, id: str, vector: list[float]) -> None: ...

    async def aupsert_batch(
        self, items: list[tuple[str, list[float]]]
    ) -> None: ...

    async def asearch(
        self,
        query: str,
        *,
        candidate_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[str, float]]: ...

    async def adelete(self, ids: Iterable[str]) -> None: ...

    async def aclose(self) -> None: ...


Encoder = Callable[[list[str]], Any]


@lru_cache(maxsize=1)
def default_encoder() -> Encoder:
    """Return the default ``bge-small-en-v1.5`` encoder callable.

    The returned callable has signature ``(texts: list[str]) -> numpy.ndarray``
    and returns L2-normalized embeddings (cosine-ready). Call it twice and
    you get the same underlying ``SentenceTransformer`` instance (cached).

    Raises ``ImportError`` with a helpful message when the ``[dense]`` extra
    is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[dense] not installed. "
            "Run: pip install 'nuggetindex[dense]'"
        ) from e

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def encode(texts: list[str]) -> Any:
        return model.encode(texts, normalize_embeddings=True)

    return encode
