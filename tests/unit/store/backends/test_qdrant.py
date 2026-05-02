"""Qdrant backend tests (gated by ``pytest.importorskip``)."""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("qdrant_client")

import numpy as np  # noqa: E402

from nuggetindex.store.backends.qdrant_backend import QdrantBackend  # noqa: E402


def make_stub_encoder(dim: int = 8) -> Any:
    cache: dict[str, np.ndarray] = {}

    def encode(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t not in cache:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal(dim).astype("float32")
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                cache[t] = v
            out.append(cache[t])
        return np.stack(out, axis=0)

    return encode


@pytest.fixture
def stub_encoder() -> Any:
    return make_stub_encoder(dim=8)


@pytest.mark.asyncio
async def test_upsert_and_search(stub_encoder: Any) -> None:
    backend = QdrantBackend(
        ":memory:",
        collection_name="test-upsert",
        encoder=stub_encoder,
        dim=8,
    )
    texts = {"n1": "alpha", "n2": "beta", "n3": "gamma"}
    vecs = stub_encoder(list(texts.values()))
    await backend.aupsert_batch(
        [(nid, list(vecs[i])) for i, nid in enumerate(texts)]
    )

    results = await backend.asearch("alpha", top_k=3)
    assert len(results) >= 1
    assert results[0][0] == "n1"
    await backend.aclose()


@pytest.mark.asyncio
async def test_candidate_id_filter(stub_encoder: Any) -> None:
    backend = QdrantBackend(
        ":memory:",
        collection_name="test-candidates",
        encoder=stub_encoder,
        dim=8,
    )
    texts = {"n1": "alpha", "n2": "alphabet", "n3": "zzz"}
    vecs = stub_encoder(list(texts.values()))
    await backend.aupsert_batch(
        [(nid, list(vecs[i])) for i, nid in enumerate(texts)]
    )

    results = await backend.asearch(
        "alpha", candidate_ids=["n2", "n3"], top_k=5
    )
    ids = {nid for nid, _ in results}
    assert "n1" not in ids
    assert ids.issubset({"n2", "n3"})
    await backend.aclose()


@pytest.mark.asyncio
async def test_delete(stub_encoder: Any) -> None:
    backend = QdrantBackend(
        ":memory:",
        collection_name="test-delete",
        encoder=stub_encoder,
        dim=8,
    )
    vecs = stub_encoder(["alpha", "beta"])
    await backend.aupsert_batch([("n1", list(vecs[0])), ("n2", list(vecs[1]))])

    await backend.adelete(["n1"])
    results = await backend.asearch("alpha", top_k=5)
    assert "n1" not in {nid for nid, _ in results}
    await backend.aclose()


@pytest.mark.asyncio
async def test_empty_collection_returns_empty(stub_encoder: Any) -> None:
    backend = QdrantBackend(
        ":memory:",
        collection_name="test-empty",
        encoder=stub_encoder,
        dim=8,
    )
    results = await backend.asearch("anything", top_k=3)
    assert results == []
    await backend.aclose()
