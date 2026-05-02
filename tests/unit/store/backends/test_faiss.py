"""FAISS backend tests.

Gated with ``pytest.importorskip('faiss')`` so the suite still passes when the
``[dense]`` extra is not installed. Uses a stub encoder throughout — tests do
NOT download a real sentence-transformers model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("faiss")

import numpy as np  # noqa: E402

from nuggetindex.store.backends.faiss_backend import FAISSBackend  # noqa: E402


def make_stub_encoder(dim: int = 8) -> Any:
    """Deterministic encoder: hashes text -> fixed dim vector.

    Normalized to the unit sphere so inner-product ~ cosine similarity.
    """
    rng = np.random.default_rng(0)
    cache: dict[str, np.ndarray] = {}

    def encode(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t not in cache:
                # Seeded so each unique string has a stable vector.
                local_rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = local_rng.standard_normal(dim).astype("float32")
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                cache[t] = v
            out.append(cache[t])
        # Ensure rng stays used (avoid unused-var warnings).
        _ = rng
        return np.stack(out, axis=0)

    return encode


@pytest.fixture
def stub_encoder() -> Any:
    return make_stub_encoder(dim=8)


def _make_backend(tmp_path: Path, stub_encoder: Any, dim: int = 8) -> FAISSBackend:
    # nlist=2 is fine for our tiny test datasets; training uses 1024 random
    # vectors internally so clusters are well populated.
    return FAISSBackend(tmp_path / "faiss.index", encoder=stub_encoder, dim=dim, nlist=2)


@pytest.mark.asyncio
async def test_upsert_and_search(tmp_path: Path, stub_encoder: Any) -> None:
    backend = _make_backend(tmp_path, stub_encoder)
    # Encode the strings and insert with their matching vectors.
    texts = {"n1": "alpha", "n2": "beta", "n3": "gamma"}
    vecs = stub_encoder(list(texts.values()))
    items = [(nid, list(vecs[i])) for i, nid in enumerate(texts)]
    await backend.aupsert_batch(items)

    results = await backend.asearch("alpha", top_k=2)
    assert len(results) >= 1
    top_ids = [nid for nid, _ in results]
    assert "n1" in top_ids  # exact-match text should win
    await backend.aclose()


@pytest.mark.asyncio
async def test_search_filters_to_candidate_ids(tmp_path: Path, stub_encoder: Any) -> None:
    backend = _make_backend(tmp_path, stub_encoder)
    texts = {"n1": "alpha", "n2": "alphabet", "n3": "zzz"}
    vecs = stub_encoder(list(texts.values()))
    await backend.aupsert_batch([(nid, list(vecs[i])) for i, nid in enumerate(texts)])

    # Constrain to just n2/n3: n1 must NOT show up.
    results = await backend.asearch("alpha", candidate_ids=["n2", "n3"], top_k=5)
    ids = {nid for nid, _ in results}
    assert "n1" not in ids
    assert ids.issubset({"n2", "n3"})
    await backend.aclose()


@pytest.mark.asyncio
async def test_adelete_soft_delete(tmp_path: Path, stub_encoder: Any) -> None:
    backend = _make_backend(tmp_path, stub_encoder)
    texts = {"n1": "alpha", "n2": "beta"}
    vecs = stub_encoder(list(texts.values()))
    await backend.aupsert_batch([(nid, list(vecs[i])) for i, nid in enumerate(texts)])

    await backend.adelete(["n1"])

    results = await backend.asearch("alpha", top_k=5)
    ids = {nid for nid, _ in results}
    assert "n1" not in ids
    await backend.aclose()


@pytest.mark.asyncio
async def test_persistence_roundtrip(tmp_path: Path, stub_encoder: Any) -> None:
    path = tmp_path / "faiss.index"
    backend = FAISSBackend(path, encoder=stub_encoder, dim=8, nlist=2)
    vecs = stub_encoder(["alpha", "beta"])
    await backend.aupsert_batch([("n1", list(vecs[0])), ("n2", list(vecs[1]))])
    await backend.aclose()

    # Re-open: id map + faiss file should both be present.
    assert path.exists()
    assert path.with_suffix(".ids.json").exists()

    backend2 = FAISSBackend(path, encoder=stub_encoder, dim=8, nlist=2)
    results = await backend2.asearch("alpha", top_k=2)
    top_ids = [nid for nid, _ in results]
    assert "n1" in top_ids
    await backend2.aclose()


@pytest.mark.asyncio
async def test_empty_index_returns_empty_search(tmp_path: Path, stub_encoder: Any) -> None:
    backend = _make_backend(tmp_path, stub_encoder)
    results = await backend.asearch("anything", top_k=5)
    assert results == []
    await backend.aclose()


@pytest.mark.asyncio
async def test_invalid_vector_dim_raises(tmp_path: Path, stub_encoder: Any) -> None:
    backend = _make_backend(tmp_path, stub_encoder, dim=8)
    with pytest.raises(ValueError, match="dim"):
        await backend.aupsert("n1", [0.0, 0.0, 0.0])  # wrong dim
    await backend.aclose()


@pytest.mark.asyncio
async def test_nuggetstore_integration_with_faiss(tmp_path: Path, stub_encoder: Any) -> None:
    """End-to-end: NuggetStore + FAISSBackend produces hybrid-ranked results."""
    from datetime import UTC, datetime

    from nuggetindex.core.enums import NuggetKind
    from nuggetindex.core.models import (
        EpistemicState,
        FactTriple,
        Nugget,
        ProvenanceRecord,
        ValidityInterval,
    )
    from nuggetindex.store import NuggetStore

    faiss_path = tmp_path / "faiss.index"
    backend = FAISSBackend(faiss_path, encoder=stub_encoder, dim=8, nlist=2)
    store = NuggetStore(db_path=tmp_path / "store.db", dense=backend, encoder=stub_encoder)

    n1 = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="ceo",
            object="Pichai",
            text="Sundar Pichai is CEO of Google",
        ),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="doc-1", evidence_span="x"),),
    )
    n2 = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Apple",
            predicate="hq",
            object="Cupertino",
            text="Apple is in Cupertino",
        ),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="doc-1", evidence_span="y"),),
    )
    for n in (n1, n2):
        await store.aadd(n)
        vec = stub_encoder([n.fact.text])[0]
        await backend.aupsert(n.id, list(vec))

    results = await store.aretrieve("CEO of Google", top_k=5)
    assert len(results) >= 1
    top = results[0]
    # Both components should be populated for the top hit.
    assert top.sparse_score is not None
    assert top.dense_score is not None
    assert "sparse" in top.component_ranks
    assert "dense" in top.component_ranks
    await store.aclose()
    await backend.aclose()
