"""Pgvector backend tests (gated by ``pytest.importorskip``).

These tests require both the ``[pgvector]`` extra (asyncpg + pgvector Python
package) and a live Postgres instance that has the ``vector`` extension
available. We use ``pytest-postgresql`` to spin up a temp DB when available.
When any of those prerequisites are missing, tests skip cleanly.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

import numpy as np  # noqa: E402

from nuggetindex.store.backends.pgvector_backend import PgvectorBackend  # noqa: E402

# ``pytest-postgresql`` is optional for the local dev loop. CI installs it in
# the dedicated Postgres+pgvector job. If it's missing, we skip this file.
pytest.importorskip("pytest_postgresql")


def _make_stub_encoder(dim: int = 384) -> Any:
    """Return a deterministic encoder that doesn't download any models."""
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
def dsn(postgresql: Any) -> str:
    """Build an asyncpg DSN against the pytest-postgresql temp DB.

    If the pgvector extension isn't available on the host Postgres, skip.
    """
    cursor = postgresql.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        postgresql.commit()
    except Exception:  # pragma: no cover - environment-dependent
        pytest.skip("pgvector extension not installed on test Postgres")
    info = postgresql.info
    return f"postgresql://{info.user}@{info.host}:{info.port}/{info.dbname}"


@pytest.mark.asyncio
async def test_upsert_and_search(dsn: str) -> None:
    encoder = _make_stub_encoder(dim=8)
    backend = PgvectorBackend(dsn=dsn, dim=8, encoder=encoder)
    vecs = encoder(["alpha", "beta", "gamma"])
    await backend.aupsert_batch(
        [
            ("n1", list(vecs[0])),
            ("n2", list(vecs[1])),
            ("n3", list(vecs[2])),
        ]
    )

    results = await backend.asearch("alpha", top_k=3)
    assert len(results) >= 1
    assert results[0][0] == "n1"
    await backend.aclose()


@pytest.mark.asyncio
async def test_adelete(dsn: str) -> None:
    encoder = _make_stub_encoder(dim=8)
    backend = PgvectorBackend(dsn=dsn, dim=8, encoder=encoder)
    vecs = encoder(["alpha", "beta"])
    await backend.aupsert_batch([("n1", list(vecs[0])), ("n2", list(vecs[1]))])

    await backend.adelete(["n1"])
    results = await backend.asearch("alpha", top_k=5)
    assert "n1" not in {nid for nid, _ in results}
    await backend.aclose()


@pytest.mark.asyncio
async def test_candidate_id_filter(dsn: str) -> None:
    encoder = _make_stub_encoder(dim=8)
    backend = PgvectorBackend(dsn=dsn, dim=8, encoder=encoder)
    vecs = encoder(["alpha", "alphabet", "zzz"])
    await backend.aupsert_batch(
        [
            ("n1", list(vecs[0])),
            ("n2", list(vecs[1])),
            ("n3", list(vecs[2])),
        ]
    )

    results = await backend.asearch("alpha", candidate_ids=["n2", "n3"], top_k=5)
    ids = {nid for nid, _ in results}
    assert "n1" not in ids
    assert ids.issubset({"n2", "n3"})
    await backend.aclose()


@pytest.mark.asyncio
async def test_empty_collection_returns_empty(dsn: str) -> None:
    encoder = _make_stub_encoder(dim=8)
    backend = PgvectorBackend(dsn=dsn, dim=8, encoder=encoder)
    results = await backend.asearch("anything", top_k=3)
    assert results == []
    await backend.aclose()


def test_missing_deps_import_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the module should succeed even when deps are missing; the
    guard only fires inside ``_require_pgvector_deps`` (called from
    ``_ensure_pool``). We simulate the missing-deps path by forcing the
    helper to raise."""
    from nuggetindex.store.backends import pgvector_backend as mod

    def fake_require() -> Any:  # pragma: no cover - trivial
        raise ImportError(
            "nuggetindex[pgvector] not installed. Run: pip install 'nuggetindex[pgvector]'"
        )

    monkeypatch.setattr(mod, "_require_pgvector_deps", fake_require)
    backend = mod.PgvectorBackend(
        dsn="postgresql://u@localhost:1/none",
        dim=4,
        encoder=lambda texts: np.zeros((len(texts), 4), dtype="float32"),
    )

    import asyncio

    with pytest.raises(ImportError, match="nuggetindex\\[pgvector\\]"):
        asyncio.run(backend.aupsert("n1", [0.0, 0.0, 0.0, 0.0]))
