"""Tests for the ``DenseBackend`` protocol + ``default_encoder`` shim."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from nuggetindex.store.dense import DenseBackend, default_encoder


class _StubDense:
    """Minimal DenseBackend-compatible stub used to exercise isinstance()."""

    async def aupsert(self, id: str, vector: list[float]) -> None:
        return None

    async def aupsert_batch(self, items: list[tuple[str, list[float]]]) -> None:
        return None

    async def asearch(
        self,
        query: str,
        *,
        candidate_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        return []

    async def adelete(self, ids: Iterable[str]) -> None:
        return None

    async def aclose(self) -> None:
        return None


def test_stub_is_structural_dense_backend() -> None:
    # Protocol is structural (not runtime_checkable), so we assert by duck-type:
    backend: DenseBackend = _StubDense()
    assert hasattr(backend, "aupsert")
    assert hasattr(backend, "aupsert_batch")
    assert hasattr(backend, "asearch")
    assert hasattr(backend, "adelete")
    assert hasattr(backend, "aclose")


def test_default_encoder_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """default_encoder is @lru_cache(maxsize=1) — calls return same object."""
    calls: list[int] = []

    class _FakeModel:
        def __init__(self, name: str) -> None:  # noqa: ARG002
            calls.append(1)

        def encode(
            self,
            texts: list[str],
            normalize_embeddings: bool = True,  # noqa: ARG002
        ) -> Any:
            import numpy as np

            return np.zeros((len(texts), 4), dtype="float32")

    import nuggetindex.store.dense as dense_mod

    # Clear cached encoder from any prior test.
    default_encoder.cache_clear()

    # Replace the sentence_transformers.SentenceTransformer symbol.
    import sys
    import types

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    enc1 = dense_mod.default_encoder()
    enc2 = dense_mod.default_encoder()
    assert enc1 is enc2, "default_encoder must be cached"
    assert len(calls) == 1, "underlying model constructed only once"

    # The returned callable should delegate to the model's .encode.
    vec = enc1(["hello", "world"])
    assert vec.shape == (2, 4)

    # Tidy up for other tests that may want their own monkeypatched encoder.
    default_encoder.cache_clear()


def test_default_encoder_missing_extra_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When sentence_transformers is missing, raise a helpful ImportError."""
    import sys

    # Hide any real sentence_transformers module and block import.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    default_encoder.cache_clear()

    with pytest.raises(ImportError) as exc_info:
        default_encoder()
    assert "nuggetindex[dense]" in str(exc_info.value)

    default_encoder.cache_clear()
