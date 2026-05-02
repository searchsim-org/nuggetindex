"""Tests for :func:`nuggetindex.auto.auto`'s cost-estimate threading.

The previous revision silently defaulted ``model_id`` to ``gpt-4o-mini``
regardless of the user's extractor, so :class:`AutoReport.cost_est_usd`
underreported costs on any non-``gpt-4o-mini`` configuration. These
tests pin the behaviour: the trigger extractor routes through the
``"trigger"`` price row (free), and an :class:`LLMExtractor` threads
``cfg.model`` into :func:`estimate_ingest_cost`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nuggetindex.auto import auto
from nuggetindex.pipeline.constructor import Document


def _synthetic_docs(n: int) -> list[Document]:
    return [
        Document(
            source_id=f"d{i}",
            text=f"doc {i}: Satya Nadella became CEO of Microsoft in 2014.",
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_cost_is_zero_for_trigger(tmp_path: Path) -> None:
    """Default (trigger) extractor -> cost_est_usd == 0.0 (no LLM)."""
    _sidecar, report = await auto(
        docs=_synthetic_docs(3),
        budget=2,
        store_path=tmp_path / "store.db",
        cache_path=tmp_path / "cache.db",
    )
    try:
        assert report.cost_est_usd == 0.0
    finally:
        await _sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_cost_reflects_llm_extractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLMExtractor threads its ``cfg.model`` into estimate_ingest_cost."""
    pytest.importorskip("openai")
    from nuggetindex.extractors.clients.base import LLMConfig
    from nuggetindex.extractors.llm import LLMExtractor

    # A lightweight extractor stub: we patch the instance so its aextract
    # returns [] without hitting any backend. The cost estimator path runs
    # independently of what the extractor actually produces.
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor.cfg = cfg  # type: ignore[attr-defined]

    async def _aextract(_text: str, **_kwargs: object) -> list[object]:
        return []

    extractor.aextract = _aextract  # type: ignore[attr-defined]
    extractor.accepts_source_id = lambda: False  # type: ignore[attr-defined]

    # Spy on estimate_ingest_cost. We return a fixed CostEstimate so auto()'s
    # downstream .net_cost_usd_est coercion still works.
    recorded_kwargs: dict[str, object] = {}

    from nuggetindex.audit import cost as cost_module

    original = cost_module.estimate_ingest_cost

    async def spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        recorded_kwargs.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(cost_module, "estimate_ingest_cost", spy)

    # Trap failures during ingest so the auto() run never calls the real
    # LLM -- but we still want the cost estimator to fire, which it does
    # in Step 6 regardless of ingest success.
    _sidecar, report = await auto(
        docs=_synthetic_docs(2),
        budget=1,
        extractor=extractor,
        store_path=tmp_path / "store.db",
        cache_path=tmp_path / "cache.db",
    )
    try:
        assert recorded_kwargs.get("model_id") == "gpt-4o-mini"
        # With pricing available and a real estimator run, the reported
        # cost is a float (non-negative, possibly very small).
        assert isinstance(report.cost_est_usd, float)
        assert report.cost_est_usd >= 0.0
    finally:
        await _sidecar.store.backend.aclose()


def test_resolve_model_id_variants() -> None:
    """Direct coverage of the extractor -> model_id resolver."""
    from nuggetindex.auto import _resolve_model_id
    from nuggetindex.extractors.cache import CachedExtractor
    from nuggetindex.extractors.trigger import TriggerExtractor

    trigger = TriggerExtractor()
    assert _resolve_model_id(trigger) == "trigger"

    # Wrapping a TriggerExtractor in a cache must not change the id.
    cached_trigger = CachedExtractor(inner=trigger, cache_path=":memory:")
    assert _resolve_model_id(cached_trigger) == "trigger"

    # An object with ``.cfg.model`` (LLMExtractor-shaped) -> the model id.
    class FakeCfg:
        provider = "openai"
        model = "gpt-4o"

    class FakeLLM:
        cfg = FakeCfg()

    assert _resolve_model_id(FakeLLM()) == "gpt-4o"

    # Unknown shape -> conservative default.
    class NoCfg:
        pass

    assert _resolve_model_id(NoCfg()) == "gpt-4o-mini"
