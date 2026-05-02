"""Dependency-injection helpers for the serve app.

FastAPI's Depends() wires these into per-request handlers. The app's lifespan
context calls ``configure(...)`` once at startup to populate ``app_state``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nuggetindex.serve.jobs import JobRegistry


@dataclass
class _AppState:
    store: Any | None = None
    sidecar: Any | None = None
    jobs: JobRegistry | None = None


app_state: _AppState = _AppState()


def configure(
    *,
    db_path: str | Path,
    mode: str = "offline-curated",
    extractor: Any | None = None,
    fallback_corpus: Any | None = None,
    freshness_threshold: Any | None = None,
) -> None:
    from nuggetindex import NuggetStore
    from nuggetindex.sidecar import Sidecar

    # Default to the LLM-free TriggerExtractor so /v1/ingest works out of the
    # box without the caller having to wire an LLM. Callers who want LLM-grade
    # extraction pass extractor=LLMExtractor(...) explicitly.
    if extractor is None:
        from nuggetindex.extractors.trigger import TriggerExtractor

        extractor = TriggerExtractor()

    store = NuggetStore(db_path=Path(db_path), extractor=extractor)
    kwargs: dict[str, Any] = {}
    if fallback_corpus is not None:
        kwargs["fallback_corpus"] = fallback_corpus
    if freshness_threshold is not None:
        kwargs["freshness_threshold"] = freshness_threshold
    sidecar = Sidecar(store=store, mode=mode, extractor=extractor, **kwargs)
    app_state.store = store
    app_state.sidecar = sidecar
    app_state.jobs = JobRegistry()


def reset() -> None:
    """Test-only helper to clear app_state between tests."""
    app_state.store = None
    app_state.sidecar = None
    app_state.jobs = None


def get_store():
    if app_state.store is None:
        raise RuntimeError("serve.deps.configure(...) was not called")
    return app_state.store


def get_sidecar():
    if app_state.sidecar is None:
        raise RuntimeError("serve.deps.configure(...) was not called")
    return app_state.sidecar


def get_jobs():
    if app_state.jobs is None:
        raise RuntimeError("serve.deps.configure(...) was not called")
    return app_state.jobs
