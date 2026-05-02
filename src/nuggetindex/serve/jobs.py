"""In-process asyncio job queue for long-running auto() runs.

Single-tenant, bounded concurrency. Job state machine:
    pending -> running -> {succeeded, failed}

For multi-tenant or multi-worker deployments, replace this module with an
external task queue (Celery, RQ, Arq, etc.). The Protocol at the bottom of
this module is what ``app.py`` depends on.
"""
from __future__ import annotations

import asyncio
import contextlib
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_JobFn = Callable[[dict[str, Any], "Job"], Awaitable[Any]]


@dataclass
class Job:
    id: str
    state: str = "pending"  # pending | running | succeeded | failed
    progress: float = 0.0
    stage: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def update(self, *, stage: str | None = None, progress: float | None = None) -> None:
        if stage is not None:
            self.stage = stage
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))


class JobRegistry:
    """Asyncio-backed job registry with bounded concurrency.

    Jobs are kept in memory; they do not survive process restart. The semaphore
    bounds concurrent active jobs to ``max_concurrency`` (default 4).
    """

    def __init__(self, *, max_concurrency: int = 4):
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._sem = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    async def submit(self, fn: _JobFn, params: dict[str, Any]) -> Job:
        job = Job(id=str(uuid.uuid4()), params=params)
        async with self._lock:
            self._jobs[job.id] = job
        task = asyncio.create_task(self._run(fn, job))
        self._tasks[job.id] = task
        return job

    async def _run(self, fn: _JobFn, job: Job) -> None:
        async with self._sem:
            job.state = "running"
            job.started_at = datetime.now(tz=UTC)
            try:
                result = await fn(job.params, job)
                job.result = result if isinstance(result, dict) else {"value": result}
                job.state = "succeeded"
                job.progress = 1.0
            except Exception as exc:  # noqa: BLE001 -- surface any failure to the job record
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            finally:
                job.finished_at = datetime.now(tz=UTC)

    async def shutdown(self, *, timeout: float = 10.0) -> None:
        """Cancel all in-flight tasks; used during FastAPI lifespan teardown."""
        for t in self._tasks.values():
            if not t.done():
                t.cancel()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=timeout,
            )
