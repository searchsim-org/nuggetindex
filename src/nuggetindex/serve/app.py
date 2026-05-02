"""FastAPI HTTP API for nuggetindex.

Endpoints (all under `/v1/` except health):

    GET  /healthz             -> {"status": "ok"}
    POST /v1/query            -> run Sidecar.ahandle() and return context + nuggets
    POST /v1/ingest           -> run NuggetStore.aingest() on a single document
    POST /v1/doctor           -> run audit.doctor.scan_index against the store
    GET  /v1/stats            -> lightweight SQL-driven store stats
    POST /v1/auto             -> submit a long-running auto() job (in-process queue)
    GET  /v1/auto/{job_id}    -> poll job status

This module is intentionally thin: each handler delegates to the existing
library APIs and shapes the result through the Pydantic schemas in
nuggetindex.serve.schemas.
"""
from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI

from nuggetindex.serve import deps
from nuggetindex.serve.schemas import (
    AutoJobRequest,
    AutoJobStatus,
    DoctorRequest,
    DoctorResponse,
    DoctorScoreOut,
    IngestRequest,
    IngestResponse,
    NuggetPayload,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)


def create_app(
    *,
    db_path: str | Path,
    mode: str = "offline-curated",
    extractor: Any | None = None,
    fallback_corpus: Any | None = None,
    freshness_threshold: Any | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        deps.configure(
            db_path=db_path, mode=mode, extractor=extractor,
            fallback_corpus=fallback_corpus,
            freshness_threshold=freshness_threshold,
        )
        yield
        deps.reset()

    app = FastAPI(title="nuggetindex", version="0.4.1", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(req: QueryRequest, sidecar=Depends(deps.get_sidecar)) -> QueryResponse:
        response = await sidecar.ahandle(
            req.query, query_time=req.query_time, top_k=req.top_k,
        )
        return QueryResponse(
            query=req.query,
            use_nugget=bool(response.decision.use_nugget) if response.decision else False,
            reason=(response.decision.reason or "") if response.decision else "",
            context_block=response.context_block,
            nuggets=[_payload(n) for n in response.nuggets],
        )

    @app.post("/v1/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest, store=Depends(deps.get_store)) -> IngestResponse:
        from nuggetindex.pipeline.constructor import Document

        doc = Document(
            source_id=req.source_id, text=req.text,
            uri=req.uri, source_date=req.source_date,
        )
        result = await store.aingest(doc)
        return IngestResponse(
            document_id=result.document_id,
            nuggets_added=result.nuggets_added,
            conflicts_detected=result.conflicts_detected,
        )

    @app.post("/v1/doctor", response_model=DoctorResponse)
    async def doctor(req: DoctorRequest, store=Depends(deps.get_store)) -> DoctorResponse:
        from nuggetindex.audit import scan_index

        docs = _reconstruct_docs(store)
        report = await scan_index(docs=docs, mode=req.mode, sample_size=req.sample_size)
        return DoctorResponse(
            sample_mode=report.sample_mode,
            verdict=report.verdict,
            scores=[
                DoctorScoreOut(
                    dimension=s.dimension,
                    percentage=s.percentage,
                    ci95_low=s.ci95[0],
                    ci95_high=s.ci95[1],
                    examples=list(s.examples),
                )
                for s in report.scores
            ],
            rendered_markdown=report.rendered_markdown,
        )

    @app.get("/v1/stats", response_model=StatsResponse)
    async def stats(store=Depends(deps.get_store)) -> StatsResponse:
        return _compute_stats(store)

    @app.post("/v1/auto", response_model=AutoJobStatus)
    async def auto_submit(
        req: AutoJobRequest, jobs=Depends(deps.get_jobs),
    ) -> AutoJobStatus:
        async def run(params: dict, job) -> dict[str, Any]:
            job.update(stage="received", progress=0.05)
            # Phase 2.7 intentionally leaves full auto() wiring as a TODO --
            # the interface is proven; real execution waits for Phase 3's
            # hosted demo to pick the corpus adapter at runtime. For now,
            # return a synthetic report that documents the contract.
            job.update(stage="planning", progress=0.1)
            await asyncio.sleep(0)  # yield so the test sees state transitions
            job.update(stage="ingesting", progress=0.5)
            await asyncio.sleep(0)
            job.update(stage="finalizing", progress=0.9)
            return {
                "params": params,
                "note": (
                    "nuggetindex.serve v0.4.1: /v1/auto accepts submissions "
                    "and tracks progress. Corpus-adapter dispatch lands in "
                    "Phase 3 alongside the hosted demo UI."
                ),
            }

        job = await jobs.submit(run, req.model_dump())
        return _job_status(job)

    @app.get("/v1/auto/{job_id}", response_model=AutoJobStatus)
    async def auto_status(job_id: str, jobs=Depends(deps.get_jobs)) -> AutoJobStatus:
        job = jobs.get(job_id)
        if job is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        return _job_status(job)

    return app


def _job_status(job: Any) -> AutoJobStatus:
    return AutoJobStatus(
        job_id=job.id,
        state=job.state,
        progress=job.progress,
        stage=job.stage,
        result=job.result,
    )


def _payload(nugget: Any) -> NuggetPayload:
    return NuggetPayload(
        subject=nugget.fact.subject,
        predicate=nugget.fact.predicate,
        object=nugget.fact.object,
        validity_start=nugget.validity.start if nugget.validity else None,
        validity_end=nugget.validity.end if nugget.validity else None,
        status=nugget.epistemic.status.value if nugget.epistemic else "?",
        source_id=nugget.provenance[0].source_id if nugget.provenance else "?",
    )


def _reconstruct_docs(store: Any) -> list[Any]:
    from nuggetindex.pipeline.constructor import Document

    conn = sqlite3.connect(str(store.db_path))
    try:
        rows = conn.execute(
            "SELECT json_extract(data, '$.provenance[0].source_id'), "
            "       json_extract(data, '$.provenance[0].evidence_span'), "
            "       json_extract(data, '$.provenance[0].created_at') "
            "FROM nuggets"
        ).fetchall()
    finally:
        conn.close()
    seen: set[str] = set()
    docs: list[Document] = []
    for src_id, evidence, created in rows:
        if not src_id or not evidence or src_id in seen:
            continue
        seen.add(src_id)
        parsed_date: datetime | None = None
        if created:
            try:
                raw = str(created).replace("Z", "+00:00")
                parsed_date = datetime.fromisoformat(raw)
            except ValueError:
                parsed_date = None
        docs.append(Document(
            source_id=src_id, text=evidence, uri=None, source_date=parsed_date,
        ))
    return docs


def _compute_stats(store: Any) -> StatsResponse:
    from nuggetindex.core.schema import RelationSchema

    conn = sqlite3.connect(str(store.db_path))
    try:
        row = conn.execute(
            """
            SELECT
              COUNT(*),
              SUM(CASE WHEN status='contested'  THEN 1 ELSE 0 END),
              SUM(CASE WHEN status='active'     THEN 1 ELSE 0 END),
              SUM(CASE WHEN status='deprecated' THEN 1 ELSE 0 END),
              COUNT(DISTINCT subject),
              COUNT(DISTINCT predicate)
            FROM nuggets
            """
        ).fetchone()
        rename_preds = RelationSchema.default().renaming_predicates
        rename_edges = 0
        if rename_preds:
            placeholders = ",".join("?" for _ in rename_preds)
            q = f"SELECT COUNT(*) FROM nuggets WHERE predicate IN ({placeholders})"
            rename_edges = conn.execute(q, list(rename_preds)).fetchone()[0] or 0
    finally:
        conn.close()
    return StatsResponse(
        total_nuggets=row[0] or 0,
        contested_nuggets=row[1] or 0,
        active_nuggets=row[2] or 0,
        deprecated_nuggets=row[3] or 0,
        distinct_subjects=row[4] or 0,
        distinct_predicates=row[5] or 0,
        rename_edges=rename_edges,
    )
