"""Pydantic request/response models for the nuggetindex HTTP API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    query_time: datetime | None = None
    top_k: int = Field(default=10, ge=1, le=100)


class NuggetPayload(BaseModel):
    subject: str
    predicate: str
    object: str
    validity_start: datetime | None
    validity_end: datetime | None
    status: str
    source_id: str


class QueryResponse(BaseModel):
    query: str
    use_nugget: bool
    reason: str
    context_block: str
    nuggets: list[NuggetPayload]


class IngestRequest(BaseModel):
    source_id: str
    text: str
    uri: str | None = None
    source_date: datetime | None = None


class IngestResponse(BaseModel):
    document_id: str
    nuggets_added: int
    conflicts_detected: int


class DoctorRequest(BaseModel):
    sample_size: int = Field(default=500, ge=1, le=10_000)
    mode: str = "fast"


class DoctorScoreOut(BaseModel):
    dimension: str
    percentage: float
    ci95_low: float
    ci95_high: float
    examples: list[str]


class DoctorResponse(BaseModel):
    sample_mode: str
    verdict: str
    scores: list[DoctorScoreOut]
    rendered_markdown: str


class StatsResponse(BaseModel):
    total_nuggets: int
    contested_nuggets: int
    active_nuggets: int
    deprecated_nuggets: int
    distinct_subjects: int
    distinct_predicates: int
    rename_edges: int


class AutoJobRequest(BaseModel):
    """Kicks off a long-running auto() run. Phase 2.7 implements the job queue."""

    corpus_type: str
    corpus_url: str | None = None
    corpus_name: str | None = None
    api_key: str | None = None
    budget: int = Field(default=50, ge=1, le=500)
    sample_size: int = Field(default=200, ge=1, le=10_000)
    mode: str = "offline-curated"
    extractor: str = "gpt-4o-mini"


class AutoJobStatus(BaseModel):
    job_id: str
    state: str
    progress: float
    stage: str
    result: dict[str, Any] | None = None
