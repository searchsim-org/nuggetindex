"""Pydantic models for the core nugget record.

Frozen everywhere that immutability matters. All datetimes are timezone-aware
(UTC). JSON round-trip is guaranteed for the full Nugget type.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind


class FactTriple(BaseModel, frozen=True):
    """A subject-predicate-object assertion with its original natural-language form.

    ``subject_type`` and ``object_type`` are optional NER-label strings
    (spaCy-compatible vocabulary: ``PERSON`` / ``ORG`` / ``GPE`` / ``LOC`` /
    ``PRODUCT`` / ``EVENT`` / ``WORK_OF_ART`` / ``DATE`` / ``QUANTITY`` /
    ``OTHER``). They default to ``None`` so legacy stores and rule-based
    extractors that do not emit types continue to work unchanged. The LLM
    extractor (fix 8) emits them inline with the triple because the LLM
    already sees full sentence context -- this is strictly more reliable
    than running spaCy NER on an isolated entity mention.

    Types are stored verbatim from the extractor; callers that compare them
    against a predicate's ``expected_*_types`` are expected to uppercase on
    the comparison side (see :func:`pipeline.entity_types.check_triple_direction`).
    """

    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    subject_type: str | None = None
    object_type: str | None = None


class ValidityInterval(BaseModel, frozen=True):
    """Temporal validity interval [start, end). end=None means open-ended.

    Datetimes MUST be timezone-aware. Naive datetimes are rejected by the
    validator below to avoid the classic UTC/local timezone bug.
    """

    start: datetime
    end: datetime | None = None
    scope: Literal["global", "user", "group"] = "global"
    source_type: str = "document"
    validity_known: bool = True

    @field_validator("start", "end")
    @classmethod
    def _check_tz(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _check_order(self) -> "ValidityInterval":
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be strictly after start")
        return self

    def contains(self, t: datetime) -> bool:
        if t.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        if self.end is None:
            return t >= self.start
        return self.start <= t < self.end

    def overlaps(self, other: "ValidityInterval") -> bool:
        a_end = self.end
        b_end = other.end
        # [self.start, a_end) overlaps [other.start, b_end) if neither is
        # fully before the other.
        if a_end is not None and a_end <= other.start:
            return False
        return not (b_end is not None and b_end <= self.start)

    @classmethod
    def unknown(cls) -> "ValidityInterval":
        """Placeholder validity for extractors that don't infer temporal
        expressions. The pipeline's temporal inference stage replaces
        placeholder validity with either the parsed expression or the
        source document date. Nuggets with placeholder validity should
        never reach the store.
        """
        return cls(
            start=datetime(1, 1, 1, tzinfo=UTC),
            end=None,
            scope="global",
            source_type="placeholder",
        )

    def is_placeholder(self) -> bool:
        return self.source_type == "placeholder"


class EpistemicState(BaseModel):
    """Governance state for a nugget: lifecycle, rank, confidence.

    Not frozen - status is updated in place by the conflict detector as new
    evidence arrives.
    """

    status: LifecycleStatus = LifecycleStatus.ACTIVE
    rank: EpistemicRank = EpistemicRank.NORMAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ProvenanceRecord(BaseModel, frozen=True):
    """A pointer from a nugget back to the evidence in a source document."""

    source_id: str = Field(..., min_length=1)
    evidence_span: str
    char_start: int = 0
    char_end: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Nugget(BaseModel, frozen=True):
    """The core governed nugget record.

    IDs are content-hashed over (subject, predicate, object, validity_start, scope).
    This makes ingestion idempotent: the same fact from two documents hashes
    to the same ID, so re-ingesting merges provenance rather than duplicating.
    """

    id: str
    kind: NuggetKind
    fact: FactTriple
    validity: ValidityInterval
    epistemic: EpistemicState
    provenance: tuple[ProvenanceRecord, ...]
    parent_id: str | None = None
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v

    @classmethod
    def new(
        cls,
        *,
        kind: NuggetKind,
        fact: FactTriple,
        validity: ValidityInterval,
        epistemic: EpistemicState,
        provenance: tuple[ProvenanceRecord, ...],
        parent_id: str | None = None,
        extraction_confidence: float = 1.0,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> "Nugget":
        """Construct a Nugget and derive its content-hashed ID."""
        from nuggetindex.utils.hashing import stable_short_hash

        content = (
            f"{fact.subject}|{fact.predicate}|{fact.object}|"
            f"{validity.start.isoformat()}|{validity.scope}"
        )
        nid = stable_short_hash(content)
        now = datetime.now(UTC)
        return cls(
            id=nid,
            kind=kind,
            fact=fact,
            validity=validity,
            epistemic=epistemic,
            provenance=provenance,
            parent_id=parent_id,
            extraction_confidence=extraction_confidence,
            created_at=created_at or now,
            updated_at=updated_at or now,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def key(self) -> tuple[str, str, str]:
        """Conflict-detection key: (subject, predicate, scope)."""
        return (self.fact.subject, self.fact.predicate, self.validity.scope)

    def is_retrievable_at(self, t: datetime) -> bool:
        """True iff validity contains t AND status is ACTIVE or CONTESTED."""
        if not self.validity.contains(t):
            return False
        return self.epistemic.status in (LifecycleStatus.ACTIVE, LifecycleStatus.CONTESTED)
