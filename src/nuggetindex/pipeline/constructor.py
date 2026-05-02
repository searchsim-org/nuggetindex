"""Pipeline orchestrator: ``DocumentConstructor`` wires all four stages.

Responsibility: given a ``Document``, run extraction (optionally gated by
``QualityGate``), canonicalize, temporal inference, dedup, and conflict
resolution. Returns the list of nuggets ready to persist.

Cross-document conflict detection inside a single ``aingest`` call is handled
by the optional ``fetch_existing_by_key`` callback, which pulls prior-ingested
peers from the backend for each extracted nugget's key.
"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from nuggetindex.core.models import EpistemicState, FactTriple, Nugget
from nuggetindex.core.schema import RelationSchema
from nuggetindex.extractors.base import BaseExtractor, accepts_source_id
from nuggetindex.extractors.quality import QualityGate
from nuggetindex.pipeline.aliases import AliasResolver
from nuggetindex.pipeline.canonicalize import canonicalize
from nuggetindex.pipeline.conflict import ConflictDetector
from nuggetindex.pipeline.dedup import Deduplicator
from nuggetindex.pipeline.entity_types import check_triple_direction
from nuggetindex.pipeline.object_validator import is_valid_object
from nuggetindex.pipeline.temporal import infer_validity


@dataclass
class Document:
    """Input for ``NuggetStore.aingest`` / ``DocumentConstructor.aprocess``."""

    source_id: str
    text: str
    uri: str | None = None
    source_date: datetime | None = None


FetchExistingByKey = Callable[[tuple[str, str, str]], Awaitable[list[Nugget]]]


class DocumentConstructor:
    """Orchestrates extract -> canonicalize -> temporal -> dedup -> conflict."""

    def __init__(
        self,
        *,
        extractor: BaseExtractor,
        schema: RelationSchema,
        deduplicator: Deduplicator,
        conflict_detector: ConflictDetector,
        quality_gate: QualityGate | None = None,
    ) -> None:
        self.extractor = extractor
        self.schema = schema
        self.deduplicator = deduplicator
        self.conflict_detector = conflict_detector
        self.quality_gate = quality_gate
        # Cached at init so we don't re-inspect the extractor signature on
        # every ``aprocess`` call. If the quality gate wraps a different
        # extractor, the gate itself forwards the kwarg into its inner
        # extractor, so inspecting ``self.extractor`` is sufficient here.
        self._extractor_accepts_source_id = accepts_source_id(extractor)
        # Placeholder-validity extractors (test fixtures today) need a real
        # ``source_date`` from the caller — otherwise we silently fall back
        # to ``datetime.now(UTC)``, which breaks idempotent re-ingest.
        # Extractors opt in via an ``emits_placeholder_validity`` attribute
        # on the class / instance; production extractors (LLMExtractor) emit
        # real intervals and leave the attribute unset.
        inner = (
            self.quality_gate.extractor
            if self.quality_gate is not None and hasattr(self.quality_gate, "extractor")
            else extractor
        )
        self._extractor_emits_placeholder = bool(
            getattr(inner, "emits_placeholder_validity", False)
        )

    async def aprocess(
        self,
        doc: Document,
        *,
        existing: list[Nugget] | None = None,
        fetch_existing_by_key: FetchExistingByKey | None = None,
        alias_resolver: AliasResolver | None = None,
    ) -> list[Nugget]:
        """Run all four stages on ``doc`` and return the resulting nuggets.

        Duplicate nuggets (same key + similar-enough object) are dropped from
        the output. Surviving nuggets have their lifecycle status adjusted by
        conflict resolution. Updated peers from conflict resolution are NOT
        returned here (the store-layer caller persists them via its own
        handle to the backend).

        When ``alias_resolver`` is supplied, it is reused across the call so
        the canonical pool it maintains can span multiple documents (see
        fix 10, ``NuggetStore._ensure_alias_resolver``). When omitted, a
        fresh per-call resolver is instantiated so direct callers (tests,
        bespoke pipelines) keep working unchanged.
        """
        existing = list(existing or [])
        # Alias resolver: prefer a caller-supplied instance (store-scoped,
        # fix 10) so the canonical pool accumulates across docs. When the
        # caller doesn't pass one, fall back to a per-call instance so
        # existing direct ``DocumentConstructor`` users keep the historical
        # behaviour. Either way, seed it with any ``existing`` peers so the
        # current doc's surface forms can collapse against them.
        if alias_resolver is None:
            alias_resolver = AliasResolver()
        for peer in existing:
            alias_resolver.resolve(peer.fact.subject)
            alias_resolver.resolve(peer.fact.object)

        if doc.source_date is None and self._extractor_emits_placeholder:
            warnings.warn(
                "Document.source_date is None and the configured extractor "
                "emits placeholder validity. The pipeline will fall back to "
                "datetime.now(UTC), which breaks idempotent re-ingest. "
                "Passing an explicit source_date will be required in "
                "nuggetindex 0.3.",
                UserWarning,
                stacklevel=2,
            )
        source_date = doc.source_date or datetime.now(UTC)

        # 1. Extract (with optional quality gate). Forward the document's
        # source_id only when the extractor's signature declares it (see
        # ``accepts_source_id``), so legacy subclasses keep working unchanged.
        extract_kwargs: dict[str, str] = {}
        if self._extractor_accepts_source_id:
            extract_kwargs["source_id"] = doc.source_id
        if self.quality_gate is not None:
            gate = await self.quality_gate.aextract(doc.text, **extract_kwargs)
            raw_results = gate.accepted
        else:
            raw_results = await self.extractor.aextract(doc.text, **extract_kwargs)

        ready: list[Nugget] = []
        rejected_reasons: list[str] = []
        # Counters for the entity-type validation stage (Fix C). Summarised
        # in a single UserWarning at the end of the doc, mirroring the
        # existing object_validator warning format.
        entity_flipped: list[str] = []
        entity_rejected: list[str] = []
        for r in raw_results:
            n = r.nugget

            # 2. Canonicalize subject + predicate.
            n = canonicalize(n, self.schema)

            # 3. Temporal inference. `infer_validity` returns an `InferredValidity`
            # wrapper when we ask for confidence — we use that to apply the
            # spec §5.4 0.75 multiplier on ambiguous cues.
            #
            # Rule-based extractors emit ``ValidityInterval.unknown()`` so the
            # pipeline can fill in a real interval here (preserving idempotent
            # re-ingest). LLM extractors may have already emitted a concrete
            # interval via structured output; in that case we pass it as a
            # ``prior`` so a weaker rule-based inference doesn't clobber it.
            prior = None if n.validity.is_placeholder() else n.validity
            inferred = infer_validity(
                n.fact.text,
                source_date=source_date,
                return_confidence=True,
                prior=prior,
            )
            new_epistemic = n.epistemic
            if inferred.confidence != 1.0:
                new_epistemic = EpistemicState(
                    status=n.epistemic.status,
                    rank=n.epistemic.rank,
                    confidence=n.epistemic.confidence * inferred.confidence,
                )
            # Rebuild via Nugget.new so the content-hashed ID picks up the
            # (possibly updated) validity.start.
            n = Nugget.new(
                kind=n.kind,
                fact=n.fact,
                validity=inferred.interval,
                epistemic=new_epistemic,
                provenance=n.provenance,
                parent_id=n.parent_id,
                extraction_confidence=n.extraction_confidence,
                created_at=n.created_at,
                updated_at=n.updated_at,
            )

            # 3b. Alias resolution (Fix A). Canonicalize subject + object
            # surface forms against the running per-ingest pool so
            # "SpaceX" / "Space X" / "Space-X" collapse to a single
            # canonical before downstream conflict detection sees them.
            # No hardcoded alias tables; the resolver learns the pool as
            # it goes. Persisting the pool is out of scope for this task.
            subj_resolution = alias_resolver.resolve(n.fact.subject)
            obj_resolution = alias_resolver.resolve(n.fact.object)
            if (
                subj_resolution.canonical != n.fact.subject
                or obj_resolution.canonical != n.fact.object
            ):
                new_fact = FactTriple(
                    subject=subj_resolution.canonical or n.fact.subject,
                    predicate=n.fact.predicate,
                    object=obj_resolution.canonical or n.fact.object,
                    text=n.fact.text,
                    subject_type=n.fact.subject_type,
                    object_type=n.fact.object_type,
                )
                n = Nugget.new(
                    kind=n.kind,
                    fact=new_fact,
                    validity=n.validity,
                    epistemic=n.epistemic,
                    provenance=n.provenance,
                    parent_id=n.parent_id,
                    extraction_confidence=n.extraction_confidence,
                    created_at=n.created_at,
                    updated_at=n.updated_at,
                )

            # 3c. Entity-type validation + direction flip (Fix C / fix 9).
            # Prefer the LLM-emitted ``subject_type`` / ``object_type``
            # carried on the FactTriple (full-sentence context, works
            # cross-lingually). When those are absent (legacy nuggets,
            # rule-based extractor) the check falls back to a spaCy NER
            # probe on the raw mentions. When the predicate has no
            # expected types, this step is a no-op.
            direction = check_triple_direction(
                n.fact.subject,
                n.fact.predicate,
                n.fact.object,
                self.schema,
                subject_type=n.fact.subject_type,
                object_type=n.fact.object_type,
            )
            if direction == "flip":
                # Swap subject/object AND their types so downstream
                # re-validation sees a consistent triple.
                flipped_fact = FactTriple(
                    subject=n.fact.object,
                    predicate=n.fact.predicate,
                    object=n.fact.subject,
                    text=n.fact.text,
                    subject_type=n.fact.object_type,
                    object_type=n.fact.subject_type,
                )
                n = Nugget.new(
                    kind=n.kind,
                    fact=flipped_fact,
                    validity=n.validity,
                    epistemic=n.epistemic,
                    provenance=n.provenance,
                    parent_id=n.parent_id,
                    extraction_confidence=n.extraction_confidence,
                    created_at=n.created_at,
                    updated_at=n.updated_at,
                )
                entity_flipped.append(n.id)
            elif direction == "reject":
                entity_rejected.append(n.id)
                continue

            # 3d. Object validator. Drop obviously-malformed object tokens
            # (bare years, interrogative titles, punctuation-only strings)
            # BEFORE dedup + conflict so they can't drive phantom CONTESTED
            # flags from LLM extractor noise. Language-agnostic, deterministic.
            ok, reason = is_valid_object(n.fact.object)
            if not ok:
                rejected_reasons.append(reason)
                continue

            # 4. Build peers: previously-ingested (existing) + ones we just
            # processed this call + store-resident peers fetched by key.
            peers: list[Nugget] = list(existing) + list(ready)
            if fetch_existing_by_key is not None:
                fetched = await fetch_existing_by_key(n.key)
                peers = peers + fetched

            # 4a. Dedup.
            dup = await self.deduplicator.afind_duplicate(n, peers)
            if dup is not None:
                continue

            # 4b. Conflict resolution.
            resolution = await self.conflict_detector.aresolve(n, peers)
            # Apply any updates to peers we've already queued in `ready`.
            if resolution.updated_existing:
                ready_by_id = {e.id: i for i, e in enumerate(ready)}
                for updated_peer in resolution.updated_existing:
                    if updated_peer.id in ready_by_id:
                        ready[ready_by_id[updated_peer.id]] = updated_peer
                    else:
                        # Peer came from store (via fetch_existing_by_key) — the
                        # store-layer caller is responsible for persisting it.
                        ready.append(updated_peer)

            ready.append(resolution.incoming)

        if rejected_reasons:
            reasons_summary = ", ".join(sorted(set(rejected_reasons)))
            total_raw = len(raw_results)
            warnings.warn(
                f"object_validator rejected {len(rejected_reasons)} of "
                f"{total_raw} nuggets from doc {doc.source_id}: "
                f"{reasons_summary}",
                UserWarning,
                stacklevel=2,
            )

        if entity_flipped or entity_rejected:
            total_raw = len(raw_results)
            warnings.warn(
                f"entity_type_validator flipped {len(entity_flipped)} and "
                f"rejected {len(entity_rejected)} of {total_raw} nuggets "
                f"from doc {doc.source_id}",
                UserWarning,
                stacklevel=2,
            )

        return ready
