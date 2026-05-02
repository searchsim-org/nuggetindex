"""Index-agnostic audit scanner: estimate how much of a RAG index is at risk.

The doctor module is a complementary entry point to the zero-index
:func:`nuggetindex.audit.api.audit` function. Where ``audit()`` operates on an
*explicit passage list* tied to a single query -- extracting nuggets and
reporting in-sample conflicts and stale candidates -- :func:`scan_index`
operates on *any existing RAG index* (streamed or materialised) and returns
population-level estimates of four risk dimensions:

* ``temporal_depth``    -- share of documents carrying explicit source dates
* ``temporal_drift``    -- share of documents whose facts likely changed since
  their source date
* ``conflict_surface``  -- share of documents that disagree with a near
  neighbour on the same key
* ``rename_events``     -- share of documents referencing entity renames
  (acquisitions, rebrands, marriage-name changes, ...)

Two sampling modes are defined:

* ``mode="fast"`` -- cheap heuristics only: TIMEX parsing, lightweight NER,
  and trigger-verb lexica. No LLM calls. Target: low-minute runtime on a
  1M-doc index.

* ``mode="deep"`` -- full extractor over a stratified sample. Higher signal,
  higher cost. Routes through the same ``BaseExtractor`` contract as
  :func:`nuggetindex.audit.api.audit`. (Implementation lands in Task 2.5.)
"""

from __future__ import annotations

import re
import warnings
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Any, Literal

from nuggetindex.audit.heuristics import (
    TimeExpression,
    TriggerMatch,
    scan_triggers,
    stratified_sample,
    tag_timex,
    timex_available,
)
from nuggetindex.pipeline.constructor import Document


@dataclass(frozen=True)
class DoctorScore:
    """One risk dimension's estimate over the sampled population.

    ``percentage`` is expressed in 0.0-100.0 (not 0-1) so the rendered
    Markdown report can print raw values without rescaling. ``ci95`` carries
    the Wilson-score 95% confidence bounds in the same units. ``n_total`` is
    ``None`` when the input stream is unbounded / unknown.
    """

    dimension: Literal["temporal_depth", "temporal_drift", "conflict_surface", "rename_events"]
    percentage: float
    ci95: tuple[float, float]
    n_sampled: int
    n_total: int | None
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate output of :func:`scan_index`.

    The report owns its own rendered Markdown via ``rendered_markdown``; the
    :meth:`to_markdown` accessor exists for parity with
    :meth:`nuggetindex.audit.api.AuditReport.to_markdown`.
    """

    sample_mode: Literal["fast", "deep"]
    scores: list[DoctorScore]
    verdict: Literal["high", "medium", "low"]
    rendered_markdown: str

    def to_markdown(self) -> str:
        """Return the pre-built Markdown rendering of this report."""
        return self.rendered_markdown


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #


_EOS_RE = re.compile(r"[.!?](?:\s|$)")
_EXAMPLE_TRUNCATE = 80
_DIMENSIONS: tuple[
    Literal["temporal_depth", "temporal_drift", "conflict_surface", "rename_events"], ...
] = ("temporal_depth", "temporal_drift", "conflict_surface", "rename_events")


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score 95% CI on a binomial proportion, returned in 0..100."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    radius = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    low = max(0.0, (center - radius) * 100)
    high = min(100.0, (center + radius) * 100)
    return (low, high)


def _truncate(s: str, limit: int = _EXAMPLE_TRUNCATE) -> str:
    """Truncate a single-line example to ``limit`` chars with an ellipsis marker."""
    s = " ".join(s.split())  # collapse whitespace so the table stays tidy
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "\u2026"


def _default_schema() -> Any:
    """Lazy-import of :class:`RelationSchema` to keep module import light."""
    from nuggetindex.core.schema import RelationSchema

    return RelationSchema.default()


def _object_text(text: str, match: TriggerMatch) -> str:
    """Return the object text for ``match``, widened to the enclosing sentence.

    The regex-based object span in :mod:`triggers` is intentionally narrow
    (non-greedy capture). Distinct-object accounting for conflict and drift
    scoring benefits from a slightly richer string, so we extend from the
    object span's start up to the next sentence terminator (``.``, ``!``,
    ``?`` followed by whitespace or EOS). Empty spans fall through to the
    empty string so they do not contribute to "distinct object" sets.
    """
    start, end_span = match.object_span
    if (start, end_span) == (0, 0):
        return ""
    eos = _EOS_RE.search(text, start)
    end = eos.start() if eos is not None else len(text)
    return text[start:end].strip()


def _subject_text(text: str, match: TriggerMatch) -> str:
    """Raw subject slice, whitespace-stripped."""
    return text[match.subject_span[0] : match.subject_span[1]].strip()


def _canonicalize_predicate(predicate: str, schema: Any) -> str:
    """Apply ``schema.canonicalize`` defensively; fall back to raw on failure."""
    if schema is None:
        return predicate
    try:
        return str(schema.canonicalize(predicate))
    except Exception:  # pragma: no cover -- defensive
        return predicate


def _is_functional(predicate: str, schema: Any) -> bool:
    """``schema.is_functional`` with the conservative unknown-predicate default."""
    if schema is None:
        return True
    try:
        return bool(schema.is_functional(predicate))
    except Exception:  # pragma: no cover -- defensive
        return True


# --------------------------------------------------------------------------- #
# Fast-mode implementation
# --------------------------------------------------------------------------- #


async def _fast_scan(
    *,
    docs: AsyncIterable[Document] | Iterable[Document],
    sample_size: int,
    stratify_by: Literal["source_date", "none", "domain", "language", "composite"],
    schema: Any | None,
    rng_seed: int,
    dedup_near_duplicates: bool,
) -> DoctorReport:
    """Run the fast-mode pipeline and assemble a :class:`DoctorReport`."""
    sampled_docs, n_total = await stratified_sample(
        docs,
        sample_size=sample_size,
        stratify_by=stratify_by,
        rng_seed=rng_seed,
        dedup_near_duplicates=dedup_near_duplicates,
    )

    if not sampled_docs:
        return _empty_report(n_total=n_total)

    effective_schema = schema if schema is not None else _default_schema()
    have_timex = timex_available()

    # Per-doc heuristic outputs ----------------------------------------------
    timex_per_doc: dict[str, list[TimeExpression]] = {}
    triggers_per_doc: dict[str, list[TriggerMatch]] = {}
    for doc in sampled_docs:
        timex_per_doc[doc.source_id] = (
            tag_timex(doc.text, reference_date=doc.source_date) if have_timex else []
        )
        triggers_per_doc[doc.source_id] = scan_triggers(doc.text)

    # Cluster by (subject, canonical-predicate) ------------------------------
    cluster_objects: dict[tuple[str, str], list[str]] = {}
    cluster_timex_dates: dict[tuple[str, str], list[datetime]] = {}
    cluster_example_doc: dict[tuple[str, str], Document] = {}
    for doc in sampled_docs:
        for match in triggers_per_doc[doc.source_id]:
            subj = _subject_text(doc.text, match)
            pred = _canonicalize_predicate(match.predicate, effective_schema)
            key = (subj, pred)
            obj = _object_text(doc.text, match)
            if obj:
                cluster_objects.setdefault(key, []).append(obj)
            cluster_example_doc.setdefault(key, doc)
            # Collect parsed timex datetimes on docs that contributed to this key.
            for te in timex_per_doc[doc.source_id]:
                if te.parsed is not None:
                    cluster_timex_dates.setdefault(key, []).append(te.parsed)

    cluster_keys: list[tuple[str, str]] = list(cluster_objects.keys())
    distinct_objects: dict[tuple[str, str], set[str]] = {
        k: set(v) for k, v in cluster_objects.items()
    }

    # --- temporal_depth -----------------------------------------------------
    n_docs = len(sampled_docs)
    if have_timex:
        td_successes = sum(1 for d in sampled_docs if timex_per_doc[d.source_id])
        td_pct = td_successes / n_docs * 100.0 if n_docs else 0.0
        td_ci = _wilson_ci(td_successes, n_docs)
        td_examples_raw: list[str] = []
        for d in sampled_docs:
            for te in timex_per_doc[d.source_id]:
                td_examples_raw.append(te.span)
                if len(td_examples_raw) >= 3:
                    break
            if len(td_examples_raw) >= 3:
                break
        td_examples = [_truncate(x) for x in td_examples_raw]
    else:
        td_pct = 0.0
        td_ci = (0.0, 0.0)
        td_examples = ["(spaCy not installed)"]

    # --- temporal_drift -----------------------------------------------------
    drift_n = len(cluster_keys)
    if have_timex and drift_n:
        drift_scored: list[tuple[tuple[str, str], int, float]] = []
        drift_successes = 0
        for key in cluster_keys:
            objs = distinct_objects.get(key, set())
            dts = cluster_timex_dates.get(key, [])
            if len(objs) >= 2 and dts:
                span_days = (max(dts) - min(dts)).days
                if span_days > 365:
                    drift_successes += 1
                    drift_scored.append((key, len(objs), float(span_days)))
        drift_pct = drift_successes / drift_n * 100.0
        drift_ci = _wilson_ci(drift_successes, drift_n)
        drift_scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        drift_examples = [_truncate(f"{k[0]} / {k[1]}") for k, _cnt, _span in drift_scored[:3]]
    elif not have_timex:
        drift_pct = 0.0
        drift_ci = (0.0, 0.0)
        drift_examples = ["(spaCy not installed)"]
    else:
        drift_pct = 0.0
        drift_ci = (0.0, 0.0)
        drift_examples = []

    # --- conflict_surface ---------------------------------------------------
    functional_keys = [k for k in cluster_keys if _is_functional(k[1], effective_schema)]
    conflict_n = len(functional_keys)
    conflict_scored: list[tuple[tuple[str, str], list[str]]] = []
    conflict_successes = 0
    for key in functional_keys:
        objs = distinct_objects.get(key, set())
        if len(objs) >= 2:
            conflict_successes += 1
            # Keep insertion-order first-two for a stable example.
            seen: list[str] = []
            for o in cluster_objects.get(key, []):
                if o not in seen:
                    seen.append(o)
                if len(seen) >= 2:
                    break
            conflict_scored.append((key, seen))
    conflict_pct = conflict_successes / conflict_n * 100.0 if conflict_n else 0.0
    conflict_ci = _wilson_ci(conflict_successes, conflict_n)
    # Stable ordering: most distinct objects first, tie-break on key text.
    conflict_scored.sort(key=lambda t: (len(distinct_objects[t[0]]), t[0]), reverse=True)
    conflict_examples = [
        _truncate(f"{k[0]} / {k[1]} / {seen[0]} \u2194 {seen[1]}")
        for k, seen in conflict_scored[:3]
    ]

    # --- rename_events ------------------------------------------------------
    rename_successes = sum(
        1
        for d in sampled_docs
        if any(m.kind == "entity_rename" for m in triggers_per_doc[d.source_id])
    )
    rename_pct = rename_successes / n_docs * 100.0 if n_docs else 0.0
    rename_ci = _wilson_ci(rename_successes, n_docs)
    rename_examples_raw: list[str] = []
    for d in sampled_docs:
        for m in triggers_per_doc[d.source_id]:
            if m.kind == "entity_rename":
                rename_examples_raw.append(m.match_text)
                if len(rename_examples_raw) >= 3:
                    break
        if len(rename_examples_raw) >= 3:
            break
    rename_examples = [_truncate(x) for x in rename_examples_raw]

    # Assemble scores in the canonical dimension order -----------------------
    scores = [
        DoctorScore(
            dimension="temporal_depth",
            percentage=td_pct,
            ci95=td_ci,
            n_sampled=n_docs,
            n_total=n_total,
            examples=td_examples,
        ),
        DoctorScore(
            dimension="temporal_drift",
            percentage=drift_pct,
            ci95=drift_ci,
            n_sampled=n_docs,
            n_total=n_total,
            examples=drift_examples,
        ),
        DoctorScore(
            dimension="conflict_surface",
            percentage=conflict_pct,
            ci95=conflict_ci,
            n_sampled=n_docs,
            n_total=n_total,
            examples=conflict_examples,
        ),
        DoctorScore(
            dimension="rename_events",
            percentage=rename_pct,
            ci95=rename_ci,
            n_sampled=n_docs,
            n_total=n_total,
            examples=rename_examples,
        ),
    ]

    verdict = _verdict(
        drift_pct=drift_pct,
        conflict_pct=conflict_pct,
        all_pcts=[td_pct, drift_pct, conflict_pct, rename_pct],
    )

    rendered = _render_markdown_scorecard(
        scores=scores,
        verdict=verdict,
        n_sampled=n_docs,
        n_total=n_total,
        drift_pct=drift_pct,
        conflict_pct=conflict_pct,
        mode_label="fast mode",
    )

    return DoctorReport(
        sample_mode="fast",
        scores=scores,
        verdict=verdict,
        rendered_markdown=rendered,
    )


# --------------------------------------------------------------------------- #
# Deep-mode implementation
# --------------------------------------------------------------------------- #


_DEEP_FAILURE_THRESHOLD = 0.30


async def _deep_scan(
    *,
    docs: AsyncIterable[Document] | Iterable[Document],
    sample_size: int,
    stratify_by: Literal["source_date", "none", "domain", "language", "composite"],
    schema: Any | None,
    extractor: Any,
    rng_seed: int,
    dedup_near_duplicates: bool,
) -> DoctorReport:
    """Run the deep-mode pipeline and assemble a :class:`DoctorReport`.

    Deep mode routes the stratified sample through the full
    :meth:`NuggetStore.aingest` pipeline (extractor + canonicalize + temporal
    inference + dedup + conflict resolution) against a transient in-memory
    store. The four scores are then computed over the resulting nugget
    population. Ingestion is serial because the configured LLM extractor
    typically has its own concurrency / rate-limit story; the doctor's job is
    honest measurement, not throughput maximisation.
    """
    # Local imports to keep the module import cheap for fast-mode callers.
    import tempfile
    from pathlib import Path

    from nuggetindex.store.base import NuggetStore

    sampled_docs, n_total = await stratified_sample(
        docs,
        sample_size=sample_size,
        stratify_by=stratify_by,
        rng_seed=rng_seed,
        dedup_near_duplicates=dedup_near_duplicates,
    )

    if not sampled_docs:
        return _empty_report(n_total=n_total, sample_mode="deep")

    effective_schema = schema if schema is not None else _default_schema()

    # Transient store. We would prefer ``:memory:``, but the SQLite backend
    # opens *separate* connections for reads and writes, and in-memory SQLite
    # databases are per-connection — so the writer's freshly-initialised
    # schema is invisible to the read pool. Using a temp-directory DB sidesteps
    # that without leaking state: the tmpdir is removed in ``finally``.
    n_sampled = len(sampled_docs)
    n_failed = 0

    with tempfile.TemporaryDirectory(prefix="nuggetindex-doctor-") as tmpdir:
        store = NuggetStore(
            db_path=Path(tmpdir) / "doctor.db",
            schema=effective_schema,
            extractor=extractor,
        )
        try:
            # 1. Ingest each sampled doc serially. Extraction failures are
            # logged via ``warnings.warn`` and counted; we keep going so the
            # report reflects as much of the sample as possible.
            for doc in sampled_docs:
                try:
                    await store.aingest(doc)
                except Exception as exc:  # noqa: BLE001 -- diagnostic tool
                    n_failed += 1
                    warnings.warn(
                        f"deep-scan ingest failed for source_id="
                        f"{doc.source_id!r}: {type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            # 2. Score over the populated store.
            (
                td_pct,
                td_ci,
                td_examples,
                drift_pct,
                drift_ci,
                drift_examples,
                conflict_pct,
                conflict_ci,
                conflict_examples,
                rename_pct,
                rename_ci,
                rename_examples,
            ) = await _score_deep_store(
                store=store,
                sampled_docs=sampled_docs,
                schema=effective_schema,
            )
        finally:
            await store.aclose()

    scores = [
        DoctorScore(
            dimension="temporal_depth",
            percentage=td_pct,
            ci95=td_ci,
            n_sampled=n_sampled,
            n_total=n_total,
            examples=td_examples,
        ),
        DoctorScore(
            dimension="temporal_drift",
            percentage=drift_pct,
            ci95=drift_ci,
            n_sampled=n_sampled,
            n_total=n_total,
            examples=drift_examples,
        ),
        DoctorScore(
            dimension="conflict_surface",
            percentage=conflict_pct,
            ci95=conflict_ci,
            n_sampled=n_sampled,
            n_total=n_total,
            examples=conflict_examples,
        ),
        DoctorScore(
            dimension="rename_events",
            percentage=rename_pct,
            ci95=rename_ci,
            n_sampled=n_sampled,
            n_total=n_total,
            examples=rename_examples,
        ),
    ]

    verdict = _verdict(
        drift_pct=drift_pct,
        conflict_pct=conflict_pct,
        all_pcts=[td_pct, drift_pct, conflict_pct, rename_pct],
    )

    warning_line: str | None = None
    if n_sampled and n_failed / n_sampled > _DEEP_FAILURE_THRESHOLD:
        warning_line = f"> {n_failed} of {n_sampled} ingestions failed; scores may be unreliable."

    rendered = _render_markdown_scorecard(
        scores=scores,
        verdict=verdict,
        n_sampled=n_sampled,
        n_total=n_total,
        drift_pct=drift_pct,
        conflict_pct=conflict_pct,
        mode_label="deep mode",
        warning_line=warning_line,
    )

    return DoctorReport(
        sample_mode="deep",
        scores=scores,
        verdict=verdict,
        rendered_markdown=rendered,
    )


async def _score_deep_store(
    *,
    store: Any,
    sampled_docs: list[Document],
    schema: Any,
) -> tuple[
    float,
    tuple[float, float],
    list[str],
    float,
    tuple[float, float],
    list[str],
    float,
    tuple[float, float],
    list[str],
    float,
    tuple[float, float],
    list[str],
]:
    """Compute the four deep-mode scores against ``store``.

    Returns a flat 12-tuple (pct, ci, examples) * 4 in canonical dimension
    order (temporal_depth, temporal_drift, conflict_surface, rename_events).
    Aggregates the populated nuggets both per-doc (for temporal_depth and
    rename_events) and globally per ``(subject, predicate, scope)`` key (for
    temporal_drift and conflict_surface).
    """
    from nuggetindex.core.enums import LifecycleStatus
    from nuggetindex.core.models import Nugget

    n_docs = len(sampled_docs)
    rename_preds = schema.entity_rename_predicates if schema is not None else frozenset()

    # Per-doc nugget lookup (used by temporal_depth + rename_events).
    per_doc_nuggets: dict[str, list[Nugget]] = {}
    # Global key -> nuggets aggregation (used by drift + conflict). Dedup by
    # nugget.id: ``aingest`` is idempotent at the row level but the same row
    # can surface under multiple docs if their provenance sets overlap.
    by_key: dict[tuple[str, str, str], dict[str, Nugget]] = {}

    for doc in sampled_docs:
        rows = await store.backend.aget_nuggets_by_source(doc.source_id)
        per_doc_nuggets[doc.source_id] = rows
        for n in rows:
            by_key.setdefault(n.key, {})[n.id] = n

    # --- temporal_depth -----------------------------------------------------
    td_successes = 0
    td_examples_raw: list[str] = []
    for doc in sampled_docs:
        has_known = False
        for n in per_doc_nuggets[doc.source_id]:
            if n.validity.validity_known:
                has_known = True
                if len(td_examples_raw) < 3 and n.provenance:
                    td_examples_raw.append(n.provenance[0].evidence_span)
        if has_known:
            td_successes += 1
    td_pct = td_successes / n_docs * 100.0 if n_docs else 0.0
    td_ci = _wilson_ci(td_successes, n_docs)
    td_examples = [_truncate(x) for x in td_examples_raw[:3] if x]

    # --- temporal_drift -----------------------------------------------------
    drift_total = len(by_key)
    drift_successes = 0
    drift_scored: list[tuple[tuple[str, str, str], int, int]] = []
    for key, nuggets_by_id in by_key.items():
        nuggets = list(nuggets_by_id.values())
        known = [n for n in nuggets if n.validity.validity_known]
        distinct_objects = {n.fact.object for n in nuggets}
        if len(distinct_objects) < 2 or len(known) < 2:
            continue
        starts = [n.validity.start for n in known]
        span_days = (max(starts) - min(starts)).days
        if span_days > 365:
            drift_successes += 1
            drift_scored.append((key, len(distinct_objects), span_days))
    drift_pct = drift_successes / drift_total * 100.0 if drift_total else 0.0
    drift_ci = _wilson_ci(drift_successes, drift_total)
    drift_scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    drift_examples = [
        _truncate(f"{k[0]} / {k[1]} \u00b7 {cnt} objects / {span}d span")
        for k, cnt, span in drift_scored[:3]
    ]

    # --- conflict_surface ---------------------------------------------------
    functional_keys = [k for k in by_key if _is_functional(k[1], schema)]
    conflict_total = len(functional_keys)
    conflict_successes = 0
    conflict_scored: list[tuple[tuple[str, str, str], list[str]]] = []
    for key in functional_keys:
        nuggets = list(by_key[key].values())
        if not any(n.epistemic.status == LifecycleStatus.CONTESTED for n in nuggets):
            continue
        conflict_successes += 1
        seen: list[str] = []
        for n in nuggets:
            if n.epistemic.status != LifecycleStatus.CONTESTED:
                continue
            if n.fact.object not in seen:
                seen.append(n.fact.object)
            if len(seen) >= 2:
                break
        # Fall back to any distinct objects if fewer than two contested
        # variants were returned (defensive — keeps example rendering robust).
        if len(seen) < 2:
            for n in nuggets:
                if n.fact.object not in seen:
                    seen.append(n.fact.object)
                if len(seen) >= 2:
                    break
        conflict_scored.append((key, seen))
    conflict_pct = conflict_successes / conflict_total * 100.0 if conflict_total else 0.0
    conflict_ci = _wilson_ci(conflict_successes, conflict_total)
    conflict_scored.sort(key=lambda t: (len(by_key[t[0]]), t[0]), reverse=True)
    conflict_examples = []
    for key, seen in conflict_scored[:3]:
        if len(seen) >= 2:
            conflict_examples.append(
                _truncate(f"{key[0]} / {key[1]} \u00b7 {seen[0]} \u2194 {seen[1]}")
            )
        elif seen:
            conflict_examples.append(_truncate(f"{key[0]} / {key[1]} \u00b7 {seen[0]}"))
        else:
            conflict_examples.append(_truncate(f"{key[0]} / {key[1]}"))

    # --- rename_events ------------------------------------------------------
    rename_successes = 0
    rename_examples_raw: list[str] = []
    for doc in sampled_docs:
        doc_has_rename = False
        for n in per_doc_nuggets[doc.source_id]:
            if n.fact.predicate in rename_preds:
                doc_has_rename = True
                if len(rename_examples_raw) < 3:
                    rename_examples_raw.append(
                        f"{n.fact.subject} \u2014{n.fact.predicate}\u2192 {n.fact.object}"
                    )
        if doc_has_rename:
            rename_successes += 1
    rename_pct = rename_successes / n_docs * 100.0 if n_docs else 0.0
    rename_ci = _wilson_ci(rename_successes, n_docs)
    rename_examples = [_truncate(x) for x in rename_examples_raw[:3]]

    return (
        td_pct,
        td_ci,
        td_examples,
        drift_pct,
        drift_ci,
        drift_examples,
        conflict_pct,
        conflict_ci,
        conflict_examples,
        rename_pct,
        rename_ci,
        rename_examples,
    )


def _verdict(
    *,
    drift_pct: float,
    conflict_pct: float,
    all_pcts: list[float],
) -> Literal["high", "medium", "low"]:
    """Combine per-dimension percentages into a single high/medium/low verdict."""
    if drift_pct + conflict_pct >= 5.0:
        return "high"
    if any(p > 0 for p in all_pcts):
        return "medium"
    return "low"


def _empty_report(
    *,
    n_total: int | None,
    sample_mode: Literal["fast", "deep"] = "fast",
) -> DoctorReport:
    """Return the zero-document fallback report."""
    scores = [
        DoctorScore(
            dimension=dim,
            percentage=0.0,
            ci95=(0.0, 0.0),
            n_sampled=0,
            n_total=n_total,
            examples=[],
        )
        for dim in _DIMENSIONS
    ]
    mode_label = "fast mode" if sample_mode == "fast" else "deep mode"
    return DoctorReport(
        sample_mode=sample_mode,
        scores=scores,
        verdict="low",
        rendered_markdown=(f"# Doctor scan \u2014 {mode_label}\n\nNo documents to scan.\n"),
    )


def _render_markdown_scorecard(
    *,
    scores: list[DoctorScore],
    verdict: Literal["high", "medium", "low"],
    n_sampled: int,
    n_total: int | None,
    drift_pct: float,
    conflict_pct: float,
    mode_label: str = "fast mode",
    warning_line: str | None = None,
) -> str:
    """Render the scan as a single Markdown string (header + table + verdict).

    ``mode_label`` controls the scan-mode text in the header (``"fast mode"``
    or ``"deep mode"``). ``warning_line`` optionally prepends a blockquote
    above the header — used by deep-mode to flag high-failure runs.
    """
    of_total = f" of {n_total}" if n_total is not None else ""
    header = f"# Doctor scan \u2014 {mode_label} (n = {n_sampled}{of_total})"

    labels: dict[str, str] = {
        "temporal_depth": "Temporal depth",
        "temporal_drift": "Temporal drift",
        "conflict_surface": "Conflict surface",
        "rename_events": "Rename events",
    }

    rows: list[str] = [
        "| Dimension          | Score   | 95% CI          | Examples                 |",
        "|--------------------|---------|-----------------|--------------------------|",
    ]
    for s in scores:
        pct = f"{s.percentage:.1f} %"
        lo, hi = s.ci95
        ci = f"{lo:.1f}\u2013{hi:.1f} %"
        examples = "; ".join(s.examples) if s.examples else "\u2014"
        rows.append(f"| {labels[s.dimension]:<18} | {pct:<7} | {ci:<15} | {examples} |")

    combined = drift_pct + conflict_pct
    if combined > 0:
        n_est = max(1, round(100 / combined))
        verdict_line = (
            f"**Verdict:** {verdict} \u2014 running nuggetindex would fix roughly "
            f"1 in {n_est} retrievals."
        )
    else:
        verdict_line = f"**Verdict:** {verdict} \u2014 few / no obvious issues found."

    sections: list[str] = []
    if warning_line:
        sections.extend([warning_line, ""])
    sections.extend([header, "", *rows, "", verdict_line, ""])
    return "\n".join(sections)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


async def scan_index(
    *,
    docs: AsyncIterable[Document] | Iterable[Document],
    mode: Literal["fast", "deep"] = "fast",
    sample_size: int = 500,
    stratify_by: Literal["source_date", "none", "domain", "language", "composite"] = "composite",
    schema: Any | None = None,  # RelationSchema | None -- avoid hard import at module level
    extractor: Any | None = None,  # BaseExtractor | None -- only used when mode="deep"
    rng_seed: int = 0,
    dedup_near_duplicates: bool = False,
) -> DoctorReport:
    """Scan an existing RAG index and estimate nuggetindex-relevant risks.

    Parameters
    ----------
    docs:
        The document source. Either a sync or async iterable of
        :class:`~nuggetindex.pipeline.constructor.Document`. Streaming input is
        supported so callers with large indexes do not need to materialise the
        full corpus.
    mode:
        ``"fast"`` (default) uses only heuristic signals (TIMEX, NER, trigger
        verbs); ``"deep"`` runs the provided ``extractor`` over the sample.
    sample_size:
        Target number of documents to draw for the estimate.
    stratify_by:
        Sampler stratification. Defaults to ``"composite"``, which buckets by
        ``(language, domain)`` for the best out-of-the-box coverage. Other
        options:

        * ``"source_date"`` -- bucket concrete inputs into source-date deciles
          plus an unknown bucket.
        * ``"none"`` -- uniform random sample.
        * ``"domain"`` -- bucket by the host component of ``doc.uri``.
        * ``"language"`` -- bucket by detected language (``langdetect`` with a
          first-letter heuristic fallback).

        Streaming inputs always degrade to reservoir sampling regardless of
        this setting.
    schema:
        Optional :class:`~nuggetindex.core.schema.RelationSchema`. Typed as
        ``Any`` here to keep the module import-light; the real type is
        resolved lazily inside the implementation.
    extractor:
        Optional :class:`~nuggetindex.extractors.base.BaseExtractor`. Only
        consulted when ``mode="deep"``.
    rng_seed:
        Deterministic seed for the sampler's RNG.
    dedup_near_duplicates:
        When ``True``, apply a SimHash-based near-duplicate filter over the
        sampled docs (64-bit hash, 3-bit Hamming threshold). Default
        ``False`` -- opt in when duplicate source pages (syndicated news,
        mirrored docs) would otherwise dominate the score estimates.

    Returns
    -------
    DoctorReport
        Per-dimension scores with 95% confidence bounds, an overall
        ``high``/``medium``/``low`` verdict, and pre-rendered Markdown.
    """
    if mode == "deep":
        if extractor is None:
            raise ValueError(
                "Deep mode requires an `extractor=` argument (e.g., LLMExtractor(...))."
            )
        return await _deep_scan(
            docs=docs,
            sample_size=sample_size,
            stratify_by=stratify_by,
            schema=schema,
            extractor=extractor,
            rng_seed=rng_seed,
            dedup_near_duplicates=dedup_near_duplicates,
        )

    if mode != "fast":
        raise ValueError(f"mode must be 'fast' or 'deep', got {mode!r}")

    # ``extractor`` is accepted for deep-mode API parity but is ignored in
    # fast mode (fast mode is heuristic-only).
    del extractor

    return await _fast_scan(
        docs=docs,
        sample_size=sample_size,
        stratify_by=stratify_by,
        schema=schema,
        rng_seed=rng_seed,
        dedup_near_duplicates=dedup_near_duplicates,
    )
