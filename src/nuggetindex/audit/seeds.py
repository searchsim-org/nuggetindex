"""Automated seed-query proposer for offline nuggetindex construction.

Given an iterable of :class:`~nuggetindex.pipeline.constructor.Document`,
:func:`propose_seeds` returns a budgeted, diversity-optimized set of
``SeedCandidate`` queries most likely to surface high-value facts when an
offline nuggetindex layer is built on top of an existing corpus. The module
is deliberately heuristic + deterministic: no LLM calls are issued here, so
the proposal is a cheap pre-ingest step that a user can run *before* paying
the cost of the structured extractor.

Pipeline summary (see module-level docstring of :func:`propose_seeds` for
the full contract):

1. Stratified sample of ``docs`` (reuses
   :func:`nuggetindex.audit.heuristics.sample.stratified_sample`).
2. Per-doc spaCy NER + trigger-verb scan.
3. Entity ranking: frequency x distinct functional-predicate co-occurrences.
4. Candidate generation:

   * ``functional``      - ``{entity} {predicate-natural-form}`` for every
     schema predicate with ``cardinality == FUNCTIONAL``.
   * ``rename``          - probes for entities that co-occur with
     entity-rename triggers (``"{entity} renamed"`` and
     ``"{entity} formerly known as"``).
   * ``disputed_check``  - predicates/subjects where the sample already
     shows more than one distinct object.
   * ``entity_coverage`` - the bare entity name as a fallback generic
     retrieval probe for the most frequent top-50 entities.
5. Diversity optimisation: TF-IDF (default) or sentence-transformers
   embeddings + greedy facility-location selection to ``budget`` queries
   (Lin & Bilmes 2011). Quality-weighted: candidates with higher
   ``expected_coverage`` get a mild boost in the greedy objective.

Heavy optional deps (``sklearn``, ``sentence-transformers``) are imported
lazily inside the entry point so importing ``nuggetindex.audit.seeds`` stays
cheap and does not force the ``[doctor]`` / ``[seeds]`` extras on callers.
"""

from __future__ import annotations

import contextlib
import re
import warnings
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from nuggetindex.pipeline.constructor import Document  # noqa: F401 - type ref


@dataclass(frozen=True)
class SeedCandidate:
    """A single proposed seed query.

    ``expected_coverage`` is a rough 0..1 score derived from the entity's
    sample frequency and the candidate's kind; it is *not* a probability,
    just a relative ranking hint consumed by the facility-location
    quality-weighting step.
    """

    query: str
    kind: Literal[
        "functional",
        "rename",
        "disputed_check",
        "entity_coverage",
    ]
    entity: str
    predicate: str | None
    expected_coverage: float
    reason: str


@dataclass(frozen=True)
class SeedProposal:
    """Output of :func:`propose_seeds`.

    The rendered Markdown is pre-built so the CLI / notebooks can dump the
    proposal without re-running the pipeline.
    """

    seeds: list[SeedCandidate]
    total_candidates_considered: int
    sample_size: int
    sample_mode: Literal["stratified"]
    rendered_markdown: str = ""


# --------------------------------------------------------------------------- #
# Predicate natural-form helper
# --------------------------------------------------------------------------- #


# Overrides for predicates whose camelCase split does not read naturally. The
# map is small on purpose -- we prefer a deterministic camelCase splitter for
# unknown keys so user-supplied schemas keep working without editing this
# module.
_PREDICATE_NATURAL_OVERRIDES: dict[str, str] = {
    "chiefExecutiveOfficer": "CEO",
    "chiefTechnologyOfficer": "CTO",
    "chiefFinancialOfficer": "CFO",
    "chiefOperatingOfficer": "COO",
    "foundedIn": "founded",
    "headquarteredIn": "headquartered in",
    "renamedTo": "renamed to",
    "formerlyKnownAs": "formerly known as",
    "corporateName": "corporate name",
    "parentCompany": "parent company",
    "dateOfBirth": "date of birth",
    "dateOfDeath": "date of death",
    "placeOfBirth": "place of birth",
    "placeOfDeath": "place of death",
    "employeeCount": "employee count",
}


_CAMEL_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _predicate_natural(predicate: str) -> str:
    """Return a human-readable form of ``predicate``.

    Uses :data:`_PREDICATE_NATURAL_OVERRIDES` for known idiomatic names
    (acronyms, multi-word phrases), falling back to a camelCase split for
    unknown predicates. Result is lowercased for the camelCase case so that
    ``{entity} {natural-form}`` reads naturally (``"Google CEO"``, not
    ``"Google C E O"``).
    """
    override = _PREDICATE_NATURAL_OVERRIDES.get(predicate)
    if override is not None:
        return override
    parts = _CAMEL_SPLIT_RE.split(predicate)
    return " ".join(p.lower() for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Expected-coverage scoring
# --------------------------------------------------------------------------- #


_FREQ_SATURATION = 20.0  # entity count past which frequency contribution plateaus


def _coverage_from_freq(count: int) -> float:
    """Saturating frequency score in ``[0, 1]``."""
    if count <= 0:
        return 0.0
    # Saturating linear ramp: count / saturation, clipped at 1.0.
    return min(1.0, count / _FREQ_SATURATION)


def _expected_coverage(kind: str, count: int) -> float:
    """Rough 0..1 expected-coverage score by kind + entity frequency.

    Per-kind bias (chosen so functional + disputed seeds dominate the
    greedy selection's quality weighting, with rename close behind and
    entity_coverage as a weak fallback):

    * ``disputed_check``    -- base 0.85 (already have multi-object evidence)
    * ``functional``        -- base 0.70
    * ``rename``            -- base 0.60
    * ``entity_coverage``   -- base 0.30
    """
    bases = {
        "disputed_check": 0.85,
        "functional": 0.70,
        "rename": 0.60,
        "entity_coverage": 0.30,
    }
    base = bases.get(kind, 0.30)
    # Blend base with frequency score, weighted 70/30 so kind dominates
    # but very rare entities still rank lower than very frequent ones of
    # the same kind.
    freq_score = _coverage_from_freq(count)
    return float(0.7 * base + 0.3 * freq_score)


# --------------------------------------------------------------------------- #
# Candidate generation helpers
# --------------------------------------------------------------------------- #


def _iter_functional_predicates(schema: Any) -> list[str]:
    """Return the list of predicate canonical names whose cardinality is FUNCTIONAL.

    ``schema`` is accepted as ``Any`` so the module stays import-light; the
    resolver performs duck-typed lookups on ``_by_name`` + ``cardinality``.
    """
    try:
        from nuggetindex.core.enums import Cardinality
    except ImportError:  # pragma: no cover -- enum always present with package
        return []
    if schema is None:
        return []
    by_name = getattr(schema, "_by_name", None)
    if not isinstance(by_name, dict):
        return []
    out: list[str] = []
    for name, rel in by_name.items():
        card = getattr(rel, "cardinality", None)
        if card == Cardinality.FUNCTIONAL:
            out.append(str(name))
    return out


# Common corporate / legal-form suffixes stripped before bucketing entity
# mentions so "Twitter" and "Twitter Inc." collapse to the same key.
_CORPORATE_SUFFIX_RE = re.compile(
    r"(?:,?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|LLC|LLP|PLC|GmbH|AG|SA|N\.V\.|NV|SE)\.?)+\s*$",
    re.IGNORECASE,
)


def _normalize_entity(text: str) -> str:
    """Collapse whitespace, strip, and drop trailing corporate-suffix tails.

    "Twitter Inc." and "Twitter" normalise to the same key so entity
    counting and duplicate detection don't fragment a single real-world
    entity across surface-form variants. Case is preserved so downstream
    query rendering keeps reasonable capitalisation.
    """
    cleaned = " ".join(text.split()).strip()
    cleaned = _CORPORATE_SUFFIX_RE.sub("", cleaned).strip()
    return cleaned


def _subject_slice(text: str, match: Any) -> str:
    """Whitespace-stripped subject slice for a :class:`TriggerMatch`."""
    start, end = match.subject_span
    if end <= start:
        return ""
    return text[start:end].strip()


def _object_slice(text: str, match: Any) -> str:
    """Narrow object slice for a :class:`TriggerMatch` (empty-span sentinel respected)."""
    start, end = match.object_span
    if (start, end) == (0, 0) or end <= start:
        return ""
    return text[start:end].strip()


# --------------------------------------------------------------------------- #
# Greedy facility-location (Lin & Bilmes 2011-style submodular selection)
# --------------------------------------------------------------------------- #


def _greedy_facility_location(
    vectors: Any,  # np.ndarray (n x d); L2-normalised
    budget: int,
    quality_scores: Any,  # np.ndarray (n,)
) -> list[int]:
    """Submodular greedy: pick ``budget`` indices maximising quality * coverage.

    The objective is the classic facility-location / max-marginal-relevance
    hybrid: each un-selected candidate's gain is

        gain(i) = q(i) * (1 - max_{j in S} cos(v_i, v_j))

    where ``q(i)`` is the candidate's per-kind-and-frequency quality weight
    (rescaled into ``[0.5, 1.0]`` so the weight never zeroes the coverage
    term). The first pick is the highest-quality candidate; from there the
    loop tracks ``min_sim_to_selected`` (i.e. the running max of cosine
    similarities to already-selected vectors) and picks whichever un-selected
    candidate maximises the gain above.

    Returns the indices of the selected candidates in the order they were
    picked. Deterministic modulo floating-point tie-breaks (argmax returns
    the first max, so ties resolve to the lower original index).
    """
    import numpy as np

    n = int(vectors.shape[0])
    if n == 0:
        return []
    if n <= budget:
        # Still order by descending quality so callers can optionally slice
        # the result back down later without losing the quality signal.
        order = np.argsort(-quality_scores, kind="stable")
        return [int(i) for i in order]

    # Normalise quality into [0.5, 1.0] -- mild weighting so diversity still
    # matters for low-quality candidates; see module docstring for rationale.
    q_max = float(quality_scores.max()) + 1e-9
    q = 0.5 + 0.5 * (quality_scores / q_max)

    # Cosine-similarity matrix assumes L2-normalised vectors (enforced by the
    # caller). For TF-IDF the matrix is dense after .toarray(); for
    # sentence-transformers it is already dense.
    S = vectors @ vectors.T

    selected: list[int] = []
    first = int(np.argmax(q))
    selected.append(first)
    min_sim_to_selected = np.asarray(S[:, first], dtype=float).copy().reshape(-1)

    for _ in range(budget - 1):
        gain = q * (1.0 - min_sim_to_selected)
        # Mask already-selected so they never win argmax again.
        for s in selected:
            gain[s] = -1.0
        next_idx = int(np.argmax(gain))
        if gain[next_idx] <= -1.0 + 1e-12:
            break  # pragma: no cover -- all candidates exhausted defensively
        selected.append(next_idx)
        min_sim_to_selected = np.maximum(
            min_sim_to_selected,
            np.asarray(S[:, next_idx], dtype=float).reshape(-1),
        )

    return selected


# --------------------------------------------------------------------------- #
# Embedding helpers
# --------------------------------------------------------------------------- #


def _embed_tfidf(queries: list[str]) -> Any:
    """L2-normalised TF-IDF over character-3 n-grams. Lazy-imports sklearn."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        max_features=5000,
    )
    try:
        mat = vectoriser.fit_transform(queries)
    except ValueError:
        # Empty vocabulary (e.g. all queries empty). Return a degenerate
        # zero matrix so the greedy step still has a defined shape.
        return np.zeros((len(queries), 1), dtype=float)
    dense = mat.toarray().astype(float)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return dense / norms


def _embed_sentence_transformer(queries: list[str], model_name: str) -> Any:
    """L2-normalised sentence-transformer embeddings; raises ImportError upward."""
    import numpy as np
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(model_name)
    vecs = np.asarray(model.encode(queries, show_progress_bar=False), dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _embed(queries: list[str], embedding_model: str | None) -> Any:
    """Embed ``queries`` using the chosen backend; fall back to TF-IDF on ImportError."""
    if embedding_model is None:
        return _embed_tfidf(queries)
    try:
        return _embed_sentence_transformer(queries, embedding_model)
    except ImportError:
        warnings.warn(
            "sentence-transformers not installed; falling back to TF-IDF "
            "character n-grams for seed-query embedding. Install "
            "``nuggetindex[seeds]`` to enable sentence-transformer embeddings.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _embed_tfidf(queries)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _render_markdown(
    *,
    seeds: list[SeedCandidate],
    total_candidates: int,
    sample_size: int,
) -> str:
    """Render a seed-proposal table as a single Markdown string."""
    header = (
        f"# Seed proposal (n = {len(seeds)} of {total_candidates} candidates "
        f"considered, sample = {sample_size})"
    )
    if not seeds:
        return header + "\n\nNo seed candidates surfaced.\n"
    rows: list[str] = [
        "| # | Query | Kind | Entity | Coverage | Reason |",
        "|---|-------|------|--------|----------|--------|",
    ]
    for i, s in enumerate(seeds, 1):
        rows.append(
            f"| {i} | {s.query} | {s.kind} | {s.entity} | "
            f"{s.expected_coverage:.2f} | {s.reason} |"
        )
    return "\n".join([header, "", *rows, ""])


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


async def propose_seeds(
    *,
    docs: AsyncIterable[Document] | Iterable[Document],
    schema: Any | None = None,
    budget: int = 50,
    sample_size: int = 500,
    stratify_by: Literal[
        "source_date", "none", "domain", "language", "composite"
    ] = "composite",
    min_entity_frequency: int = 3,
    embedding_model: str | None = None,
    rng_seed: int = 0,
) -> SeedProposal:
    """Return a budgeted, diversity-optimized set of seed queries.

    Pipeline:

    1. Stratified sample of ``docs``.
    2. Per-doc spaCy NER + trigger-verb scan (reuse
       :mod:`nuggetindex.audit.heuristics`).
    3. Entity ranking: ``count * (1 + |predicates|) * (1 + 0.2 * |languages|)``.
    4. Candidate generation: top-K entities x {functional predicates, rename
       triggers, disputed-check} + generic entity-coverage queries.
    5. Diversity optimisation: TF-IDF (default) or sentence-transformer
       embeddings + greedy facility-location selection to ``budget`` queries.

    Parameters
    ----------
    docs:
        Sync or async iterable of :class:`Document`.
    schema:
        Optional :class:`~nuggetindex.core.schema.RelationSchema`. Defaults to
        :meth:`RelationSchema.default` when ``None``.
    budget:
        Maximum number of seed candidates to return.
    sample_size:
        Target number of documents to draw for entity ranking.
    stratify_by:
        Sampler stratification (see
        :func:`nuggetindex.audit.heuristics.stratified_sample`).
    min_entity_frequency:
        Minimum observed count for an entity to qualify for the top-K ranking.
    embedding_model:
        ``None`` -> TF-IDF character 3-gram embeddings (no extra deps);
        otherwise a sentence-transformers model id
        (``"all-MiniLM-L6-v2"`` etc.). Falls back to TF-IDF with a warning
        if ``sentence-transformers`` is not installed.
    rng_seed:
        Deterministic seed for the sampler's RNG.

    Returns
    -------
    SeedProposal
        Budgeted, diversity-optimised seed candidates with pre-rendered
        Markdown.
    """
    # Lazy imports to keep module-level import cheap ------------------------
    import numpy as np

    from nuggetindex.audit.heuristics import (
        extract_entities,
        scan_triggers,
        stratified_sample,
    )
    from nuggetindex.audit.heuristics.language import _detect_language

    # --- Step 1: stratified sample -----------------------------------------
    sampled_docs, _n_total = await stratified_sample(
        docs,
        sample_size=sample_size,
        stratify_by=stratify_by,
        rng_seed=rng_seed,
    )

    if not sampled_docs:
        return SeedProposal(
            seeds=[],
            total_candidates_considered=0,
            sample_size=0,
            sample_mode="stratified",
            rendered_markdown=_render_markdown(
                seeds=[], total_candidates=0, sample_size=0
            ),
        )

    # Resolve schema lazily so core module import stays light.
    if schema is None:
        try:
            from nuggetindex.core.schema import RelationSchema

            schema = RelationSchema.default()
        except Exception:  # pragma: no cover -- defensive; default schema should load
            schema = None

    # --- Step 2: NER + trigger scan ----------------------------------------
    entity_counts: dict[str, int] = {}
    entity_predicates: dict[str, set[str]] = {}
    entity_languages: dict[str, set[str]] = {}
    rename_entities: set[str] = set()
    disputed_keys: dict[tuple[str, str], set[str]] = {}

    for doc in sampled_docs:
        text = doc.text or ""
        if not text:
            continue
        lang = _detect_language(text)

        # --- entities --------------------------------------------------------
        for ent in extract_entities(text):
            name = _normalize_entity(ent.text)
            if not name:
                continue
            entity_counts[name] = entity_counts.get(name, 0) + 1
            entity_languages.setdefault(name, set()).add(lang)

        # --- triggers --------------------------------------------------------
        for match in scan_triggers(text):
            subj = _subject_slice(text, match)
            subj_norm = _normalize_entity(subj)
            if not subj_norm:
                continue

            # Normalise predicate via schema when available.
            pred = match.predicate
            if schema is not None:
                with contextlib.suppress(Exception):  # defensive
                    pred = str(schema.canonicalize(pred))

            entity_predicates.setdefault(subj_norm, set()).add(pred)
            # Bump the trigger subject's count so entities surfaced only
            # by the trigger scanner (not spaCy NER) still qualify for
            # the top-K ranking. spaCy-tagged entities that happen to
            # also be a trigger subject will get a small double-count
            # here; that bias favours entities with both NER and trigger
            # evidence, which is the correct direction.
            entity_counts[subj_norm] = entity_counts.get(subj_norm, 0) + 1
            entity_languages.setdefault(subj_norm, set()).add(lang)

            if match.kind == "entity_rename":
                rename_entities.add(subj_norm)

            obj = _object_slice(text, match)
            if obj:
                key = (subj_norm, pred)
                disputed_keys.setdefault(key, set()).add(obj)

    # --- Step 3: entity ranking --------------------------------------------
    eligible: list[tuple[str, float, int]] = []
    for name, count in entity_counts.items():
        if count < min_entity_frequency:
            continue
        preds = entity_predicates.get(name, set())
        langs = entity_languages.get(name, set())
        score = count * (1 + len(preds)) * (1 + 0.2 * len(langs))
        eligible.append((name, score, count))
    eligible.sort(key=lambda t: (t[1], t[2], t[0]), reverse=True)
    top_k_cap = min(200, 4 * budget)
    top_entities = eligible[:top_k_cap]

    # --- Step 4: candidate generation --------------------------------------
    functional_predicates = _iter_functional_predicates(schema)
    candidates: list[SeedCandidate] = []
    seen_queries: set[str] = set()

    def _add(cand: SeedCandidate) -> None:
        if cand.query in seen_queries:
            return
        seen_queries.add(cand.query)
        candidates.append(cand)

    # Rank-indexed for reason-line text ("top-N entity + ...").
    rank_of: dict[str, int] = {name: i + 1 for i, (name, _s, _c) in enumerate(top_entities)}

    for name, _score, count in top_entities:
        rank = rank_of[name]
        # 4a. functional predicates x entity -----------------------------
        for pred in functional_predicates:
            natural = _predicate_natural(pred)
            if not natural:
                continue
            query = f"{name} {natural}".strip()
            reason = f"top-{rank} entity + functional predicate {natural}"
            _add(
                SeedCandidate(
                    query=query,
                    kind="functional",
                    entity=name,
                    predicate=pred,
                    expected_coverage=_expected_coverage("functional", count),
                    reason=reason,
                )
            )

    # 4b. rename probes ------------------------------------------------------
    for name in rename_entities:
        count = entity_counts.get(name, 0)
        for tmpl in (f"{name} renamed", f"{name} formerly known as"):
            _add(
                SeedCandidate(
                    query=tmpl,
                    kind="rename",
                    entity=name,
                    predicate="renamedTo",
                    expected_coverage=_expected_coverage("rename", count),
                    reason="entity observed with rename trigger",
                )
            )

    # 4c. disputed-check probes ---------------------------------------------
    for (subj, pred), objects in disputed_keys.items():
        if len(objects) < 2:
            continue
        natural = _predicate_natural(pred)
        if not natural:
            continue
        query = f"{subj} {natural}".strip()
        count = entity_counts.get(subj, 0)
        _add(
            SeedCandidate(
                query=query,
                kind="disputed_check",
                entity=subj,
                predicate=pred,
                expected_coverage=_expected_coverage("disputed_check", count),
                reason=f"disputed candidate: {len(objects)} distinct objects already seen",
            )
        )

    # 4d. entity-coverage fallbacks (top-50 by rank) ------------------------
    for name, _score, count in top_entities[:50]:
        _add(
            SeedCandidate(
                query=name,
                kind="entity_coverage",
                entity=name,
                predicate=None,
                expected_coverage=_expected_coverage("entity_coverage", count),
                reason=f"generic coverage probe for top-{rank_of[name]} entity",
            )
        )

    # Make sure top-K entities that fell below NER-only counting still end up
    # with at least one candidate even if no schema-functional predicate
    # produced a query (e.g. empty schema or no functional predicates).
    if not candidates:
        for name, _score, count in top_entities[:budget]:
            _add(
                SeedCandidate(
                    query=name,
                    kind="entity_coverage",
                    entity=name,
                    predicate=None,
                    expected_coverage=_expected_coverage("entity_coverage", count),
                    reason="fallback entity-coverage probe",
                )
            )

    total_candidates_considered = len(candidates)
    if total_candidates_considered == 0:
        return SeedProposal(
            seeds=[],
            total_candidates_considered=0,
            sample_size=len(sampled_docs),
            sample_mode="stratified",
            rendered_markdown=_render_markdown(
                seeds=[], total_candidates=0, sample_size=len(sampled_docs)
            ),
        )

    # --- Step 5: diversity optimisation ------------------------------------
    quality_scores = np.asarray(
        [c.expected_coverage for c in candidates], dtype=float
    )
    queries = [c.query for c in candidates]
    vectors = _embed(queries, embedding_model)

    effective_budget = max(0, int(budget))
    if effective_budget == 0:
        selected_idx: list[int] = []
    else:
        selected_idx = _greedy_facility_location(
            vectors, effective_budget, quality_scores
        )
        selected_idx = selected_idx[:effective_budget]

    selected_seeds = [candidates[i] for i in selected_idx]

    # --- Step 6: render + return -------------------------------------------
    rendered = _render_markdown(
        seeds=selected_seeds,
        total_candidates=total_candidates_considered,
        sample_size=len(sampled_docs),
    )
    return SeedProposal(
        seeds=selected_seeds,
        total_candidates_considered=total_candidates_considered,
        sample_size=len(sampled_docs),
        sample_mode="stratified",
        rendered_markdown=rendered,
    )


__all__ = ["SeedCandidate", "SeedProposal", "propose_seeds"]
