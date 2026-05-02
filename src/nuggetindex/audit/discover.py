"""Corpus-driven predicate discovery for ``nuggetindex`` v0.3.

``nuggetindex`` ships a Silicon-Valley-flavoured default schema (CEO,
acquired, renamedTo, ...) that doesn't fit legal firms, hospitals, pharma
companies, or universities. Asking those users to hand-author a YAML is a
non-starter. :func:`discover_schema` takes a corpus, samples / extracts /
aggregates, and proposes the predicates that actually show up -- with
cardinality (``functional`` / ``multi_valued`` / ``event_log``) and
expected subject/object NER types. Users can merge the proposal into the
default schema via :func:`merge_proposal` and keep going.

Two modes:

* **Cheap** (``extractor=None`` -> :class:`TriggerExtractor`): uses the
  trigger-verb patterns + spaCy NER. Zero LLM cost. Recovers ~60% of
  predicates for typical business corpora.
* **LLM** (``extractor=LLMExtractor(...)``): uses the full LLM extractor.
  Recovers all predicates the LLM surfaces, including domain-specific
  ones the trigger patterns don't know about.

Pipeline summary (``discover_schema`` docstring has the full contract):

1. Stratified sample of ``docs``.
2. Extract nuggets from each sampled doc via the given extractor.
3. Aggregate by canonical predicate. Predicates already in the default
   schema canonicalize through it; truly new predicates are canonicalised
   by :func:`_to_camel_case` so a corpus that emits ``"filed against"``
   bubbles up as ``"filedAgainst"``.
4. For each predicate with frequency >= ``min_frequency``:

   * Infer cardinality from event-log word set / functional-ratio / else
     ``multi_valued``.
   * Infer expected subject/object NER types (threshold: label appears in
     >= 50% of the sampled rows for that predicate).
   * Collect aliases seen before canonicalisation.
   * Keep up to 3 ``(subject, predicate, object)`` examples.

5. Render:

   * ``rendered_yaml`` -- drop-in YAML that can be concatenated / merged
     with the default schema.
   * ``rendered_markdown`` -- human-readable summary with cardinality
     breakdown and a "5 most confident predicates" highlight.

The proposal is standalone; the default YAML is never touched.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover -- type-check-only import
    from nuggetindex.pipeline.constructor import Document

# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PredicateProposal:
    """A single predicate inferred from the corpus.

    ``aliases`` collects surface forms seen in the sample that canonicalise
    to ``name`` (either via the existing schema's alias table or via the
    discovery module's own camelCase canonicaliser).
    """

    name: str
    cardinality: str
    expected_subject_types: list[str] = field(default_factory=list)
    expected_object_types: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    frequency: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaProposal:
    """Output of :func:`discover_schema`.

    ``rendered_yaml`` / ``rendered_markdown`` are pre-built so the CLI /
    notebooks can dump the proposal without re-running the pipeline.
    """

    predicates: list[PredicateProposal]
    n_docs_sampled: int
    n_docs_total: int | None
    rendered_yaml: str
    rendered_markdown: str


# --------------------------------------------------------------------------- #
# Event-log word set
# --------------------------------------------------------------------------- #


# Predicates whose extractions are a stream of events rather than a
# contested single-valued attribute. Matched against the canonical name
# and against every alias. Keep in sync with the ``cardinality: event_log``
# entries in ``core/schemas/default_predicates.yaml``.
_EVENT_LOG_WORDS: frozenset[str] = frozenset(
    {
        "announced",
        "announces",
        "announcing",
        "said",
        "says",
        "saying",
        "reported",
        "reports",
        "reporting",
        "launched",
        "launches",
        "launching",
        "released",
        "releases",
        "releasing",
        "published",
        "publishes",
        "publishing",
        "confirmed",
        "confirms",
        "confirming",
        "presented",
        "presents",
        "presenting",
        "disclosed",
        "discloses",
        "disclosing",
        "stated",
        "states",
        "stating",
        "noted",
        "notes",
        "noting",
    }
)


# --------------------------------------------------------------------------- #
# Canonicalisation helpers
# --------------------------------------------------------------------------- #


_CAMEL_ALREADY_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_CAMEL_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")
_PUNCT_STRIP_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_TOKEN_SPLIT_RE = re.compile(r"[\s_]+")


def _to_camel_case(raw: str) -> str:
    """Return ``raw`` in lowercase camelCase.

    Splits on whitespace / underscore, strips punctuation, lowercases the
    first token, title-cases subsequent tokens. Already-camelCase names
    (matching ``[a-z][a-zA-Z0-9]*`` with no whitespace / punctuation) are
    returned unchanged so a predicate that round-trips through the default
    schema's alias table keeps its form.
    """
    s = raw.strip()
    if not s:
        return ""
    if _CAMEL_ALREADY_RE.match(s):
        return s
    stripped = _PUNCT_STRIP_RE.sub(" ", s)
    tokens = [t for t in _TOKEN_SPLIT_RE.split(stripped) if t]
    if not tokens:
        return ""
    first = tokens[0].lower()
    rest = [t[:1].upper() + t[1:].lower() for t in tokens[1:]]
    return first + "".join(rest)


def _predicate_is_event_log(name: str, aliases: Iterable[str]) -> bool:
    """Return True iff the canonical name or any alias is in the event-log set."""
    if name.lower() in _EVENT_LOG_WORDS:
        return True
    return any(alias.lower() in _EVENT_LOG_WORDS for alias in aliases)


# --------------------------------------------------------------------------- #
# Canonicalisation entry point
# --------------------------------------------------------------------------- #


def _canonicalize_predicate(raw: str, schema: Any | None) -> str:
    """Resolve ``raw`` through ``schema`` first, then camelCase fallback.

    If the schema knows ``raw`` (via the alias table), the canonical name
    wins -- that keeps ``"CEO"`` / ``"chief executive officer"`` collapsing
    onto ``"chiefExecutiveOfficer"``. Otherwise the name is camelCased.
    """
    if schema is not None:
        try:
            canonical = str(schema.canonicalize(raw))
        except Exception:  # pragma: no cover -- defensive
            canonical = raw
        # If canonicalize actually rewrote the input (we hit the alias
        # table), trust it. Otherwise we fell through and should camelCase.
        if canonical != raw:
            return canonical
    return _to_camel_case(raw) or raw


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #


# Sentinels returned by :func:`probe_entity_type` that are NOT real NER
# labels -- they indicate "spaCy not installed", "no entity found", or
# "multiple entities in mention". Filter them out of the majority-labels
# pass so the proposal never leaks sentinels into ``expected_*_types``.
_NER_SENTINELS: frozenset[str] = frozenset({"UNAVAILABLE", "NONE", "COMPOUND"})


def _majority_labels(labels: list[str | None], threshold: float = 0.5) -> list[str]:
    """Return real NER labels that appear in >= ``threshold`` share of observations.

    ``None`` and sentinel entries (``UNAVAILABLE`` / ``NONE`` / ``COMPOUND``)
    are kept in the denominator so "sometimes unknown" doesn't push a
    minority label above the threshold -- but they are never returned.
    Labels are sorted by frequency descending for a stable YAML output.
    """
    if not labels:
        return []
    total = len(labels)
    counts = Counter(
        label for label in labels if label and label not in _NER_SENTINELS
    )
    out = [
        (label, c)
        for label, c in counts.items()
        if c / total >= threshold
    ]
    out.sort(key=lambda t: (-t[1], t[0]))
    return [label for label, _ in out]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class _PredicateAccumulator:
    """Per-predicate bag of evidence collected across the sample."""

    name: str
    raw_surface_forms: Counter = field(default_factory=Counter)
    subject_types: list[str | None] = field(default_factory=list)
    object_types: list[str | None] = field(default_factory=list)
    # Distinct (subject, object) pairs, keyed by subject -> set of objects.
    subj_to_objs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    examples: list[str] = field(default_factory=list)
    frequency: int = 0


def _infer_cardinality(
    acc: _PredicateAccumulator,
    aliases: list[str],
) -> str:
    """Return ``"event_log"`` / ``"functional"`` / ``"multi_valued"``.

    * EVENT_LOG if the canonical name or any alias is in the event-log set.
    * FUNCTIONAL if every (subject, predicate) key has exactly one distinct
      object in the sample AND there is >= one such key.
    * MULTI_VALUED otherwise.
    """
    if _predicate_is_event_log(acc.name, aliases):
        return "event_log"
    if acc.subj_to_objs:
        all_single = all(len(objs) == 1 for objs in acc.subj_to_objs.values())
        if all_single:
            return "functional"
    return "multi_valued"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_yaml(predicates: list[PredicateProposal]) -> str:
    """Render a proposal as a drop-in ``predicates:`` YAML mapping.

    We build the structure as a plain ``dict`` and dump via PyYAML so the
    output is syntactically valid even with oddly-named predicates.
    """
    import yaml

    if not predicates:
        return "predicates: {}\n"

    payload: dict[str, Any] = {"predicates": {}}
    for p in predicates:
        entry: dict[str, Any] = {
            "cardinality": p.cardinality,
            "functional": p.cardinality == "functional",
        }
        if p.expected_subject_types:
            entry["expected_subject_types"] = list(p.expected_subject_types)
        if p.expected_object_types:
            entry["expected_object_types"] = list(p.expected_object_types)
        if p.aliases:
            entry["aliases"] = list(p.aliases)
        payload["predicates"][p.name] = entry

    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def _render_markdown(
    *,
    predicates: list[PredicateProposal],
    n_docs_sampled: int,
    n_docs_total: int | None,
) -> str:
    """Human-readable summary: counts by cardinality + top predicates."""
    denom = (
        f"{n_docs_sampled}"
        if n_docs_total is None
        else f"{n_docs_sampled} of {n_docs_total}"
    )
    header = f"# Schema proposal (n = {len(predicates)} predicates, sample = {denom} docs)"

    if not predicates:
        return header + "\n\nNo predicates surfaced above the frequency threshold.\n"

    by_card: Counter = Counter(p.cardinality for p in predicates)
    breakdown_lines = ["## Cardinality breakdown", ""]
    for card in ("functional", "multi_valued", "event_log"):
        if by_card.get(card):
            breakdown_lines.append(f"- **{card}**: {by_card[card]}")
    breakdown_lines.append("")

    # Confidence = frequency (predicates seen more often win). Break ties by
    # name for determinism.
    most_confident = sorted(
        predicates, key=lambda p: (-p.frequency, p.name)
    )[:5]
    top_lines = ["## 5 most confident predicates", ""]
    if not most_confident:
        top_lines.append("_(none)_")
        top_lines.append("")
    else:
        top_lines.append("| # | Predicate | Cardinality | Frequency | Subject types | Object types |")
        top_lines.append("|---|-----------|-------------|-----------|---------------|--------------|")
        for i, p in enumerate(most_confident, 1):
            subj = ", ".join(p.expected_subject_types) or "-"
            obj = ", ".join(p.expected_object_types) or "-"
            top_lines.append(
                f"| {i} | {p.name} | {p.cardinality} | {p.frequency} | {subj} | {obj} |"
            )
        top_lines.append("")

    all_lines = ["## All proposed predicates", ""]
    all_lines.append("| Predicate | Cardinality | Frequency | Aliases |")
    all_lines.append("|-----------|-------------|-----------|---------|")
    for p in sorted(predicates, key=lambda p: p.name):
        alias = ", ".join(p.aliases) or "-"
        all_lines.append(
            f"| {p.name} | {p.cardinality} | {p.frequency} | {alias} |"
        )
    all_lines.append("")

    return "\n".join(
        [header, "", *breakdown_lines, *top_lines, *all_lines]
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


async def discover_schema(
    *,
    docs: Iterable[Document] | AsyncIterable[Document],
    sample_size: int = 500,
    stratify_by: Literal[
        "composite", "source_date", "language", "domain", "none"
    ] = "composite",
    extractor: Any | None = None,
    min_frequency: int = 3,
    rng_seed: int = 0,
) -> SchemaProposal:
    """Discover predicates in a corpus by sampling + extracting + aggregating.

    Parameters
    ----------
    docs:
        Sync or async iterable of :class:`Document`.
    sample_size:
        Target number of documents to draw for extraction.
    stratify_by:
        Sampler stratification (see
        :func:`nuggetindex.audit.heuristics.stratified_sample`).
    extractor:
        Optional :class:`BaseExtractor`. When ``None`` a fresh
        :class:`TriggerExtractor` is instantiated -- zero LLM cost. Pass
        :class:`LLMExtractor` (or any subclass) to recover domain-specific
        predicates the trigger patterns don't know.
    min_frequency:
        Predicates seen fewer than this many times are dropped as noise.
    rng_seed:
        Deterministic seed for the sampler's RNG.

    Returns
    -------
    SchemaProposal
        Predicate proposals, pre-rendered YAML + Markdown.
    """
    # Lazy imports keep module-level import light.
    from nuggetindex.audit.heuristics.sample import stratified_sample

    sampled_docs, n_total = await stratified_sample(
        docs,
        sample_size=sample_size,
        stratify_by=stratify_by,
        rng_seed=rng_seed,
    )

    if not sampled_docs:
        return SchemaProposal(
            predicates=[],
            n_docs_sampled=0,
            n_docs_total=n_total,
            rendered_yaml=_render_yaml([]),
            rendered_markdown=_render_markdown(
                predicates=[], n_docs_sampled=0, n_docs_total=n_total
            ),
        )

    # Resolve schema (for alias-table canonicalisation).
    schema: Any | None
    try:
        from nuggetindex.core.schema import RelationSchema

        schema = RelationSchema.default()
    except Exception:  # pragma: no cover -- defensive
        schema = None

    # Resolve extractor.
    effective_extractor: Any
    if extractor is None:
        from nuggetindex.extractors.trigger import TriggerExtractor

        effective_extractor = TriggerExtractor()
    else:
        effective_extractor = extractor

    # --- Extract + aggregate ------------------------------------------------
    accumulators: dict[str, _PredicateAccumulator] = {}

    for doc in sampled_docs:
        text = doc.text or ""
        if not text:
            continue
        try:
            results = await effective_extractor.aextract(
                text,
                context="",
                source_id=doc.source_id,
            )
        except TypeError:
            # Pre-0.2 extractors without source_id kwarg.
            results = await effective_extractor.aextract(text, context="")
        except Exception:  # pragma: no cover -- defensive; skip malformed docs
            continue

        for res in results:
            nug = res.nugget
            fact = getattr(nug, "fact", None)
            if fact is None:
                continue
            raw_predicate = str(fact.predicate)
            if not raw_predicate.strip():
                continue
            canonical = _canonicalize_predicate(raw_predicate, schema)
            if not canonical:
                continue

            acc = accumulators.get(canonical)
            if acc is None:
                acc = _PredicateAccumulator(name=canonical)
                accumulators[canonical] = acc

            acc.frequency += 1
            acc.raw_surface_forms[raw_predicate] += 1
            acc.subject_types.append(fact.subject_type)
            acc.object_types.append(fact.object_type)
            subj = str(fact.subject).strip()
            obj = str(fact.object).strip()
            if subj and obj:
                acc.subj_to_objs[subj].add(obj)
            if len(acc.examples) < 3 and subj and obj:
                acc.examples.append(f"({subj}, {canonical}, {obj})")

    # --- Filter + build proposals ------------------------------------------
    proposals: list[PredicateProposal] = []
    for canonical, acc in accumulators.items():
        if acc.frequency < min_frequency:
            continue

        # Aliases = non-canonical surface forms seen in the sample.
        aliases = sorted(
            form
            for form in acc.raw_surface_forms
            if form != canonical
        )
        cardinality = _infer_cardinality(acc, aliases)
        subj_types = _majority_labels(acc.subject_types)
        obj_types = _majority_labels(acc.object_types)

        proposals.append(
            PredicateProposal(
                name=canonical,
                cardinality=cardinality,
                expected_subject_types=subj_types,
                expected_object_types=obj_types,
                aliases=aliases,
                frequency=acc.frequency,
                examples=list(acc.examples),
            )
        )

    # Deterministic ordering: most frequent first, ties by name.
    proposals.sort(key=lambda p: (-p.frequency, p.name))

    rendered_yaml = _render_yaml(proposals)
    rendered_markdown = _render_markdown(
        predicates=proposals,
        n_docs_sampled=len(sampled_docs),
        n_docs_total=n_total,
    )

    return SchemaProposal(
        predicates=proposals,
        n_docs_sampled=len(sampled_docs),
        n_docs_total=n_total,
        rendered_yaml=rendered_yaml,
        rendered_markdown=rendered_markdown,
    )


# --------------------------------------------------------------------------- #
# merge_proposal: weave the proposal into a RelationSchema
# --------------------------------------------------------------------------- #


def merge_proposal(
    base: Any,
    proposal: SchemaProposal,
    *,
    accept_all: bool = False,
    accepted_names: set[str] | None = None,
) -> Any:
    """Return a new :class:`RelationSchema` extended with proposal predicates.

    Predicates that already exist in ``base`` (either as the canonical name
    or as an alias) are left unchanged -- the base always wins on conflict
    so re-running discovery never trashes curated schemas.

    Parameters
    ----------
    base:
        The existing :class:`RelationSchema`.
    proposal:
        The :class:`SchemaProposal` returned by :func:`discover_schema`.
    accept_all:
        When ``True`` (and ``accepted_names`` is ``None``), every new
        predicate in the proposal is merged.
    accepted_names:
        Optional explicit allow-list of predicate names. When provided,
        only these predicates are merged (``accept_all`` is ignored).
    """
    from nuggetindex.core.enums import Cardinality
    from nuggetindex.core.schema import Relation, RelationKind, RelationSchema

    # Snapshot the base's relations so we can rebuild a new schema with
    # the additions folded in. RelationSchema is intentionally
    # immutable-ish, so we construct a fresh one from the merged list.
    base_relations: list[Relation] = list(getattr(base, "_by_name", {}).values())
    existing_names = {r.name for r in base_relations}

    # Names (and aliases) the base already knows -- treated as "already
    # covered" so the proposal doesn't shadow curated entries.
    covered_lower: set[str] = set()
    for r in base_relations:
        covered_lower.add(r.name.lower())
        for a in r.aliases:
            covered_lower.add(str(a).lower())

    additions: list[Relation] = []
    for p in proposal.predicates:
        # Acceptance gate: explicit allow-list wins, else ``accept_all`` flag.
        if accepted_names is not None:
            if p.name not in accepted_names:
                continue
        elif not accept_all:
            continue
        # Do not overwrite base predicates (by canonical name or alias).
        if p.name in existing_names:
            continue
        if p.name.lower() in covered_lower:
            continue

        try:
            cardinality = Cardinality(p.cardinality)
        except ValueError:  # pragma: no cover -- defensive; values come from us
            cardinality = Cardinality.MULTI_VALUED

        kind = (
            RelationKind.FUNCTIONAL
            if cardinality == Cardinality.FUNCTIONAL
            else RelationKind.MULTI_VALUED
        )
        additions.append(
            Relation(
                name=p.name,
                kind=kind,
                renaming=False,
                cardinality=cardinality,
                aliases=tuple(p.aliases),
                expected_subject_types=frozenset(p.expected_subject_types),
                expected_object_types=frozenset(p.expected_object_types),
            )
        )

    if not additions:
        # Still return a fresh schema so callers can rely on ``is not base``
        # semantics when they need an independent instance.
        return RelationSchema(base_relations)

    return RelationSchema([*base_relations, *additions])


__all__ = [
    "PredicateProposal",
    "SchemaProposal",
    "discover_schema",
    "merge_proposal",
]
