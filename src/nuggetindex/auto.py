"""One-call adoption facade: discovery → seeds → ingest → sidecar.

:func:`auto` composes the four steps a first-time user otherwise stitches
together by hand and returns a ready-to-query :class:`~Sidecar` plus an
:class:`AutoReport` audit trail. The goal is to turn 4-5 configuration
decisions (schema extension, seed budget, extractor choice, cache path,
store path) into a three-parameter call that still produces sensible
defaults.

Pipeline::

    discover_schema(docs, extractor=TriggerExtractor())  # zero-cost
      -> merge_proposal(RelationSchema.default(), proposal, accept_all=True)
      -> propose_seeds(docs, budget=budget)              # greedy facility-location
      -> NuggetStore(schema=merged, extractor=CachedExtractor(inner=extractor))
      -> aingest(doc) for each doc
      -> Sidecar(store=..., mode=mode, extractor=extractor)

Each step is *additive*: if the user wants to skip schema discovery they
pass ``schema_discovery=False``; if they want to swap the default trigger
extractor for an LLM one they pass ``extractor=LLMExtractor(...)``. The
default configuration never needs an LLM key — the trigger extractor runs
zero-cost and the caller can upgrade piecemeal.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex.audit.discover import (
    SchemaProposal,
    discover_schema,
    merge_proposal,
)
from nuggetindex.audit.seeds import SeedProposal, propose_seeds
from nuggetindex.core.schema import RelationSchema
from nuggetindex.extractors.cache import CachedExtractor
from nuggetindex.pipeline.constructor import Document
from nuggetindex.sidecar import Sidecar
from nuggetindex.store import NuggetStore

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.adapters.base import CorpusSource
    from nuggetindex.extractors.base import BaseExtractor


@dataclass(frozen=True)
class AutoReport:
    """Summary of what :func:`auto` did during a single call.

    Attributes
    ----------
    n_docs_processed:
        Number of documents ingested into the store. When
        ``two_pass_enabled`` is ``True`` this is the sum of Pass 1
        (bootstrap) + Pass 2 (seed-driven deep pull), minus dedup.
    schema_proposal_size:
        Count of predicates the discovery step proposed *beyond* the
        default schema. ``0`` when ``schema_discovery=False``.
    seed_budget:
        The seed-query budget passed by the caller.
    seeds_accepted:
        The number of seed queries the proposer actually returned
        (bounded above by ``seed_budget`` and by the candidate pool).
    nuggets_extracted:
        Total nuggets the ingest pipeline added to the store.
    contested_count:
        Number of nuggets that ended up in a contested (conflicting)
        state after ingest.
    rename_edges:
        Count of persisted nuggets whose predicate is ``renamedTo`` /
        ``corporateName`` — an at-a-glance signal that the corpus has
        entity-rename evidence.
    cost_est_usd:
        Rough cost estimate for the ingest run in USD. ``0.0`` for the
        trigger extractor (LLM-free path).
    cache_hit_rate:
        Fraction of cache hits over the ingest (``hits / total``). ``0.0``
        when the extractor wasn't wrapped in a cache.
    sidecar_mode:
        The mode the returned :class:`Sidecar` is configured in.
    rendered_markdown:
        Human-readable dump of the report.
    two_pass_enabled:
        ``True`` when :func:`auto` ran the optional seed-driven deep pull
        after the bootstrap pass.
    bootstrap_docs:
        Docs ingested during Pass 1 (``corpus.sample(...)`` output or the
        caller's ``docs=`` list). Equals ``n_docs_processed`` when
        ``two_pass_enabled`` is ``False``.
    deep_pass_docs:
        Docs ingested during Pass 2 (targeted ``corpus.search(seed)``
        pulls, de-duplicated against Pass 1). ``0`` when two-pass is off.
    nuggets_bootstrap:
        Nuggets added during the Pass 1 ingest.
    nuggets_deep_pass:
        Nuggets added during the Pass 2 ingest. ``0`` when two-pass is
        off.
    """

    n_docs_processed: int
    schema_proposal_size: int
    seed_budget: int
    seeds_accepted: int
    nuggets_extracted: int
    contested_count: int
    rename_edges: int
    cost_est_usd: float
    cache_hit_rate: float
    sidecar_mode: Literal["offline-curated", "just-in-time"]
    rendered_markdown: str
    two_pass_enabled: bool = False
    bootstrap_docs: int = 0
    deep_pass_docs: int = 0
    nuggets_bootstrap: int = 0
    nuggets_deep_pass: int = 0


# --------------------------------------------------------------------------- #
# Document loading helpers
# --------------------------------------------------------------------------- #


def _parse_source_date(raw: Any) -> datetime | None:
    """Lenient ISO-8601 parser (mirrors the CLI tools' behaviour)."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _load_jsonl(path: Path) -> list[Document]:
    """Load :class:`Document` records from a JSONL file.

    Skips blank lines; raises on invalid JSON so typos don't silently
    produce an under-indexed corpus. ``source_id`` + ``text`` are
    required; ``uri`` / ``source_date`` are optional.
    """
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)  # let JSONDecodeError bubble up
            source_id = row.get("source_id")
            text = row.get("text")
            if not source_id or not text:
                raise ValueError(f"{path}:{lineno}: missing required 'source_id' or 'text'")
            docs.append(
                Document(
                    source_id=str(source_id),
                    text=str(text),
                    uri=row.get("uri"),
                    source_date=_parse_source_date(row.get("source_date")),
                )
            )
    return docs


async def _materialise_docs(
    docs: AsyncIterable[Any] | Iterable[Any] | Path | str,
) -> list[Document]:
    """Normalise the ``docs`` argument into a concrete ``list[Document]``.

    Accepts:

    * :class:`Path` / ``str`` / any object with ``__fspath__`` pointing at
      a ``.jsonl`` file → load via :func:`_load_jsonl`.
    * :class:`AsyncIterable` / :class:`Iterable` of :class:`Document` or
      dict-shaped records → flatten into a list.
    """
    # Path-like branch — covers Path, str, os.PathLike subclasses.
    if isinstance(docs, (str, Path, PathLike)):
        return _load_jsonl(Path(docs))

    out: list[Document] = []
    if isinstance(docs, AsyncIterable):
        async for item in docs:
            out.append(_coerce_document(item))
        return out
    for item in docs:  # type: ignore[assignment]
        out.append(_coerce_document(item))
    return out


def _coerce_document(item: Any) -> Document:
    """Turn ``item`` into a :class:`Document`.

    Passes :class:`Document` instances through unchanged. Dict-like inputs
    are unpacked with the same field vocabulary as the JSONL loader.
    Anything else raises :class:`TypeError` with a helpful hint.
    """
    if isinstance(item, Document):
        return item
    if isinstance(item, dict):
        source_id = item.get("source_id")
        text = item.get("text")
        if not source_id or not text:
            raise ValueError("dict-shaped docs require 'source_id' and 'text' keys")
        return Document(
            source_id=str(source_id),
            text=str(text),
            uri=item.get("uri"),
            source_date=_parse_source_date(item.get("source_date")),
        )
    raise TypeError(f"auto() docs items must be Document or dict; got {type(item).__name__}")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_markdown(*, report: dict[str, Any]) -> str:
    """Compact audit-trail Markdown for :class:`AutoReport`."""
    lines: list[str] = []
    lines.append("# auto() report")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| docs processed | {report['n_docs_processed']} |")
    if report.get("two_pass_enabled"):
        lines.append(f"| bootstrap docs (pass 1) | {report.get('bootstrap_docs', 0)} |")
        lines.append(f"| deep-pass docs (pass 2) | {report.get('deep_pass_docs', 0)} |")
    lines.append(f"| schema predicates added | {report['schema_proposal_size']} |")
    lines.append(
        f"| seed budget / accepted | {report['seed_budget']} / {report['seeds_accepted']} |"
    )
    lines.append(f"| nuggets extracted | {report['nuggets_extracted']} |")
    if report.get("two_pass_enabled"):
        lines.append(
            f"| nuggets bootstrap / deep | "
            f"{report.get('nuggets_bootstrap', 0)} / "
            f"{report.get('nuggets_deep_pass', 0)} |"
        )
    lines.append(f"| contested nuggets | {report['contested_count']} |")
    lines.append(f"| rename edges | {report['rename_edges']} |")
    lines.append(f"| cost estimate (USD) | {report['cost_est_usd']:.4f} |")
    lines.append(f"| extractor cache hit rate | {report['cache_hit_rate']:.2f} |")
    lines.append(f"| sidecar mode | {report['sidecar_mode']} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


_RENAME_PREDICATES: frozenset[str] = frozenset({"renamedTo", "corporateName", "formerlyKnownAs"})


async def auto(
    docs: AsyncIterable[Any] | Iterable[Any] | Path | str | None = None,
    *,
    corpus: CorpusSource | None = None,
    bootstrap: Literal["caller", "topic_diverse", "uniform", "random_ids"] = "caller",
    budget: int = 100,
    sample_size: int = 500,
    mode: Literal["offline-curated", "just-in-time"] = "offline-curated",
    extractor: Any | None = None,
    store_path: Path | str = ".nuggetindex/store.db",
    cache_path: Path | str | None = ".nuggetindex/extractor-cache.db",
    schema_discovery: bool = True,
    two_pass: bool = False,
    deep_docs_per_seed: int = 10,
    deep_budget: int | None = None,
    verbose: bool = False,
) -> tuple[Sidecar, AutoReport]:
    """Build an end-to-end sidecar with one call.

    Parameters
    ----------
    docs:
        Concrete list / iterator / async iterator of :class:`Document` (or
        dict-shaped records), or a path to a ``.jsonl`` file. Mutually
        exclusive with ``corpus`` + ``bootstrap != "caller"``.
    corpus:
        A :class:`~nuggetindex.adapters.base.CorpusSource` (e.g.
        :class:`~nuggetindex.adapters.vespa.VespaCorpus`). When supplied
        together with a non-``"caller"`` ``bootstrap``, ``auto()`` samples
        the bootstrap set from the corpus itself instead of requiring the
        caller to curate ``docs``.
    bootstrap:
        Bootstrap-sampling strategy:

        * ``"caller"`` (default) -- legacy behaviour; caller provides ``docs``.
        * ``"topic_diverse"`` -- run the built-in topic-diverse query pack
          against the corpus. Recommended default for unknown-shape corpora.
        * ``"uniform"`` -- paginate with an empty-ish stopword query; slow
          but unbiased.
        * ``"random_ids"`` -- random doc IDs (backend support required).
    budget:
        Seed-query budget — the proposer returns at most this many.
    sample_size:
        Target size of the bootstrap sample drawn from ``corpus`` when
        ``bootstrap != "caller"``. Ignored otherwise.
    mode:
        Sidecar runtime mode. ``"offline-curated"`` queries the pre-built
        store; ``"just-in-time"`` defers extraction until retrieval time.
    extractor:
        Extractor instance. Defaults to :class:`TriggerExtractor`
        (zero-cost; patterns + NER).
    store_path:
        Location of the on-disk :class:`NuggetStore` file.
    cache_path:
        Location of the extractor cache file. Pass ``None`` to disable
        caching entirely.
    schema_discovery:
        When ``True`` (default) the corpus is scanned for schema
        additions; when ``False`` only the default schema is used.
    two_pass:
        When ``True`` (and ``corpus`` is provided), :func:`auto` runs a
        second ingest pass after discovery: the proposed seeds are sent
        back to ``corpus.search(seed, limit=deep_docs_per_seed)`` to pull
        targeted, topically-focused documents that the bootstrap sample
        likely missed. Those docs are de-duplicated against Pass 1 by
        ``source_id`` and ingested into the same store. Off by default.
    deep_docs_per_seed:
        Per-seed ``limit`` passed to ``corpus.search`` during Pass 2.
        Ignored when ``two_pass=False``.
    deep_budget:
        Optional cap on the number of seeds fanned out during Pass 2.
        ``None`` (default) means "use the same ``budget``". The seed
        proposer still operates under ``budget``; ``deep_budget`` just
        slices the returned seed list.
    verbose:
        Print a short progress summary to stderr. Off by default so
        library callers don't see unexpected output.

    Returns
    -------
    (Sidecar, AutoReport)
        A ready-to-query sidecar and an audit trail of what was done.
    """
    import sys

    # --- Step 0: materialise docs -----------------------------------------
    # Validate the docs/corpus/bootstrap combination before doing any work.
    if corpus is not None and bootstrap == "caller":
        raise ValueError(
            "auto(corpus=...) requires bootstrap != 'caller'; pass "
            "bootstrap='topic_diverse' (recommended) to sample from the "
            "corpus, or omit corpus= and provide docs=... directly."
        )
    if corpus is None and docs is None:
        raise ValueError(
            "auto() requires either docs=... or corpus=... with a non-'caller' bootstrap."
        )
    if two_pass and corpus is None:
        raise ValueError(
            "auto(two_pass=True) requires corpus=... -- Pass 2 issues "
            "corpus.search(seed) to pull deeper documents, which needs "
            "a live backend."
        )

    if corpus is not None and bootstrap != "caller":
        docs_list = await corpus.sample(mode=bootstrap, n=sample_size)
        if verbose:
            label = getattr(corpus, "corpus", type(corpus).__name__)
            print(
                f"auto: bootstrap sampled {len(docs_list)} doc(s) via {bootstrap} from {label}",
                file=sys.stderr,
            )
    else:
        # ``docs`` is guaranteed non-None here thanks to the guard above.
        docs_list = await _materialise_docs(docs)  # type: ignore[arg-type]
        if verbose:
            print(f"auto: loaded {len(docs_list)} document(s)", file=sys.stderr)

    # --- Step 1: resolve the extractor ------------------------------------
    # Default to the zero-cost trigger extractor so first-time users never
    # need an LLM key. Callers who want LLM quality pass their own.
    base_extractor: BaseExtractor
    if extractor is None:
        from nuggetindex.extractors.trigger import TriggerExtractor

        base_extractor = TriggerExtractor()
    else:
        base_extractor = extractor

    # --- Step 2: schema discovery (opt-in) --------------------------------
    schema = RelationSchema.default()
    schema_proposal_size = 0
    proposal: SchemaProposal | None = None
    if schema_discovery and docs_list:
        # Use the same extractor for schema discovery so domain-specific
        # predicates that only the LLM surfaces end up in the schema too.
        proposal = await discover_schema(
            docs=docs_list,
            extractor=base_extractor,
        )
        schema_proposal_size = len(proposal.predicates)
        schema = merge_proposal(schema, proposal, accept_all=True)
        if verbose:
            print(
                f"auto: schema discovery added {schema_proposal_size} predicate(s)",
                file=sys.stderr,
            )

    # --- Step 3: propose seeds (informational in the report) --------------
    seed_proposal: SeedProposal | None = None
    seeds_accepted = 0
    if docs_list:
        seed_proposal = await propose_seeds(
            docs=docs_list,
            schema=schema,
            budget=budget,
        )
        seeds_accepted = len(seed_proposal.seeds)
        if verbose:
            print(
                f"auto: proposed {seeds_accepted} seed(s) (budget={budget})",
                file=sys.stderr,
            )

    # --- Step 4: wrap the extractor in a cache (if a path was given) ------
    store_extractor: Any = base_extractor
    cached: CachedExtractor | None = None
    if cache_path is not None:
        cached = CachedExtractor(inner=base_extractor, cache_path=cache_path)
        store_extractor = cached

    # --- Step 5: build the store + ingest Pass 1 --------------------------
    store_path_obj = Path(store_path)
    store_path_obj.parent.mkdir(parents=True, exist_ok=True)
    store = NuggetStore(
        db_path=store_path_obj,
        schema=schema,
        extractor=store_extractor,
    )

    (
        nuggets_bootstrap,
        contested_bootstrap,
        bootstrap_failures,
    ) = await _ingest_docs(store, docs_list)
    if verbose:
        msg = f"auto: ingested {nuggets_bootstrap} nugget(s); {contested_bootstrap} contested"
        if bootstrap_failures:
            msg += f"; {bootstrap_failures} doc(s) failed extraction"
        print(msg, file=sys.stderr)

    # --- Step 5b: Pass 2 -- targeted deep pull via seeds ------------------
    deep_docs: list[Document] = []
    nuggets_deep = 0
    contested_deep = 0
    deep_failures = 0
    if two_pass and corpus is not None and seed_proposal is not None and seed_proposal.seeds:
        seen_source_ids = {d.source_id for d in docs_list}
        effective_deep_budget = deep_budget if deep_budget is not None else budget
        seeds_for_pass2 = seed_proposal.seeds[: max(0, effective_deep_budget)]
        for seed in seeds_for_pass2:
            try:
                hits = await corpus.search(seed.query, limit=max(1, deep_docs_per_seed))
            except Exception:  # noqa: BLE001 -- best-effort per-seed search
                continue
            for hit in hits:
                if hit.source_id in seen_source_ids:
                    continue
                seen_source_ids.add(hit.source_id)
                deep_docs.append(hit)
        if verbose:
            print(
                f"auto: pass 2 pulled {len(deep_docs)} new doc(s) across "
                f"{len(seeds_for_pass2)} seed(s)",
                file=sys.stderr,
            )
        (
            nuggets_deep,
            contested_deep,
            deep_failures,
        ) = await _ingest_docs(store, deep_docs)
        if verbose:
            msg = f"auto: pass 2 ingested {nuggets_deep} nugget(s); {contested_deep} contested"
            if deep_failures:
                msg += f"; {deep_failures} doc(s) failed extraction"
            print(msg, file=sys.stderr)

    nuggets_extracted = nuggets_bootstrap + nuggets_deep
    contested_count = contested_bootstrap + contested_deep
    total_docs_processed = len(docs_list) + len(deep_docs)

    # --- Step 6: rename edges + cost + cache stats ------------------------
    rename_edges = await _count_rename_edges(store)
    cache_hit_rate = 0.0
    if cached is not None:
        stats = cached.stats()
        total = stats.get("total", 0)
        if total:
            cache_hit_rate = stats.get("hits", 0) / total

    cost_est_usd = await _estimate_cost(
        docs_list + deep_docs,
        extractor=base_extractor,
    )

    # --- Step 7: assemble the Sidecar -------------------------------------
    # just-in-time mode requires an extractor; we reuse the (cached) one.
    sidecar_kwargs: dict[str, Any] = {"store": store, "mode": mode}
    if mode == "just-in-time":
        sidecar_kwargs["extractor"] = store_extractor
    else:
        # offline-curated still benefits from having an extractor available
        # in case callers attach a JIT fallback later; harmless when unused.
        sidecar_kwargs["extractor"] = store_extractor
    sidecar = Sidecar(**sidecar_kwargs)

    # --- Step 8: render + return ------------------------------------------
    report_payload: dict[str, Any] = {
        "n_docs_processed": total_docs_processed,
        "schema_proposal_size": schema_proposal_size,
        "seed_budget": budget,
        "seeds_accepted": seeds_accepted,
        "nuggets_extracted": nuggets_extracted,
        "contested_count": contested_count,
        "rename_edges": rename_edges,
        "cost_est_usd": cost_est_usd,
        "cache_hit_rate": cache_hit_rate,
        "sidecar_mode": mode,
        "two_pass_enabled": two_pass,
        "bootstrap_docs": len(docs_list),
        "deep_pass_docs": len(deep_docs),
        "nuggets_bootstrap": nuggets_bootstrap,
        "nuggets_deep_pass": nuggets_deep,
    }
    rendered = _render_markdown(report=report_payload)
    report = AutoReport(
        n_docs_processed=total_docs_processed,
        schema_proposal_size=schema_proposal_size,
        seed_budget=budget,
        seeds_accepted=seeds_accepted,
        nuggets_extracted=nuggets_extracted,
        contested_count=contested_count,
        rename_edges=rename_edges,
        cost_est_usd=cost_est_usd,
        cache_hit_rate=cache_hit_rate,
        sidecar_mode=mode,
        rendered_markdown=rendered,
        two_pass_enabled=two_pass,
        bootstrap_docs=len(docs_list),
        deep_pass_docs=len(deep_docs),
        nuggets_bootstrap=nuggets_bootstrap,
        nuggets_deep_pass=nuggets_deep,
    )
    return sidecar, report


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


async def _ingest_docs(
    store: NuggetStore,
    docs: list[Document],
) -> tuple[int, int, int]:
    """Ingest ``docs`` into ``store`` defensively.

    Returns ``(nuggets_added, conflicts_detected, ingest_failures)``. LLM
    extractors occasionally return malformed triples (e.g. empty object
    strings) that violate ``FactTriple``'s Pydantic invariants; one failed
    doc shouldn't abort the whole auto-run, so we count the failure and
    keep going (mirrors ``audit_scanner.ingest_scenarios``' defensive
    pattern).
    """
    nuggets_added = 0
    conflicts = 0
    failures = 0
    for doc in docs:
        try:
            result = await store.aingest(doc)
            nuggets_added += result.nuggets_added
            conflicts += result.conflicts_detected
        except Exception:  # noqa: BLE001 -- best-effort per-doc ingest
            failures += 1
    return nuggets_added, conflicts, failures


async def _count_rename_edges(store: NuggetStore) -> int:
    """Count persisted nuggets whose predicate is a rename-edge predicate."""
    count = 0
    backend = store.backend
    try:
        source_ids = await backend.alist_source_ids()
    except Exception:  # pragma: no cover -- defensive
        return 0
    for sid in source_ids:
        try:
            nuggets = await backend.aget_nuggets_by_source(sid)
        except Exception:  # pragma: no cover -- defensive
            continue
        for n in nuggets:
            if n.fact.predicate in _RENAME_PREDICATES:
                count += 1
    return count


def _resolve_model_id(extractor: Any) -> str:
    """Return the canonical ``model_id`` for :func:`estimate_ingest_cost`.

    * ``None``                   -> ``"trigger"`` (auto() default extractor;
      auto() swaps None for ``TriggerExtractor`` internally, so they must
      resolve the same way).
    * :class:`TriggerExtractor`  -> ``"trigger"`` (LLM-free; priced at 0).
    * :class:`LLMExtractor`      -> ``extractor.cfg.model``.
    * :class:`CachedExtractor`   -> recurse into the inner extractor so
      wrapping an LLM in a cache still charges the right price table.
    * Anything else              -> fall back to ``"gpt-4o-mini"`` (the
      conservative-but-calibrated default the estimator used historically).
    """
    # None -> TriggerExtractor is auto()'s internal default.
    if extractor is None:
        return "trigger"

    # TriggerExtractor -> canonical "trigger" id (priced at 0).
    try:
        from nuggetindex.extractors.trigger import TriggerExtractor

        if isinstance(extractor, TriggerExtractor):
            return "trigger"
    except Exception:  # pragma: no cover -- first-party import is stable
        pass

    # CachedExtractor -> recurse into the inner extractor (or read the
    # auto-inferred extractor_id off the cache wrapper if we can't reach
    # the inner).
    try:
        from nuggetindex.extractors.cache import CachedExtractor

        if isinstance(extractor, CachedExtractor):
            inner = getattr(extractor, "_inner", None)
            if inner is not None:
                return _resolve_model_id(inner)
            # Fall back: strip the "llm:provider:" prefix from extractor_id
            # ("llm:openai:gpt-4o-mini:v1" -> "gpt-4o-mini"). Best-effort.
            ext_id = getattr(extractor, "extractor_id", "") or ""
            if ext_id.startswith("llm:"):
                parts = ext_id.split(":")
                if len(parts) >= 3:
                    return parts[2]
            if ext_id.startswith("trigger:"):
                return "trigger"
    except Exception:  # pragma: no cover -- first-party import is stable
        pass

    # LLMExtractor (or anything exposing ``.cfg.model``).
    cfg = getattr(extractor, "cfg", None)
    if cfg is not None:
        model = getattr(cfg, "model", None)
        if model:
            return str(model)

    return "gpt-4o-mini"


async def _estimate_cost(
    docs_list: list[Document],
    *,
    extractor: Any,
) -> float:
    """Rough cost estimate routed through :func:`estimate_ingest_cost`.

    The caller passes the actual ``extractor`` (the one the user asked
    :func:`auto` to run) and we thread its canonical ``model_id`` into
    :func:`estimate_ingest_cost`. The previous revision silently ignored
    the user's model and defaulted to ``"gpt-4o-mini"``, which meant
    ``AutoReport.cost_est_usd`` always reflected ``gpt-4o-mini`` pricing
    even when the user had wired in ``gpt-4o`` or a Claude model. For
    the trigger (LLM-free) extractor the table prices the run at $0 and
    the estimator returns ``0.0`` without touching an LLM.

    Best-effort: any failure returns ``0.0`` rather than bubbling up.
    """
    model_id = _resolve_model_id(extractor)
    try:
        from nuggetindex.audit.cost import estimate_ingest_cost

        # Intentionally pass ``cache_path=None`` — by the time we compute the
        # cost for the AutoReport the cache has been fully populated by this
        # run's ingest, so probing it would report 100% hits and a net cost of
        # $0 even for an expensive run. The caller's more honest signal is the
        # *gross* cost: "what would this ingest cost on a fresh machine?"
        # The current-run cache-hit rate is already surfaced separately in the
        # AutoReport, so the two fields together tell the full story.
        est = await estimate_ingest_cost(
            docs=docs_list,
            model_id=model_id,
            cache_path=None,
        )
        return float(est.total_cost_usd_est)
    except Exception:  # pragma: no cover -- best-effort
        return 0.0


__all__ = ["AutoReport", "auto"]
