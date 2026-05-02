"""NuggetStore public class + backend protocols.

NuggetStore is the single class users touch. It delegates to pluggable
backends for sparse retrieval, dense retrieval, and metadata storage.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from nuggetindex.chains.models import (
    ChainEdge,
    ChainEdgeType,
    NuggetChain,
)
from nuggetindex.core.enums import LifecycleStatus
from nuggetindex.core.errors import ChainAmbiguousError
from nuggetindex.core.models import Nugget
from nuggetindex.core.schema import RelationSchema

if TYPE_CHECKING:
    from nuggetindex.store.backends.sqlite import SQLiteBackend

ViewMode = Literal["active", "active_contested", "all"]


def _require_no_running_loop(method_name: str, async_name: str) -> None:
    """Raise if a sync wrapper is called from inside a running event loop.

    Gives a clearer error than the default
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``
    by naming the async equivalent the caller should use instead.
    (findings-A4)
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"NuggetStore.{method_name}() is synchronous and cannot be called "
        f"from inside a running event loop. Use {async_name}() instead."
    )


@dataclass
class AddResult:
    nugget: Nugget
    created: bool  # True if new row, False if existing provenance merged
    duplicates_merged: int = 0
    conflicts_detected: int = 0


@dataclass
class IngestResult:
    document_id: str
    nuggets_added: int
    nuggets_merged: int
    conflicts_detected: int


@dataclass
class DiffReport:
    added_ids: list[str] = field(default_factory=list)
    deprecated_ids: list[str] = field(default_factory=list)
    contested_now_resolved: list[str] = field(default_factory=list)
    key_changes: list[tuple[str, str]] = field(default_factory=list)  # (key, description)


class MetadataBackend(Protocol):
    """Storage for structured nugget records and provenance."""

    async def aupsert(self, nugget: Nugget) -> None: ...
    async def aget(self, nugget_id: str) -> Nugget | None: ...
    async def afind_by_key(self, key: tuple[str, str, str]) -> list[Nugget]: ...
    async def aget_nuggets_by_source(self, source_id: str) -> list[Nugget]: ...
    async def afilter(
        self,
        *,
        query_time: datetime,
        view: ViewMode,
        extra_filters: dict[str, Any] | None = None,
    ) -> list[str]: ...
    async def acount(self, status: LifecycleStatus | None = None) -> int: ...
    async def aupsert_passage(self, source_id: str, uri: str | None, text: str) -> None: ...
    async def aget_passages(self, source_ids: Iterable[str]) -> dict[str, str]: ...
    async def acount_passages(self) -> int: ...
    async def alist_source_ids(self) -> list[str]: ...
    async def aget_passage_records(
        self, source_ids: Iterable[str]
    ) -> dict[str, tuple[str, str | None]]: ...
    async def adelete_by_source_ids(self, ids: list[str]) -> None: ...
    async def aupsert_passage_with_meta(
        self,
        source_id: str,
        uri: str | None,
        text: str,
        meta_json: str | None,
    ) -> None: ...
    async def apassage_exists(self, source_id: str) -> bool: ...
    async def asuccession_for_key(
        self,
        key: str,
        as_of: datetime | None,
        statuses: list[str],
        limit: int,
    ) -> list[Nugget]: ...
    async def arename_candidates(
        self,
        *,
        subject: str,
        as_of: datetime | None,
        renaming_predicates: frozenset[str],
        direction: str = "forward",
        include_contested: bool = False,
        limit: int = 3,
    ) -> list[Nugget]: ...
    async def acandidate_keys(
        self,
        *,
        subject_contains: str | None = None,
        predicate_contains: str | None = None,
        scope: str = "global",
        limit: int = 20,
    ) -> list[tuple[str, str, str]]: ...
    async def acontested_keys(self) -> list[tuple[str, str, str, int]]: ...
    async def aclose(self) -> None: ...


class NuggetStore:
    """The single public class for nugget storage and retrieval.

    Pluggable internals:
      - metadata/sparse backend: SQLite by default
      - dense backend: optional (None = sparse-only, 92% of hybrid recall)
      - schema: RelationSchema (functional vs multi-valued predicates)
      - judge: optional LLMJudge for ambiguous conflicts (Improvement C)
      - quality_gate: optional QualityGate for extraction (Improvement D)
    """

    def __init__(
        self,
        db_path: Path | str = "nuggetindex.db",
        *,
        schema: RelationSchema | None = None,
        dense: Any | None = None,
        encoder: Any | None = None,
        judge: Any | None = None,
        quality_gate: Any | None = None,
        extractor: Any | None = None,
    ) -> None:
        # Import here to avoid circular imports between base and backends.
        from nuggetindex.store.backends.sqlite import SQLiteBackend

        self.db_path = Path(db_path)
        self._backend_impl: SQLiteBackend = SQLiteBackend(self.db_path)
        self.schema = schema or RelationSchema.default()
        self.dense = dense
        self.encoder = encoder
        self.judge = judge
        self.quality_gate = quality_gate
        self._extractor = extractor
        # DocumentConstructor is built lazily on first aingest call so users
        # without an extractor don't pay the import cost.
        self._constructor: Any | None = None
        # Retriever is also built lazily on first aretrieve call; the dense
        # backend wiring can change (e.g. attached post-init) before the
        # first query, and we want to pick up the final state.
        self._retriever: Any | None = None
        # Store-scoped alias resolver (fix 10). Lazy-initialised on the
        # first aingest call and seeded from the backend's existing
        # subjects/objects so "Microsoft" in doc A and "Microsoft
        # Corporation" in doc B collapse to a single canonical. Safe to
        # leave un-locked: aingest writes are serialised through the
        # SQLite backend's writer queue (see ``backends/sqlite.py``),
        # so the resolver is only ever touched from one coroutine at a
        # time.
        self._alias_resolver: Any | None = None

    # --- Backend accessor ---

    @property
    def backend(self) -> MetadataBackend:
        """Public handle to the metadata/sparse backend.

        Use this instead of the deprecated ``store._backend`` dunder-name
        access. The deprecated form remains available in 0.2 with a
        :class:`DeprecationWarning` and will be removed in 0.3.
        """
        return self._backend_impl

    @property
    def _backend(self) -> MetadataBackend:
        """Deprecated alias for :pyattr:`backend`. Removed in 0.3."""
        import warnings

        warnings.warn(
            "NuggetStore._backend is deprecated; use store.backend instead. "
            "The dunder alias will be removed in nuggetindex 0.3.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._backend_impl

    # --- Async primary ---

    async def aadd(self, nugget: Nugget) -> AddResult:
        existing = await self._backend_impl.aget(nugget.id)
        await self._backend_impl.aupsert(nugget)
        return AddResult(nugget=nugget, created=existing is None)

    async def aget(self, nugget_id: str) -> Nugget | None:
        return await self._backend_impl.aget(nugget_id)

    async def amark_preferred(
        self,
        nugget_id: str,
        *,
        set_active: bool = True,
    ) -> Nugget:
        """Pin ``nugget_id`` as the canonical winner of its key.

        Used for human-in-the-loop resolution of ``Contested`` facts:
        sets ``rank=Preferred`` and (by default) ``status=Active``. Use
        :meth:`asuppress` on the rivals to ensure the LLM stops seeing
        the disagreement.

        Returns the updated nugget. The original row is replaced via
        ``aupsert`` so provenance and validity are preserved unchanged.
        """
        from nuggetindex.core.enums import EpistemicRank, LifecycleStatus
        from nuggetindex.core.models import EpistemicState

        existing = await self._backend_impl.aget(nugget_id)
        if existing is None:
            raise KeyError(f"nugget not found: {nugget_id!r}")
        new_status = LifecycleStatus.ACTIVE if set_active else existing.epistemic.status
        new_state = EpistemicState(
            status=new_status,
            rank=EpistemicRank.PREFERRED,
            confidence=existing.epistemic.confidence,
        )
        updated = existing.model_copy(
            update={
                "epistemic": new_state,
                "updated_at": datetime.now(existing.updated_at.tzinfo or UTC),
            }
        )
        await self._backend_impl.aupsert(updated)
        return updated

    async def asuppress(self, nugget_id: str) -> Nugget:
        """Mark ``nugget_id`` as a suppressed loser of a contested key.

        Sets ``status=Deprecated`` and ``rank=Deprecated``. Provenance is
        preserved, the row is not hard-deleted; the suppressed nugget
        simply drops out of the ``active`` and ``active_contested``
        retrieval views.

        Returns the updated nugget.
        """
        from nuggetindex.core.enums import EpistemicRank, LifecycleStatus
        from nuggetindex.core.models import EpistemicState

        existing = await self._backend_impl.aget(nugget_id)
        if existing is None:
            raise KeyError(f"nugget not found: {nugget_id!r}")
        new_state = EpistemicState(
            status=LifecycleStatus.DEPRECATED,
            rank=EpistemicRank.DEPRECATED,
            confidence=existing.epistemic.confidence,
        )
        updated = existing.model_copy(
            update={
                "epistemic": new_state,
                "updated_at": datetime.now(existing.updated_at.tzinfo or UTC),
            }
        )
        await self._backend_impl.aupsert(updated)
        return updated

    async def acontested_keys(self) -> list[tuple[str, str, str, int]]:
        """List ``(subject, predicate, scope, n_contested)`` for every key
        with at least one ``Contested`` nugget, ordered by descending
        contest size. Used by the ``nuggetindex resolve`` CLI."""
        return await self._backend_impl.acontested_keys()

    async def acount(self, *, status: LifecycleStatus | None = None) -> int:
        return await self._backend_impl.acount(status)

    async def aget_source_passages(self, nuggets: list[Nugget]) -> dict[str, str]:
        ids: set[str] = set()
        for n in nuggets:
            for p in n.provenance:
                ids.add(p.source_id)
        return await self._backend_impl.aget_passages(ids)

    async def _ensure_alias_resolver(self) -> Any:
        """Lazy-init the store-scoped :class:`AliasResolver` (fix 10).

        On the first call we instantiate a fresh resolver with default
        config and pre-seed its canonical pool from every distinct
        subject and object string currently stored in the backend.
        That way the first doc ingested after the store is opened can
        immediately collapse its surface forms against anything the
        backend already knows about. Subsequent calls are cheap: the
        already-built resolver is returned as-is and new mentions
        accumulate into its pool as docs flow through.

        Backends that don't expose ``adistinct_entities`` (e.g. a
        future non-SQL backend that hasn't opted in yet) simply get an
        empty pool; the resolver still works, just without seeding.
        """
        from nuggetindex.pipeline.aliases import AliasResolver

        if self._alias_resolver is not None:
            return self._alias_resolver
        resolver = AliasResolver()
        distinct_fn = getattr(self._backend_impl, "adistinct_entities", None)
        if distinct_fn is not None:
            mentions = await distinct_fn()
            # Seed via resolve() so the normalized-lookup and TF-IDF
            # internals are populated identically to the runtime path.
            for m in mentions:
                resolver.resolve(m)
        self._alias_resolver = resolver
        return resolver

    async def aingest(self, doc: Any) -> IngestResult:
        """Run the full pipeline on ``doc`` and persist the resulting nuggets.

        Requires ``extractor=`` to have been passed at init. Delegates to a
        lazily-built ``DocumentConstructor`` that wires the four pipeline
        stages; cross-document conflict detection happens inside
        ``DocumentConstructor.aprocess`` via the
        ``fetch_existing_by_key=self._backend_impl.afind_by_key`` callback so that
        peers already persisted from earlier ``aingest`` calls are visible.

        A store-scoped :class:`AliasResolver` is built lazily and threaded
        into ``aprocess`` so its canonical pool accumulates across every
        document ingested through this store (fix 10). The pool is seeded
        on first call from the backend's existing subjects/objects so
        aliases work immediately after re-opening an existing store.

        Each persisted nugget is upserted via ``aupsert`` which handles its
        own per-row transaction; partial failures leave the store in a
        consistent (though possibly incomplete) state. Running ``aingest``
        twice with the same ``doc`` is idempotent because content-hashed IDs
        collapse repeated facts to the same row.
        """
        from nuggetindex.pipeline.conflict import ConflictDetector
        from nuggetindex.pipeline.constructor import DocumentConstructor
        from nuggetindex.pipeline.dedup import Deduplicator

        if self._extractor is None:
            raise RuntimeError(
                "NuggetStore has no extractor configured. Pass extractor= at init "
                "or use aadd() to insert pre-built nuggets."
            )

        if self._constructor is None:
            self._constructor = DocumentConstructor(
                extractor=self._extractor,
                schema=self.schema,
                deduplicator=Deduplicator(encoder=self.encoder),
                conflict_detector=ConflictDetector(self.schema, judge=self.judge),
                quality_gate=self.quality_gate,
            )

        # Persist the source passage for two-tier retrieval.
        await self._backend_impl.aupsert_passage(doc.source_id, doc.uri, doc.text)

        alias_resolver = await self._ensure_alias_resolver()
        result_nuggets = await self._constructor.aprocess(
            doc,
            fetch_existing_by_key=self._backend_impl.afind_by_key,
            alias_resolver=alias_resolver,
        )

        added = merged = conflicts = 0
        for n in result_nuggets:
            prior = await self._backend_impl.aget(n.id)
            await self._backend_impl.aupsert(n)
            if prior is None:
                added += 1
            else:
                merged += 1
            if n.epistemic.status == LifecycleStatus.CONTESTED:
                conflicts += 1

        return IngestResult(
            document_id=doc.source_id,
            nuggets_added=added,
            nuggets_merged=merged,
            conflicts_detected=conflicts,
        )

    async def aretrieve(
        self,
        query: str,
        *,
        query_time: datetime | None = None,
        view: str = "active",
        top_k: int = 20,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Run the fusion retrieval pipeline.

        Delegates to a lazily-built ``Retriever`` that wires together the
        SQLite backend's view filter + BM25 search with an optional dense
        backend. Returns ``RetrievalResult`` entries sorted by fused score;
        when no candidates survive the view filter, the result is an empty
        list.
        """
        from nuggetindex.retrieve.retriever import (
            FusionMode,
            RetrievalResult,
            Retriever,
        )

        if self._retriever is None:
            self._retriever = Retriever(
                backend=self._backend_impl,
                dense_backend=self.dense,
            )
        # Cast to satisfy the Literal FusionMode param without forcing callers
        # to import the type.
        mode: FusionMode = fusion  # type: ignore[assignment]
        results: list[RetrievalResult] = await self._retriever.aretrieve(
            query,
            query_time=query_time,
            view=view,
            top_k=top_k,
            fusion=mode,
            filters=filters,
        )
        return results

    # --- Chain methods (v0.2) ---

    async def achain_succession(
        self,
        *,
        subject: str,
        predicate: str,
        scope: str = "global",
        as_of: datetime | None = None,
        include_contested: bool = False,
        max_depth: int = 50,
    ) -> NuggetChain:
        """Return the ordered history of values for ``(subject, predicate, scope)``.

        Pure SQL: ``SELECT`` by key, ``ORDER BY validity_start``. Returns
        ``ACTIVE`` and ``DEPRECATED`` nuggets by default; set
        ``include_contested=True`` to also surface ``CONTESTED`` entries.

        When ``as_of`` is set, only nuggets with ``validity_start <= as_of``
        appear. If the chain contains more than ``max_depth`` entries it is
        truncated to ``max_depth`` and ``NuggetChain.truncated`` is set.

        The predicate is canonicalised via ``self.schema.canonicalize`` before
        the SQL lookup, so passing an alias (``"ceo"``) matches nuggets keyed
        under the canonical name (``"chiefExecutiveOfficer"``). (findings-A3)
        """
        predicate = self.schema.canonicalize(predicate)
        key = f"{subject}|{predicate}|{scope}"
        statuses = ["active", "deprecated"]
        if include_contested:
            statuses.append("contested")
        rows = await self._backend_impl.asuccession_for_key(key, as_of, statuses, max_depth + 1)
        truncated = len(rows) > max_depth
        nuggets = tuple(rows[:max_depth] if truncated else rows)
        edges: list[ChainEdge] = []
        for i in range(len(nuggets) - 1):
            prev, nxt = nuggets[i], nuggets[i + 1]
            gap: timedelta | None = None
            if prev.validity.end is not None:
                gap = nxt.validity.start - prev.validity.end
                if gap.total_seconds() < 0:
                    # Overlapping intervals: don't surface a negative gap.
                    gap = None
            edges.append(
                ChainEdge(
                    from_idx=i,
                    to_idx=i + 1,
                    edge_type=ChainEdgeType.SUCCEEDS,
                    gap=gap,
                )
            )
        return NuggetChain(
            nuggets=nuggets,
            edges=tuple(edges),
            chain_type="succession",
            as_of=as_of,
            truncated=truncated,
        )

    async def achain_rename(
        self,
        *,
        subject: str,
        as_of: datetime | None = None,
        direction: Literal["forward", "backward", "both"] = "forward",
        max_depth: int = 10,
        include_contested: bool = False,
        resolver: Any | None = None,
        strict: bool = False,
    ) -> NuggetChain:
        """Walk the renaming-predicate graph starting at ``subject``.

        ``direction="forward"`` follows ``object -> subject`` transitions
        ("what is X known as now?"); ``"backward"`` walks in reverse ("what
        was X formerly called?"); ``"both"`` merges predecessors and
        successors into a single timeline ordered by validity.

        Ambiguous steps (multiple candidates at a single hop) either delegate
        to ``resolver`` or raise :class:`ChainAmbiguousError` when
        ``resolver`` is ``None``. Cycles in the rename graph terminate the
        walk cleanly via a seen-set.

        When ``strict=True`` the walk consumes
        :pyattr:`RelationSchema.entity_rename_predicates` instead of the
        broader :pyattr:`RelationSchema.renaming_predicates`. This excludes
        any predicate that is not in the library-level entity-rename
        whitelist (``renamedTo``, ``formerlyKnownAs``, ``corporateName``),
        so role-succession predicates like ``succeededBy`` / ``precededBy``
        never drive an entity-rename walk even if a user-supplied schema
        marks them ``renaming: true``. ``strict=False`` (default) preserves
        the historical behaviour.
        """
        renaming = (
            self.schema.entity_rename_predicates if strict else self.schema.renaming_predicates
        )
        truncated = False

        if direction in ("forward", "both"):
            fwd_raw, fwd_truncated = await self._walk_rename(
                subject=subject,
                as_of=as_of,
                direction="forward",
                renaming_predicates=renaming,
                max_depth=max_depth,
                include_contested=include_contested,
                resolver=resolver,
            )
            if fwd_truncated:
                truncated = True
        else:
            fwd_raw = []

        if direction in ("backward", "both"):
            bwd_raw, bwd_truncated = await self._walk_rename(
                subject=subject,
                as_of=as_of,
                direction="backward",
                renaming_predicates=renaming,
                max_depth=max_depth,
                include_contested=include_contested,
                resolver=resolver,
            )
            if bwd_truncated:
                truncated = True
        else:
            bwd_raw = []

        if direction == "both":
            combined = list(bwd_raw) + list(fwd_raw)
            # De-dup by id while preserving first occurrence
            seen_ids: set[str] = set()
            unique: list[Nugget] = []
            for n in combined:
                if n.id in seen_ids:
                    continue
                seen_ids.add(n.id)
                unique.append(n)
            ordered = sorted(unique, key=lambda n: n.validity.start)
        elif direction == "backward":
            # Backward walks produce a chain ordered by decreasing time;
            # reverse so the output reads chronologically.
            ordered = list(reversed(bwd_raw))
        else:
            ordered = list(fwd_raw)

        edges: list[ChainEdge] = []
        for i in range(len(ordered) - 1):
            edges.append(
                ChainEdge(
                    from_idx=i,
                    to_idx=i + 1,
                    edge_type=ChainEdgeType.RENAMES_TO,
                    gap=None,
                )
            )
        return NuggetChain(
            nuggets=tuple(ordered),
            edges=tuple(edges),
            chain_type="rename",
            as_of=as_of,
            truncated=truncated,
        )

    async def _walk_rename(
        self,
        *,
        subject: str,
        as_of: datetime | None,
        direction: Literal["forward", "backward"],
        renaming_predicates: frozenset[str],
        max_depth: int,
        include_contested: bool,
        resolver: Any | None,
    ) -> tuple[list[Nugget], bool]:
        """Return ``(nuggets, truncated)`` for a single-direction walk."""
        if not renaming_predicates:
            return [], False
        seen: set[str] = {subject}
        current = subject
        nuggets: list[Nugget] = []
        truncated = False
        for step in range(max_depth):
            candidates = await self._backend_impl.arename_candidates(
                subject=current,
                as_of=as_of,
                renaming_predicates=renaming_predicates,
                direction=direction,
                include_contested=include_contested,
                limit=3,
            )
            if not candidates:
                break
            if len(candidates) > 1:
                if resolver is None:
                    raise ChainAmbiguousError(subject=current, candidates=candidates, step=step)
                resolution = await resolver.adisambiguate(
                    candidates=candidates,
                    context=(f"rename walk ({direction}) from {current!r} at as_of={as_of}"),
                )
                picked = resolution.picked
            else:
                picked = candidates[0]
            nuggets.append(picked)
            next_subject = picked.fact.object if direction == "forward" else picked.fact.subject
            if next_subject in seen:
                break  # cycle guard
            seen.add(next_subject)
            current = next_subject
        else:
            # Loop completed max_depth iterations without breaking; we might
            # still have more to walk, which counts as truncated. Probe once
            # more: if no further candidate exists, we're actually done.
            if nuggets:
                more = await self._backend_impl.arename_candidates(
                    subject=current,
                    as_of=as_of,
                    renaming_predicates=renaming_predicates,
                    direction=direction,
                    include_contested=include_contested,
                    limit=1,
                )
                if more and more[0].id != nuggets[-1].id:
                    truncated = True
        return nuggets, truncated

    async def achain_join(
        self,
        *,
        start: tuple[str, str],
        then: list[str],
        scope: str = "global",
        as_of: datetime | None = None,
        resolver: Any | None = None,
    ) -> NuggetChain:
        """Bounded 1--3 hop functional temporal join.

        Example::

            start=("Google", "parentCompany"),
            then=["ceo"],
            as_of=datetime(2020, 1, 1, tzinfo=UTC)

        returns a 2-nugget chain: Google's parent at 2020 + that parent's CEO
        at 2020.

        Each hop must resolve to exactly one nugget under ``(as_of,
        not-DEPRECATED)``. Zero or multiple candidates either delegate to
        ``resolver`` or raise :class:`ChainAmbiguousError`.
        """
        if len(then) > 3:
            raise ValueError(f"max_hops guardrail: len(then) must be <= 3 (got {len(then)})")
        subject, predicate = start
        # Canonicalise the start predicate and every ``then`` hop so aliases
        # (``"ceo"``) match nuggets keyed under the canonical predicate
        # (``"chiefExecutiveOfficer"``). (findings-A3)
        predicate = self.schema.canonicalize(predicate)
        then = [self.schema.canonicalize(p) for p in then]
        first = await self._get_functional_at(
            subject=subject,
            predicate=predicate,
            scope=scope,
            as_of=as_of,
            resolver=resolver,
            step=0,
        )
        hops: list[Nugget] = [first]
        bound = first.fact.object
        for i, pred in enumerate(then):
            nxt = await self._get_functional_at(
                subject=bound,
                predicate=pred,
                scope=scope,
                as_of=as_of,
                resolver=resolver,
                step=i + 1,
            )
            hops.append(nxt)
            bound = nxt.fact.object

        edges = tuple(
            ChainEdge(
                from_idx=i,
                to_idx=i + 1,
                edge_type=ChainEdgeType.OBJECT_IS_SUBJECT,
                gap=None,
            )
            for i in range(len(hops) - 1)
        )
        return NuggetChain(
            nuggets=tuple(hops),
            edges=edges,
            chain_type="joined",
            as_of=as_of,
            truncated=False,
        )

    async def _get_functional_at(
        self,
        *,
        subject: str,
        predicate: str,
        scope: str = "global",
        as_of: datetime | None,
        resolver: Any | None,
        step: int = -1,
    ) -> Nugget:
        """Return the single non-DEPRECATED functional nugget valid at ``as_of``.

        If zero or multiple candidates are eligible, either delegate to
        ``resolver`` (when supplied) or raise :class:`ChainAmbiguousError`.
        """
        # Match the raw predicate — v0.1 ``afind_by_key`` compares on the
        # stored key which uses whatever predicate the caller supplied.
        nuggets = await self._backend_impl.afind_by_key((subject, predicate, scope))
        eligible = [
            n
            for n in nuggets
            if n.epistemic.status != LifecycleStatus.DEPRECATED
            and (as_of is None or n.validity.contains(as_of))
        ]
        if len(eligible) == 1:
            return eligible[0]
        if resolver is None:
            raise ChainAmbiguousError(subject=subject, candidates=eligible, step=step)
        if not eligible:
            raise ChainAmbiguousError(subject=subject, candidates=[], step=step)
        resolution = await resolver.adisambiguate(
            candidates=eligible,
            context=f"join hop {subject}.{predicate}@{as_of}",
        )
        picked: Nugget = resolution.picked
        return picked

    # --- Discovery (v0.2.1) ---

    async def acandidate_keys(
        self,
        *,
        subject_contains: str | None = None,
        predicate_contains: str | None = None,
        scope: str = "global",
        limit: int = 20,
    ) -> list[tuple[str, str, str]]:
        """Return distinct ``(subject, predicate, scope)`` triples matching filters.

        Thin pass-through to :meth:`MetadataBackend.acandidate_keys`.
        Case-insensitive substring match on subject and/or predicate.
        Useful for discovering how a store is keyed before an exact-match
        chain lookup; powers the CLI ``chain --discover`` flag.
        """
        return await self._backend_impl.acandidate_keys(
            subject_contains=subject_contains,
            predicate_contains=predicate_contains,
            scope=scope,
            limit=limit,
        )

    async def aclose(self) -> None:
        await self._backend_impl.aclose()

    # --- Sync wrappers ---

    def add(self, nugget: Nugget) -> AddResult:
        _require_no_running_loop("add", "aadd")
        return asyncio.run(self.aadd(nugget))

    def get(self, nugget_id: str) -> Nugget | None:
        _require_no_running_loop("get", "aget")
        return asyncio.run(self.aget(nugget_id))

    def count(self, *, status: LifecycleStatus | None = None) -> int:
        _require_no_running_loop("count", "acount")
        return asyncio.run(self._backend_impl.acount(status))

    def get_source_passages(self, nuggets: list[Nugget]) -> dict[str, str]:
        _require_no_running_loop("get_source_passages", "aget_source_passages")
        return asyncio.run(self.aget_source_passages(nuggets))

    def ingest(self, doc: Any) -> IngestResult:
        _require_no_running_loop("ingest", "aingest")
        return asyncio.run(self.aingest(doc))

    def retrieve(
        self,
        query: str,
        *,
        query_time: datetime | None = None,
        view: str = "active",
        top_k: int = 20,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        _require_no_running_loop("retrieve", "aretrieve")
        return asyncio.run(
            self.aretrieve(
                query,
                query_time=query_time,
                view=view,
                top_k=top_k,
                fusion=fusion,
                filters=filters,
            )
        )

    def close(self) -> None:
        _require_no_running_loop("close", "aclose")
        asyncio.run(self._backend_impl.aclose())

    def mark_preferred(self, nugget_id: str, *, set_active: bool = True) -> Nugget:
        _require_no_running_loop("mark_preferred", "amark_preferred")
        return asyncio.run(self.amark_preferred(nugget_id, set_active=set_active))

    def suppress(self, nugget_id: str) -> Nugget:
        _require_no_running_loop("suppress", "asuppress")
        return asyncio.run(self.asuppress(nugget_id))

    def contested_keys(self) -> list[tuple[str, str, str, int]]:
        _require_no_running_loop("contested_keys", "acontested_keys")
        return asyncio.run(self.acontested_keys())

    # --- Chain sync wrappers (v0.2) ---

    def chain_succession(
        self,
        *,
        subject: str,
        predicate: str,
        scope: str = "global",
        as_of: datetime | None = None,
        include_contested: bool = False,
        max_depth: int = 50,
    ) -> NuggetChain:
        _require_no_running_loop("chain_succession", "achain_succession")
        return asyncio.run(
            self.achain_succession(
                subject=subject,
                predicate=predicate,
                scope=scope,
                as_of=as_of,
                include_contested=include_contested,
                max_depth=max_depth,
            )
        )

    def chain_rename(
        self,
        *,
        subject: str,
        as_of: datetime | None = None,
        direction: Literal["forward", "backward", "both"] = "forward",
        max_depth: int = 10,
        include_contested: bool = False,
        resolver: Any | None = None,
    ) -> NuggetChain:
        _require_no_running_loop("chain_rename", "achain_rename")
        return asyncio.run(
            self.achain_rename(
                subject=subject,
                as_of=as_of,
                direction=direction,
                max_depth=max_depth,
                include_contested=include_contested,
                resolver=resolver,
            )
        )

    def chain_join(
        self,
        *,
        start: tuple[str, str],
        then: list[str],
        scope: str = "global",
        as_of: datetime | None = None,
        resolver: Any | None = None,
    ) -> NuggetChain:
        _require_no_running_loop("chain_join", "achain_join")
        return asyncio.run(
            self.achain_join(
                start=start,
                then=then,
                scope=scope,
                as_of=as_of,
                resolver=resolver,
            )
        )

    def candidate_keys(
        self,
        *,
        subject_contains: str | None = None,
        predicate_contains: str | None = None,
        scope: str = "global",
        limit: int = 20,
    ) -> list[tuple[str, str, str]]:
        _require_no_running_loop("candidate_keys", "acandidate_keys")
        return asyncio.run(
            self.acandidate_keys(
                subject_contains=subject_contains,
                predicate_contains=predicate_contains,
                scope=scope,
                limit=limit,
            )
        )
