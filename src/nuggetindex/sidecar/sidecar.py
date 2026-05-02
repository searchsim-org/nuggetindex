"""The :class:`Sidecar` runtime.

Wraps a :class:`~nuggetindex.NuggetStore`, a :class:`~Router`, a
:class:`~ContextFormatter`, and one of two resolution modes
(:class:`~OfflineCurated`, :class:`~JustInTime`). The sidecar is deliberately
framework-agnostic; the framework adapters under ``nuggetindex.integrations``
bridge it onto LangChain / Haystack / LlamaIndex pipelines.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from nuggetindex.core.models import Nugget
from nuggetindex.sidecar.context import ContextFormatter
from nuggetindex.sidecar.freshness import FreshnessChecker
from nuggetindex.sidecar.jit_cache import JITPassageCache
from nuggetindex.sidecar.modes import JustInTime, OfflineCurated, SidecarMode
from nuggetindex.sidecar.router import Router, RouterDecision
from nuggetindex.store.base import NuggetStore, _require_no_running_loop


@dataclass(frozen=True)
class SidecarResponse:
    """The output of :meth:`Sidecar.ahandle`.

    Attributes:
        original_hits: Whatever the caller passed in (or an empty list). The
            sidecar never mutates or filters these; adapters can still forward
            them to the LLM alongside the ``context_block``.
        nuggets: Governance facts the sidecar injected.
        context_block: The formatted block to prepend to the LLM prompt. May
            be empty when the router decided to pass through.
        decision: The router's reasoning (for telemetry / debugging).
    """

    original_hits: list[Any] = field(default_factory=list)
    nuggets: list[Nugget] = field(default_factory=list)
    context_block: str = ""
    decision: RouterDecision | None = None


@dataclass
class Sidecar:
    """The sidecar runtime.

    Wraps a :class:`~nuggetindex.NuggetStore` + :class:`Router` +
    :class:`ContextFormatter` + one of the :class:`SidecarMode` strategies.

    Usage::

        sidecar = Sidecar(store=store, mode="offline-curated")
        response = await sidecar.ahandle("who was Google's CEO in 2013?")
        prompt = response.context_block + "\\n\\nUser: " + query
    """

    store: NuggetStore
    mode: Literal["offline-curated", "just-in-time"] = "offline-curated"
    router: Router = field(default_factory=Router)
    context_formatter: ContextFormatter = field(default_factory=ContextFormatter)
    original_retriever: Any | None = None
    extractor: Any | None = None
    jit_cache: JITPassageCache | None = None
    fallback_corpus: Any | None = None
    freshness_threshold: timedelta = field(default_factory=lambda: timedelta(days=90))
    freshness_checker: FreshnessChecker | None = None

    _mode_strategy: SidecarMode = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode == "just-in-time" and self.extractor is None:
            raise ValueError(
                "just-in-time mode requires an `extractor=` argument (e.g., LLMExtractor(...))."
            )
        if self.fallback_corpus is not None and self.extractor is None:
            raise ValueError(
                "Sidecar with fallback_corpus also requires an extractor. "
                "Pass extractor=LLMExtractor(...) or TriggerExtractor()."
            )
        # Default the JIT cache to an in-memory instance so repeat-passage
        # queries amortise across a session without the caller having to
        # wire it explicitly. offline-curated mode never reads it.
        if self.mode == "just-in-time" and self.jit_cache is None:
            self.jit_cache = JITPassageCache()
        # Auto-construct a FreshnessChecker when a fallback_corpus is set
        # without one.
        if self.fallback_corpus is not None and self.freshness_checker is None:
            self.freshness_checker = FreshnessChecker(threshold=self.freshness_threshold)
        self._mode_strategy = OfflineCurated() if self.mode == "offline-curated" else JustInTime()

    async def ahandle(
        self,
        query: str,
        *,
        query_time: datetime | None = None,
        top_k: int = 10,
        original_hits: list[Any] | None = None,
    ) -> SidecarResponse:
        """Classify the query, resolve nugget context, and format it.

        Algorithm:

        1. Ask the router whether nugget context is worth computing.
        2. If not, return a passthrough ``SidecarResponse``.
        3. Otherwise invoke the mode strategy to resolve nuggets.
        4. Collect contested facts (disputes) from the resolved set.
        5. Hand off to the :class:`ContextFormatter`.
        """
        now = query_time or datetime.now(tz=UTC)
        decision = self.router.classify(query, now=now)

        if not decision.use_nugget:
            return SidecarResponse(
                original_hits=original_hits or [],
                decision=decision,
            )

        if isinstance(self._mode_strategy, JustInTime):
            nuggets = await self._mode_strategy.aresolve(
                self.store,
                decision,
                query,
                top_k,
                original_hits=original_hits,
                extractor=self.extractor,
                jit_cache=self.jit_cache,
            )
        else:
            nuggets = await self._mode_strategy.aresolve(
                self.store,
                decision,
                query,
                top_k,
            )

        # Freshness fallback: if the caller wired a fallback_corpus AND the
        # store's evidence for this decision's key is stale, augment with live
        # web search results before formatting the context block.
        if (
            self.mode == "offline-curated"
            and self.fallback_corpus is not None
            and self.freshness_checker is not None
            and decision.use_nugget
        ):
            # Heuristic key extraction: first (subject, predicate) from the
            # store's returned nuggets, or skip when there's nothing to
            # anchor on.
            key_subject = nuggets[0].fact.subject if nuggets else None
            key_predicate = nuggets[0].fact.predicate if nuggets else None
            stale = True
            if key_subject and key_predicate:
                freshness = await self.freshness_checker.check_store(
                    self.store,
                    subject=key_subject,
                    predicate=key_predicate,
                )
                stale = not freshness.is_fresh
            if stale:
                try:
                    fresh_hits = await self.fallback_corpus.search(query, limit=5)
                except Exception:  # noqa: BLE001 -- best-effort fallback
                    fresh_hits = []
                if fresh_hits and self.extractor is not None:
                    for hit in fresh_hits:
                        try:
                            results = await self.extractor.aextract(
                                hit.text,
                                source_id=getattr(hit, "source_id", "") or "",
                            )
                        except Exception:  # noqa: BLE001
                            continue
                        for r in results:
                            nuggets.append(r.nugget)
                # Tag the decision so observers can tell the fallback fired.
                decision = RouterDecision(
                    use_nugget=decision.use_nugget,
                    query_time=decision.query_time,
                    expand_aliases=decision.expand_aliases,
                    reason=(
                        decision.reason + " · fallback:web" if decision.reason else "fallback:web"
                    ),
                )

        # Contested nuggets are already inside ``nuggets`` (view
        # ``active_contested``); the formatter will pull them into the
        # DISPUTED FACTS group. We don't need a separate disputes list.
        context_block = self.context_formatter.format(
            nuggets=nuggets,
            disputes=None,
        )

        return SidecarResponse(
            original_hits=original_hits or [],
            nuggets=nuggets,
            context_block=context_block,
            decision=decision,
        )

    def handle(self, *args: Any, **kwargs: Any) -> SidecarResponse:
        """Synchronous wrapper around :meth:`ahandle`.

        Guards against running inside an event loop, matching the
        :meth:`NuggetStore` sync-wrapper idiom. Use :meth:`ahandle` from
        async code; :meth:`handle` is a convenience for scripts and notebooks
        that have no running loop.
        """
        _require_no_running_loop("handle", "ahandle")
        return asyncio.run(self.ahandle(*args, **kwargs))
