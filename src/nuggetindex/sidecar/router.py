"""Cheap, deterministic query classifier for the sidecar runtime.

The router decides whether a query is likely to benefit from nugget-grounded
context (temporal or functional-predicate intent) or whether it should pass
through unchanged. The classifier is regex + keyword based so it stays cheap
on the hot path; an optional LLM fallback is available via the
``llm_classifier`` kwarg.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nuggetindex.core.schema import RelationKind, RelationSchema


@dataclass(frozen=True)
class RouterDecision:
    """The outcome of :meth:`Router.classify`.

    Attributes:
        use_nugget: ``True`` when the query should be routed through the
            sidecar (temporal or functional-predicate intent). ``False`` means
            the caller should fall back to the original retriever unchanged.
        query_time: The instant to use for validity filtering when
            ``use_nugget`` is ``True``. ``None`` for non-temporal triggers.
        expand_aliases: Subject aliases worth expanding against the store.
            Left empty here; :meth:`Sidecar.ahandle` populates this via a
            rename-chain lookup if required.
        reason: Human-readable reason string (for telemetry / debugging).
    """

    use_nugget: bool
    query_time: datetime | None = None
    expand_aliases: list[str] = field(default_factory=list)
    reason: str = ""


class Router:
    """Cheap, deterministic query classifier.

    The classifier combines two signals:

    1. A regex sweep for temporal intent (``in 2013``, ``as of``, ``when``,
       ``before 2013``, ``after 2013``) that also returns a ``query_time`` the
       store can use for validity filtering.
    2. A substring search for functional-predicate keywords derived from the
       :class:`~nuggetindex.core.schema.RelationSchema` (names + aliases of
       every ``FUNCTIONAL`` relation). This catches ``who is the ceo of...``
       style queries that have no explicit time expression.

    An LLM fallback can be plugged in via the ``llm_classifier`` kwarg; it is
    only consulted when the cheap path returns ``use_nugget=False`` so the hot
    path stays deterministic and fast.
    """

    # Regex patterns for temporal intent
    _RE_YEAR = re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE)
    _RE_AS_OF = re.compile(r"\bas of\b", re.IGNORECASE)
    _RE_WHEN = re.compile(r"\bwhen\b", re.IGNORECASE)
    _RE_BEFORE = re.compile(r"\b(?:before|prior to)\s+(\d{4})\b", re.IGNORECASE)
    _RE_AFTER = re.compile(r"\b(?:after|since)\s+(\d{4})\b", re.IGNORECASE)

    def __init__(
        self,
        *,
        schema: RelationSchema | None = None,
        llm_classifier: Any | None = None,
    ) -> None:
        self.schema = schema or RelationSchema.default()
        self.llm_classifier = llm_classifier
        self._functional_keywords = self._build_functional_keywords()

    def _build_functional_keywords(self) -> set[str]:
        """Derive a keyword vocabulary from the functional predicates in the schema.

        For each functional relation (and its aliases), derive the kebab / camel
        / space forms that are likely to appear in user queries. E.g.
        ``chiefExecutiveOfficer`` + aliases ``[ceo, "chief executive officer"]``
        → ``{"chief executive officer", "ceo"}``.

        ``RelationSchema`` does not expose a public iterator over relations in
        the current API, so we reach into ``_by_name`` with a ``noqa`` comment.
        """
        kws: set[str] = set()
        # RelationSchema exposes no public relations iterator today; the private
        # mapping is the cheapest hook. If a public accessor is added later,
        # swap this for it.
        relations = self.schema._by_name.values()  # noqa: SLF001
        for relation in relations:
            if relation.kind != RelationKind.FUNCTIONAL:
                continue
            kws.add(_camel_to_words(relation.name).lower())
            for alias in relation.aliases:
                alias_clean = alias.strip().lower()
                if alias_clean:
                    kws.add(alias_clean)
        return kws

    def classify(self, query: str, *, now: datetime) -> RouterDecision:
        """Classify a user query. Returns a :class:`RouterDecision`."""
        q = query.lower().strip()
        reasons: list[str] = []
        query_time: datetime | None = None

        # Temporal intent — first match wins, checked in rough specificity order.
        m = self._RE_YEAR.search(query)
        if m is not None:
            try:
                query_time = datetime(int(m.group(1)), 6, 15, tzinfo=now.tzinfo)
                reasons.append(f"temporal:year={m.group(1)}")
            except ValueError:
                pass
        elif self._RE_AS_OF.search(q) or self._RE_WHEN.search(q):
            query_time = now
            reasons.append("temporal:as_of/when")
        elif (m := self._RE_BEFORE.search(query)) is not None:
            try:
                query_time = datetime(int(m.group(1)) - 1, 12, 31, tzinfo=now.tzinfo)
                reasons.append(f"temporal:before={m.group(1)}")
            except ValueError:
                pass
        elif (m := self._RE_AFTER.search(query)) is not None:
            query_time = now
            reasons.append(f"temporal:after={m.group(1)}")

        # Functional-predicate intent.
        hit_functional = self._match_functional_keyword(q)
        if hit_functional is not None:
            reasons.append(f"functional_predicate:{hit_functional}")

        use_nugget = bool(query_time is not None or hit_functional)
        if not use_nugget and self.llm_classifier is not None:
            return self._llm_classify(query, now)

        return RouterDecision(
            use_nugget=use_nugget,
            query_time=query_time,
            expand_aliases=[],
            reason=" · ".join(reasons) or "no_trigger",
        )

    def _match_functional_keyword(self, q: str) -> str | None:
        """Return the first functional keyword found in ``q`` (lowercased)."""
        # Try longer keywords first so "chief executive officer" wins over "ceo"
        # when both would match.
        for kw in sorted(self._functional_keywords, key=len, reverse=True):
            if not kw:
                continue
            # Use word-ish boundaries for short tokens to avoid false hits
            # (e.g. "ceo" matching inside "ceoing" or a URL).
            if len(kw) <= 4:
                pattern = rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])"
                if re.search(pattern, q):
                    return kw
            elif kw in q:
                return kw
        return None

    def _llm_classify(self, query: str, now: datetime) -> RouterDecision:
        """Delegate classification to the caller-supplied classifier.

        The classifier must return a :class:`RouterDecision`; any other return
        value degrades to a ``use_nugget=False`` passthrough with a telemetry
        reason. This keeps the LLM opt-in and robust to misbehaving shims.
        """
        try:
            decision = self.llm_classifier(query, now)
        except Exception:  # noqa: BLE001 — LLM fallback is best-effort
            return RouterDecision(use_nugget=False, reason="llm_classifier_raised")
        if not isinstance(decision, RouterDecision):
            return RouterDecision(use_nugget=False, reason="llm_classifier_returned_unknown")
        if not decision.reason or "llm" not in decision.reason.lower():
            # Ensure the reason string makes the LLM source obvious downstream.
            decision = RouterDecision(
                use_nugget=decision.use_nugget,
                query_time=decision.query_time,
                expand_aliases=list(decision.expand_aliases),
                reason=(f"llm:{decision.reason}" if decision.reason else "llm_fallback"),
            )
        return decision


def _camel_to_words(name: str) -> str:
    """Split a camelCase identifier into space-separated words."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
