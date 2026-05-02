"""Stage 4b: conflict detection (Algorithm 2, paper §3.4, spec §5.5).

Four branches:

1. **New key** -- no prior nuggets with matching ``(subject, predicate, scope)``;
   the incoming stays ACTIVE, nothing changes.
2. **Non-overlapping or multi-valued** -- succession (e.g. CEO transitions)
   or coexisting board members; both stay ACTIVE.
3. **Functional + overlap + asymmetric evidence** (>= 2 on one side, < 2 on
   the other) -- deprecate the loser; when newer wins, tighten the older
   interval's ``end`` so point-in-time queries return the right answer.
4. **Functional + overlap + symmetric evidence** -- both become CONTESTED.
   If an ``LLMJudge`` is attached, it is invoked to pick a side (or confirm
   the contest is genuine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from nuggetindex.core.enums import Cardinality, LifecycleStatus
from nuggetindex.core.models import EpistemicState, Nugget, ValidityInterval
from nuggetindex.core.schema import RelationSchema

if TYPE_CHECKING:
    from nuggetindex.pipeline.judge import LLMJudge


@dataclass
class ResolutionResult:
    """Outcome of conflict resolution for a single incoming nugget.

    ``incoming`` is the (possibly updated) incoming nugget. ``updated_existing``
    contains only the existing nuggets whose lifecycle or validity changed.
    """

    incoming: Nugget
    updated_existing: list[Nugget] = field(default_factory=list)
    judge_invoked: bool = False
    judge_decision: str | None = None


class ConflictDetector:
    """Algorithm 2 + optional LLM-as-judge."""

    def __init__(
        self,
        schema: RelationSchema,
        judge: LLMJudge | None = None,
    ) -> None:
        self.schema = schema
        self.judge = judge

    async def aresolve(
        self,
        incoming: Nugget,
        existing: list[Nugget],
    ) -> ResolutionResult:
        same_key = [e for e in existing if e.key == incoming.key]

        # Branch 1: brand-new key -> ACTIVE, nothing changed.
        if not same_key:
            return ResolutionResult(incoming=incoming)

        predicate = incoming.fact.predicate
        # Prefer the richer ``Cardinality`` API when the schema exposes it
        # (schemas built by newer code do).  Fall back to the older
        # ``is_functional`` boolean so third-party ``RelationSchema`` subclasses
        # that haven't been updated keep working.
        if hasattr(self.schema, "cardinality"):
            cardinality = self.schema.cardinality(predicate)
        else:  # pragma: no cover -- legacy RelationSchema subclasses
            cardinality = (
                Cardinality.FUNCTIONAL
                if self.schema.is_functional(predicate)
                else Cardinality.MULTI_VALUED
            )

        # Branch 2a: multi-valued or event-log predicates -> coexist.  Event-log
        # predicates (announced, said, published, …) are treated as a stream
        # of distinct events rather than competing claims about a single
        # functional attribute, so differing objects never indicate a conflict.
        if cardinality in (Cardinality.MULTI_VALUED, Cardinality.EVENT_LOG):
            return ResolutionResult(incoming=incoming)

        # Branch 2b: functional but non-overlapping -> succession.
        overlapping = [e for e in same_key if e.validity.overlaps(incoming.validity)]
        if not overlapping:
            return ResolutionResult(incoming=incoming)

        updated: list[Nugget] = []
        judge_required = False
        for ex in overlapping:
            new_ev = len(incoming.provenance)
            old_ev = len(ex.provenance)
            new_is_newer = incoming.validity.start > ex.validity.start
            old_is_newer = ex.validity.start > incoming.validity.start

            if new_ev >= 2 and old_ev < 2 and new_is_newer:
                # Branch 3a: newer wins (has more evidence) -> deprecate older,
                # tighten its end.
                updated.append(
                    _with_status_and_end(
                        ex, LifecycleStatus.DEPRECATED, incoming.validity.start
                    )
                )
            elif old_ev >= 2 and new_ev < 2 and old_is_newer:
                # Branch 3b: older wins (later AND more evidence) -> deprecate incoming.
                incoming = _with_status(incoming, LifecycleStatus.DEPRECATED)
            else:
                # Branch 4: symmetric / insufficient asymmetry -> both CONTESTED.
                incoming = _with_status(incoming, LifecycleStatus.CONTESTED)
                updated.append(_with_status(ex, LifecycleStatus.CONTESTED))
                judge_required = True

        # Branch 4 (cont'd): invoke judge on (1,1) evidence contested cases.
        if (
            judge_required
            and self.judge is not None
            and incoming.epistemic.status == LifecycleStatus.CONTESTED
        ):
            for i, ex in enumerate(updated):
                if ex.epistemic.status != LifecycleStatus.CONTESTED:
                    continue
                new_ev = len(incoming.provenance)
                old_ev = len(ex.provenance)
                # Only adjudicate true (1,1)-style contested cases; if we
                # somehow got here with asymmetric evidence, leave it alone.
                if new_ev != 1 or old_ev != 1:
                    continue
                decision = await self.judge.aadjudicate(incoming, ex)
                decision_str = str(decision)
                if decision_str == "A_WINS":
                    incoming = _with_status(incoming, LifecycleStatus.ACTIVE)
                    updated[i] = _with_status_and_end(
                        ex, LifecycleStatus.DEPRECATED, incoming.validity.start
                    )
                elif decision_str == "B_WINS":
                    incoming = _with_status(incoming, LifecycleStatus.DEPRECATED)
                    updated[i] = _with_status(ex, LifecycleStatus.ACTIVE)
                # GENUINELY_CONTESTED / NEED_MORE_EVIDENCE: leave CONTESTED.
                return ResolutionResult(
                    incoming=incoming,
                    updated_existing=updated,
                    judge_invoked=True,
                    judge_decision=decision_str,
                )

        return ResolutionResult(incoming=incoming, updated_existing=updated)


# --- helpers ---------------------------------------------------------------


def _with_status(n: Nugget, status: LifecycleStatus) -> Nugget:
    new_epistemic = EpistemicState(
        status=status,
        rank=n.epistemic.rank,
        confidence=n.epistemic.confidence,
    )
    return n.model_copy(update={"epistemic": new_epistemic})


def _with_status_and_end(n: Nugget, status: LifecycleStatus, new_end: datetime) -> Nugget:
    new_epistemic = EpistemicState(
        status=status,
        rank=n.epistemic.rank,
        confidence=n.epistemic.confidence,
    )
    # Only tighten when the new end is strictly after start AND earlier than
    # (or equal to) any existing end. This preserves the invariant
    # end > start inside ValidityInterval's validator.
    if new_end <= n.validity.start:
        new_interval = n.validity
    else:
        existing_end = n.validity.end
        tightened_end = new_end if existing_end is None else min(existing_end, new_end)
        new_interval = ValidityInterval(
            start=n.validity.start,
            end=tightened_end,
            scope=n.validity.scope,
            source_type=n.validity.source_type,
        )
    return n.model_copy(update={"epistemic": new_epistemic, "validity": new_interval})
