"""Format governance facts into an LLM-prompt-ready context block.

The :class:`ContextFormatter` is deliberately neutral about prompt templates:
it emits a structured block that callers can prepend to whatever prompt
template they already use. The output is deterministic so tests can assert
on it and so evaluation pipelines can cache prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nuggetindex.core.enums import LifecycleStatus
from nuggetindex.core.models import Nugget


@dataclass
class ContextFormatter:
    """Format a set of governance facts into an LLM-prompt-ready context block.

    Output format is deterministic and neutral; callers stitch it into
    whatever prompt template they already use. Contested facts are grouped by
    ``(subject, predicate)`` so the model can see the disagreement shape.
    """

    def format(
        self,
        *,
        nuggets: list[Nugget],
        disputes: list[Any] | None = None,
    ) -> str:
        if not nuggets and not disputes:
            return ""

        lines: list[str] = []

        known = [n for n in nuggets if n.epistemic.status != LifecycleStatus.CONTESTED]
        if known:
            lines.append("KNOWN FACTS:")
            for n in known:
                lines.append(self._format_fact(n))
            lines.append("")

        contested = [n for n in nuggets if n.epistemic.status == LifecycleStatus.CONTESTED]
        # ``disputes`` may be a separate iterable provided by the caller; we
        # treat items that look like Nuggets the same as inline contested
        # nuggets. Anything else is ignored so this method stays total.
        if disputes:
            for item in disputes:
                if isinstance(item, Nugget):
                    contested.append(item)

        if contested:
            lines.append("DISPUTED FACTS (sources disagree — surface the disagreement):")
            lines.extend(self._format_disputes(contested))
            lines.append("")

        return "\n".join(lines).strip()

    def _format_fact(self, n: Nugget) -> str:
        start = n.validity.start.date().isoformat() if n.validity else "?"
        end = n.validity.end.date().isoformat() if (n.validity and n.validity.end) else "present"
        status = n.epistemic.status.value if n.epistemic else "?"
        source = n.provenance[0].source_id if n.provenance else "?"
        known_suffix = ""
        if n.validity and not n.validity.validity_known:
            known_suffix = " · (calendar date unknown — source-date fallback)"
        return (
            f"  {n.fact.subject} / {n.fact.predicate} → {n.fact.object}  "
            f"({status}, valid {start} — {end}{known_suffix}, source: {source})"
        )

    def _format_disputes(self, contested: list[Nugget]) -> list[str]:
        """Group contested nuggets by (subject, predicate) and list claims."""
        groups: dict[tuple[str, str], list[Nugget]] = {}
        for n in contested:
            key = (n.fact.subject, n.fact.predicate)
            groups.setdefault(key, []).append(n)
        out: list[str] = []
        for (subj, pred), nuggs in groups.items():
            out.append(f"  {subj} / {pred}:")
            for n in nuggs:
                src = n.provenance[0].source_id if n.provenance else "?"
                raw_evidence = n.provenance[0].evidence_span if n.provenance else ""
                evidence = raw_evidence[:140]
                out.append(f'    • {n.fact.object}  — {src}: "{evidence}"')
        return out
