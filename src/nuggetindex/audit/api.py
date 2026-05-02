"""Zero-index audit: extract nuggets from passages and flag conflicts / stale facts.

The audit function is async-first and defaults to the rule-based extractor so
first-run usage requires no API keys and no network access. The function
pipes each passage through the configured extractor, resolves each produced
nugget against everything extracted so far via :class:`ConflictDetector`,
and records both conflicts (same key + overlapping functional validity) and
potentially-stale nuggets (open-ended validity).

Stale heuristic (v0.2.1): a nugget is flagged iff its ``validity.end`` is
``None`` *and* its most recent provenance ``created_at`` is older than
``stale_threshold_days`` (default 180). Nuggets with no provenance are not
flagged under the age-aware rule because the evidence age is unknown. Pass
``stale_threshold_days=None`` to restore the v0.2.0 behaviour where any
open-ended validity was flagged regardless of age. (findings-A2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nuggetindex.core.models import Nugget

if TYPE_CHECKING:
    from rich.console import RenderableType

    from nuggetindex.core.schema import RelationSchema
    from nuggetindex.extractors.base import BaseExtractor


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class ConflictRecord:
    """One detected conflict between two nuggets sharing ``key``.

    ``nugget_a`` is the incoming nugget (possibly already marked CONTESTED /
    DEPRECATED by the detector); ``nugget_b`` is the previously-seen nugget
    that the detector updated. ``reason`` is a short descriptive string --
    kept human-readable rather than a machine enum so Markdown / CLI output
    is self-explanatory.
    """

    key: tuple[str, str, str]
    nugget_a: Nugget
    nugget_b: Nugget
    reason: str


@dataclass
class StaleRecord:
    """A nugget that may be outdated (v0.1: open-ended validity interval)."""

    nugget: Nugget
    source_date: datetime | None
    reason: str


@dataclass
class AuditReport:
    """Result of :func:`audit`: conflicts, stale candidates, consistent count.

    ``consistent`` is the count of extracted nuggets that triggered neither
    a conflict nor the stale heuristic -- useful as a sanity metric when
    visualising the report.
    """

    conflicts: list[ConflictRecord] = field(default_factory=list)
    potentially_stale: list[StaleRecord] = field(default_factory=list)
    consistent: int = 0

    # -- serialization -----------------------------------------------------

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Pydantic ``Nugget`` instances are expanded via ``model_dump(mode="json")``
        so datetimes become ISO-8601 strings. The dataclass envelopes themselves
        are translated to plain dicts by hand (``dataclasses.asdict`` walks
        into Pydantic models, which defeats the point).
        """
        payload = {
            "conflicts": [_conflict_to_dict(c) for c in self.conflicts],
            "potentially_stale": [_stale_to_dict(s) for s in self.potentially_stale],
            "consistent": self.consistent,
        }
        return json.dumps(payload, indent=2, sort_keys=False)

    def to_markdown(self) -> str:
        """Render the report as human-readable Markdown."""
        lines: list[str] = ["# Audit Report", ""]
        lines.append(
            f"**Summary:** {len(self.conflicts)} conflict(s), "
            f"{len(self.potentially_stale)} potentially stale, "
            f"{self.consistent} consistent."
        )
        lines.append("")

        lines.append("## Conflicts")
        if not self.conflicts:
            lines.append("_None detected._")
        else:
            for i, c in enumerate(self.conflicts, 1):
                subj, pred, scope = c.key
                lines.append(f"### {i}. `{subj}` / `{pred}` ({scope})")
                lines.append(f"- **Reason:** {c.reason}")
                lines.append(
                    f"- **A:** `{c.nugget_a.fact.object}` "
                    f"(status={c.nugget_a.epistemic.status.value}, "
                    f"validity=[{_fmt_dt(c.nugget_a.validity.start)}"
                    f" .. {_fmt_dt(c.nugget_a.validity.end)}])"
                )
                lines.append(
                    f"- **B:** `{c.nugget_b.fact.object}` "
                    f"(status={c.nugget_b.epistemic.status.value}, "
                    f"validity=[{_fmt_dt(c.nugget_b.validity.start)}"
                    f" .. {_fmt_dt(c.nugget_b.validity.end)}])"
                )
                lines.append("")
        lines.append("")

        lines.append("## Potentially Stale")
        if not self.potentially_stale:
            lines.append("_None detected._")
        else:
            for i, s in enumerate(self.potentially_stale, 1):
                lines.append(
                    f"{i}. `{s.nugget.fact.subject}` / "
                    f"`{s.nugget.fact.predicate}` -> "
                    f"`{s.nugget.fact.object}` -- {s.reason}"
                )
        lines.append("")

        lines.append(f"## Consistent: {self.consistent}")
        return "\n".join(lines)

    def to_rich_console(self) -> RenderableType:
        """Return a Rich renderable that groups a conflicts table and a summary panel."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        table = Table(title="Conflicts", show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("A (object)")
        table.add_column("B (object)")
        table.add_column("Reason")
        if self.conflicts:
            for c in self.conflicts:
                subj, pred, _ = c.key
                table.add_row(
                    f"{subj} / {pred}",
                    c.nugget_a.fact.object,
                    c.nugget_b.fact.object,
                    c.reason,
                )
        else:
            table.add_row("-", "-", "-", "No conflicts detected")

        summary = Panel.fit(
            f"[bold]Conflicts:[/bold] {len(self.conflicts)}    "
            f"[bold]Potentially stale:[/bold] {len(self.potentially_stale)}    "
            f"[bold]Consistent:[/bold] {self.consistent}",
            title="Audit Summary",
        )
        return Group(table, summary)


# ---------------------------------------------------------------------------
# audit() and audit_batch()
# ---------------------------------------------------------------------------


async def audit(
    *,
    query: str,  # noqa: ARG001 - surfaced for API symmetry; not used in v0.1 scoring
    passages: list[str],
    query_time: datetime,
    extractor: str | BaseExtractor = "rule_based",
    schema: RelationSchema | None = None,
    stale_threshold_days: int | None = 180,
) -> AuditReport:
    """Zero-index audit: extract nuggets from ``passages`` and report issues.

    Parameters
    ----------
    query:
        The user's question. Not used for conflict detection in v0.1 but
        kept in the signature so downstream tools (CLI, Phase 8 governance)
        can route / label reports by query without a breaking change.
    passages:
        Retrieved context passages to audit.
    query_time:
        The "as of" timestamp. Used by the staleness heuristic.
    extractor:
        Either the sentinel string ``"rule_based"`` (default, zero-setup),
        any other string treated as an OpenAI model name, or a
        ``BaseExtractor`` instance. Construction is lazy so importing this
        module does not require the ``[openai]`` extra.
    schema:
        Optional custom :class:`RelationSchema`; defaults to the bundled
        50-predicate schema.
    stale_threshold_days:
        Age cut-off in days applied to the most recent provenance
        ``created_at`` when deciding whether an open-ended nugget is
        "potentially stale". Defaults to ``180``. Pass ``None`` to
        restore the v0.2.0 behaviour where any open-ended validity was
        flagged regardless of age. (findings-A2)

    Returns
    -------
    AuditReport
        Conflicts, stale candidates, and a consistent-count.
    """
    from nuggetindex.core.schema import RelationSchema
    from nuggetindex.pipeline.conflict import ConflictDetector

    resolved_schema = schema or RelationSchema.default()
    ex = _resolve_extractor(extractor)

    detector = ConflictDetector(resolved_schema)
    report = AuditReport()
    all_nuggets: list[Nugget] = []

    for passage in passages:
        extraction_results = await ex.aextract(passage)
        for r in extraction_results:
            resolution = await detector.aresolve(r.nugget, all_nuggets)
            incoming = resolution.incoming
            all_nuggets.append(incoming)
            for other in resolution.updated_existing:
                # Replace the mutated existing nugget in our running list so
                # further resolutions see the updated state.
                _replace_by_id(all_nuggets, other)
                report.conflicts.append(
                    ConflictRecord(
                        key=incoming.key,
                        nugget_a=incoming,
                        nugget_b=other,
                        reason="functional predicate + overlapping validity",
                    )
                )

    # Walk the final list once to classify stale vs consistent.
    conflicted_ids = {c.nugget_a.id for c in report.conflicts} | {
        c.nugget_b.id for c in report.conflicts
    }
    for n in all_nuggets:
        if _classify_stale(
            n,
            query_time=query_time,
            stale_threshold_days=stale_threshold_days,
        ):
            report.potentially_stale.append(
                StaleRecord(
                    nugget=n,
                    source_date=_infer_source_date(n),
                    reason=_stale_reason(n, query_time),
                )
            )
        elif n.id not in conflicted_ids:
            report.consistent += 1

    return report


async def audit_batch(
    *,
    jsonl_path: str | Path,
    query_time: datetime,
    extractor: str | BaseExtractor = "rule_based",
    schema: RelationSchema | None = None,
) -> list[AuditReport]:
    """Run :func:`audit` over each row of a JSONL file.

    Each row must be an object with at least ``query`` (str) and ``passages``
    (list[str]). Blank lines are skipped. Returns one :class:`AuditReport`
    per row, in source order.
    """
    path = Path(jsonl_path)
    reports: list[AuditReport] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            report = await audit(
                query=row["query"],
                passages=list(row["passages"]),
                query_time=query_time,
                extractor=extractor,
                schema=schema,
            )
            reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_extractor(extractor: str | BaseExtractor) -> BaseExtractor:
    """Map ``extractor`` (string or instance) to a concrete ``BaseExtractor``.

    String dispatch:

    * ``"trigger"`` -- the LLM-free :class:`TriggerExtractor` (default for
      the CLI; zero setup, no API key).
    * ``"rule_based"`` -- deprecated alias retained for script compatibility;
      emits a :class:`DeprecationWarning` and delegates to ``"trigger"``.
    * any other string -- treated as an OpenAI-compatible model id and
      wrapped in an :class:`LLMExtractor`.

    Construction of the LLM extractor is the first point that touches the
    ``openai`` extra, which keeps the default import path light.
    """
    import warnings

    from nuggetindex.extractors.base import BaseExtractor

    if isinstance(extractor, BaseExtractor):
        return extractor
    if extractor == "rule_based":
        warnings.warn(
            "extractor='rule_based' is deprecated: delegating to "
            "TriggerExtractor ('trigger'). Pass 'trigger' explicitly to "
            "silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        extractor = "trigger"
    if extractor == "trigger":
        from nuggetindex.extractors.trigger import TriggerExtractor

        return TriggerExtractor()
    if isinstance(extractor, str):
        from nuggetindex.extractors.clients.base import LLMConfig
        from nuggetindex.extractors.llm import LLMExtractor

        return LLMExtractor(LLMConfig(provider="openai", model=extractor))
    raise TypeError(f"extractor must be a string or BaseExtractor, got {type(extractor).__name__}")


def _replace_by_id(nuggets: list[Nugget], updated: Nugget) -> None:
    """In-place swap of the first nugget with ``updated.id`` for ``updated``."""
    for i, n in enumerate(nuggets):
        if n.id == updated.id:
            nuggets[i] = updated
            return


def _classify_stale(
    nugget: Nugget,
    *,
    query_time: datetime,
    stale_threshold_days: int | None,
) -> bool:
    """True iff the nugget qualifies as 'potentially stale'.

    v0.2.0 always returned True for any open-ended validity. v0.2.1
    additionally requires that the most recent provenance record is older
    than ``stale_threshold_days``. Passing ``None`` restores the v0.2.0
    behaviour. Nuggets with no provenance are *not* flagged under the
    age-aware rule because we can't tell how old the evidence is.
    """
    if nugget.validity.end is not None:
        return False  # explicit end: not stale by definition
    if stale_threshold_days is None:
        return True  # v0.2.0 fallback
    if not nugget.provenance:
        return False  # no source date -> can't tell -> don't flag
    last = max(p.created_at for p in nugget.provenance)
    delta = query_time - last
    return delta > timedelta(days=stale_threshold_days)


def _infer_source_date(n: Nugget) -> datetime | None:
    """Earliest provenance ``created_at`` is our best-available source date."""
    if not n.provenance:
        return None
    return min(p.created_at for p in n.provenance)


def _stale_reason(n: Nugget, query_time: datetime) -> str:
    """Short human-readable staleness justification.

    If provenance is older than 180 days relative to ``query_time``, call
    that out explicitly; otherwise report the generic "no explicit end time"
    reason.
    """
    source_date = _infer_source_date(n)
    if source_date is None:
        return "no explicit end-time in evidence; may be outdated"
    age_days = (query_time - source_date).days
    if age_days > 180:
        return f"no explicit end-time; source is {age_days} days older than query_time"
    return "no explicit end-time in evidence; may be outdated"


def _fmt_dt(dt: datetime | None) -> str:
    return "open" if dt is None else dt.isoformat()


def _nugget_to_dict(n: Nugget) -> dict[str, Any]:
    """Pydantic JSON-safe dump for a Nugget."""
    return n.model_dump(mode="json")


def _conflict_to_dict(c: ConflictRecord) -> dict[str, Any]:
    return {
        "key": list(c.key),
        "nugget_a": _nugget_to_dict(c.nugget_a),
        "nugget_b": _nugget_to_dict(c.nugget_b),
        "reason": c.reason,
    }


def _stale_to_dict(s: StaleRecord) -> dict[str, Any]:
    return {
        "nugget": _nugget_to_dict(s.nugget),
        "source_date": s.source_date.isoformat() if s.source_date else None,
        "reason": s.reason,
    }
