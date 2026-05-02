"""Tier-0 audit API: extract nuggets from passages, report conflicts and staleness.

No index, no persistence. Given ``(query, passages, query_time)``, ``audit``
runs the extractor + conflict detector in memory and returns a report.
Intended as the zero-setup on-ramp; see Tier-1 (governance / store) for
persistent workflows.

Public surface:

* :func:`audit` -- async entry point, returns :class:`AuditReport`.
* :func:`audit_batch` -- run ``audit`` over a JSONL file of ``{query, passages}`` rows.
* :class:`AuditReport`, :class:`ConflictRecord`, :class:`StaleRecord` -- report types.
* :func:`scan_index` -- index-agnostic scanner returning :class:`DoctorReport`
  (see :mod:`nuggetindex.audit.doctor`).
* :func:`propose_seeds` -- automated seed-query proposer returning
  :class:`SeedProposal` (see :mod:`nuggetindex.audit.seeds`).
"""

from nuggetindex.audit.api import (
    AuditReport,
    ConflictRecord,
    StaleRecord,
    audit,
    audit_batch,
)
from nuggetindex.audit.cost import CostEstimate, estimate_ingest_cost
from nuggetindex.audit.discover import (
    PredicateProposal,
    SchemaProposal,
    discover_schema,
    merge_proposal,
)
from nuggetindex.audit.doctor import DoctorReport, DoctorScore, scan_index
from nuggetindex.audit.seeds import SeedCandidate, SeedProposal, propose_seeds

__all__ = [
    "AuditReport",
    "ConflictRecord",
    "CostEstimate",
    "DoctorReport",
    "DoctorScore",
    "PredicateProposal",
    "SchemaProposal",
    "SeedCandidate",
    "SeedProposal",
    "StaleRecord",
    "audit",
    "audit_batch",
    "discover_schema",
    "estimate_ingest_cost",
    "merge_proposal",
    "propose_seeds",
    "scan_index",
]
