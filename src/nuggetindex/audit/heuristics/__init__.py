"""Heuristic components for the audit doctor scan."""

from nuggetindex.audit.heuristics.language import _detect_language
from nuggetindex.audit.heuristics.ner import Entity, extract_entities, ner_available
from nuggetindex.audit.heuristics.sample import stratified_sample
from nuggetindex.audit.heuristics.timex import TimeExpression, tag_timex, timex_available
from nuggetindex.audit.heuristics.triggers import (
    TriggerKind,
    TriggerMatch,
    scan_triggers,
    trigger_kinds,
)

__all__ = [
    "Entity",
    "TimeExpression",
    "TriggerKind",
    "TriggerMatch",
    "_detect_language",
    "extract_entities",
    "ner_available",
    "scan_triggers",
    "stratified_sample",
    "tag_timex",
    "timex_available",
    "trigger_kinds",
]
