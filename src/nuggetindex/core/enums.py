"""Enumerations for nugget kinds, lifecycle states, and epistemic ranks.

All enums inherit from StrEnum so they serialize transparently to strings in
SQLite columns and JSON payloads.
"""

from enum import StrEnum


class NuggetKind(StrEnum):
    SEMANTIC_FACT = "semantic_fact"
    EPISODIC_EVENT = "episodic_event"
    INSTRUCTION = "instruction"
    USER_PREFERENCE = "user_preference"


class LifecycleStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    CONTESTED = "contested"


class EpistemicRank(StrEnum):
    PREFERRED = "preferred"
    NORMAL = "normal"
    DEPRECATED = "deprecated"


class Cardinality(StrEnum):
    """How many distinct object values a relation can hold for one subject/scope.

    Orthogonal to :class:`~nuggetindex.core.schema.RelationKind`; new code should
    prefer ``Cardinality`` because it distinguishes true "one value at a time"
    relations from news-verb / event-log predicates whose extractions are a
    stream of events rather than a contested single-valued attribute.
    """

    FUNCTIONAL = "functional"      # one value at a time (CEO, headquarters)
    MULTI_VALUED = "multi_valued"  # many values coexist (board members, acquisitions, aliases)
    EVENT_LOG = "event_log"        # news-verb / timestamped event stream (announced, says, published)
