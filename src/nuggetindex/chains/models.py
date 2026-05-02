"""Data types for temporal provenance chains.

Three chain kinds share the same ``NuggetChain`` envelope:

* **succession** -- same ``(subject, predicate, scope)`` key ordered by
  ``validity_start``.
* **rename** -- a walk over renaming-predicate edges (``renamedTo``,
  ``corporateName``, ...).
* **joined** -- a bounded 1--3 hop functional lookup chain.

Every chain carries its nuggets in order plus one :class:`ChainEdge` per
adjacent pair that carries the inter-nugget gap (for succession chains) and
the edge semantics.

Chains are frozen -- callers mutate them by returning a new chain.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from nuggetindex.core.models import Nugget


class ChainEdgeType(StrEnum):
    """Semantic label on the edge between two adjacent chain nuggets."""

    SUCCEEDS = "succeeds"
    RENAMES_TO = "renames_to"
    OBJECT_IS_SUBJECT = "object_is_subject"


class ChainEdge(BaseModel, frozen=True):
    """Directed edge between two indices in :attr:`NuggetChain.nuggets`.

    ``gap`` is the time between consecutive validity intervals for succession
    chains; for rename/join edges it is typically ``None`` because validity
    intervals are not expected to be adjacent in that way.
    """

    from_idx: int
    to_idx: int
    edge_type: ChainEdgeType
    gap: timedelta | None = None


class NuggetChain(BaseModel, frozen=True):
    """An ordered sequence of :class:`Nugget` records plus their edges.

    ``truncated`` becomes ``True`` when a walk hits ``max_depth`` before
    terminating naturally. ``as_of`` echoes back the temporal cutoff that was
    applied when constructing the chain (so callers can reproduce the query).
    """

    nuggets: tuple[Nugget, ...]
    edges: tuple[ChainEdge, ...]
    chain_type: Literal["succession", "rename", "joined"]
    as_of: datetime | None = None
    truncated: bool = False

    @property
    def head(self) -> Nugget:
        """First nugget in the chain.

        Raises ``ValueError`` on empty chains so callers fail loudly rather
        than silently using ``None``.
        """
        if not self.nuggets:
            raise ValueError("empty chain has no head")
        return self.nuggets[0]

    @property
    def tail(self) -> Nugget:
        """Last nugget in the chain.

        Raises ``ValueError`` on empty chains.
        """
        if not self.nuggets:
            raise ValueError("empty chain has no tail")
        return self.nuggets[-1]

    def window(self, t: datetime) -> Nugget | None:
        """Return the nugget whose validity contains ``t``, or ``None``."""
        for n in self.nuggets:
            if n.validity.contains(t):
                return n
        return None
