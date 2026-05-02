"""Typed exception hierarchy for nuggetindex.

All project-raised exceptions inherit from NuggetIndexError so callers can
catch everything with a single except clause if they want to, or catch
specific subclasses for targeted handling.
"""

from typing import Any


class NuggetIndexError(Exception):
    """Root exception for all nuggetindex-raised errors."""


class ExtractionFailed(NuggetIndexError):
    """LLM extractor returned unparseable or invalid output."""


class ConflictUnresolved(NuggetIndexError):
    """Conflict detection produced an ambiguous state that could not be resolved."""


class BackendUnavailable(NuggetIndexError):
    """A storage backend (SQLite, FAISS, Qdrant, ...) is not reachable or missing deps."""


class InvalidRelationSchema(NuggetIndexError):
    """RelationSchema YAML failed validation (missing fields, invalid types)."""


class JudgeTimeout(NuggetIndexError):
    """LLM-as-judge call exceeded the configured timeout."""


class ChainAmbiguousError(NuggetIndexError):
    """Multiple (or zero) valid candidates at a chain step; no resolver configured.

    Raised by :meth:`NuggetStore.achain_rename` and
    :meth:`NuggetStore.achain_join` when a deterministic walk cannot pick a
    single next hop and no :class:`ChainResolver` was supplied.

    Attributes
    ----------
    subject:
        The subject of the nugget at which the ambiguity was hit.
    candidates:
        The tied candidates (list of :class:`Nugget`).
    step:
        0-indexed position in the walk where the ambiguity arose; ``-1`` for
        join steps that don't have a meaningful step index.
    """

    def __init__(
        self,
        subject: str,
        candidates: list[Any],
        step: int,
    ) -> None:
        super().__init__(
            f"Ambiguous rename/join at step {step} for subject {subject!r}: "
            f"{len(candidates)} candidates. Pass resolver=ChainResolver(...) "
            f"to delegate to an LLM, or disambiguate manually."
        )
        self.subject = subject
        self.candidates = candidates
        self.step = step


class ChainCycleDetected(NuggetIndexError):
    """Cycle detected in rename graph; walker terminated cleanly. Usually benign."""


class ChainDepthExceeded(NuggetIndexError):
    """Hit max_depth before reaching termination. Chain returned with ``truncated=True``."""
