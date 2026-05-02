"""BaseExtractor ABC + ExtractionResult.

Every extractor implementation produces a list of `ExtractionResult` objects.
Each result wraps a freshly-minted `Nugget` plus the extractor's confidence
and optional rationale. Downstream components (QualityGate, pipeline) consume
this shape uniformly regardless of whether the backend is rule-based or LLM.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from nuggetindex.core.models import Nugget


class ExtractionResult(BaseModel):
    """Single extraction outcome: a nugget with per-extraction confidence."""

    nugget: Nugget
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str | None = None


class BaseExtractor(ABC):
    """Extract atomic nuggets from natural-language text.

    Subclasses override :meth:`aextract`. The recommended 0.2+ signature is::

        async def aextract(
            self,
            text: str,
            *,
            context: str = "",
            source_id: str | None = None,
        ) -> list[ExtractionResult]: ...

    Accepting ``source_id`` stamps it into each emitted nugget's provenance,
    replacing the subclass's default ``source_id``. The pipeline checks the
    subclass's signature at registration time via :func:`accepts_source_id`
    and only passes ``source_id`` when supported, so subclasses predating 0.2
    keep working unchanged.
    """

    @abstractmethod
    async def aextract(
        self, text: str, *, context: str = ""
    ) -> list[ExtractionResult]:  # pragma: no cover - ABC
        ...

    def extract(self, text: str, *, context: str = "") -> list[ExtractionResult]:
        """Synchronous wrapper around `aextract`.

        Uses `asyncio.run`, so it must not be called from inside a running
        event loop; use `aextract` directly in async contexts.
        """
        return asyncio.run(self.aextract(text, context=context))


def accepts_source_id(extractor: BaseExtractor) -> bool:
    """Return True iff ``extractor.aextract`` declares a ``source_id`` kwarg.

    Used by :class:`nuggetindex.pipeline.constructor.DocumentConstructor` to
    decide whether to forward the current document's ``source_id`` to the
    extractor. Subclasses predating the 0.2 convention lack the parameter and
    so the pipeline omits it for them.
    """
    sig = inspect.signature(extractor.aextract)
    return "source_id" in sig.parameters
