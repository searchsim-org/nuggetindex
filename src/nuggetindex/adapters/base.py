"""Backend-agnostic corpus access for :func:`nuggetindex.auto.auto`.

``auto()`` used to take a concrete list of :class:`Document` records (or a
JSONL path) which made it only as good as the *caller's* sampling choices.
A hand-curated corporate-seed JSONL biases schema discovery towards
``CEO`` / ``acquired`` / ``founder``; a topic-diverse pull from the same
corpus surfaces ``hasTravelGuide`` / ``isRecipeFor`` / ``reviewedBy`` / etc.

``CorpusSource`` defines the minimum interface ``auto()`` needs to sample a
corpus unbiasedly, so users can "point it at their corpus" and let the
library draw the bootstrap set. Concrete implementations live next to this
module:

* :class:`~nuggetindex.adapters.jsonl.JsonlCorpus` - flat-file source.
* :class:`~nuggetindex.adapters.vespa.VespaCorpus` - live Vespa-style
  BM25 REST endpoint.

Future adapters: ``HaystackCorpus``, ``LlamaIndexCorpus``, ``OpenSearch``,
``Qdrant``. They only need to satisfy the protocol below -- no internal
nuggetindex knobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.pipeline.constructor import Document


@runtime_checkable
class CorpusSource(Protocol):
    """Backend-agnostic corpus access for auto()'s unbiased bootstrap.

    Implementations: :class:`JsonlCorpus` (file), :class:`VespaCorpus`
    (Vespa BM25 REST), and future ``HaystackCorpus`` / ``LlamaIndexCorpus``
    (not in this round).
    """

    async def sample(
        self,
        *,
        mode: Literal["topic_diverse", "uniform", "random_ids"],
        n: int,
    ) -> list[Document]:
        """Return up to ``n`` documents sampled unbiasedly.

        * ``topic_diverse`` - run a built-in pack of broad queries to cover
          many topics. Cheap, works on any BM25 backend, doesn't require
          full-corpus pagination. Default for unknown-shape corpora.
        * ``uniform`` - paginate through the corpus with no ranking
          (offset-based). Accurate but may be slow on large corpora.
        * ``random_ids`` - draw random doc IDs from the corpus. Requires
          backend support; callers get :class:`NotImplementedError` on
          backends that can't do it.
        """
        ...

    async def search(self, query: str, *, limit: int) -> list[Document]:
        """Backend's native keyword search.

        Used by the targeted second pass to pull docs for each seed the
        discovery step proposed.
        """
        ...


# Hand-curated broad topic coverage. Intentionally non-overlapping across:
# cooking, medicine, travel, technical tutorials, politics, sports, science,
# entertainment, finance, education. Deliberately avoid corporate-event verbs
# ("acquired", "CEO", "renamed") that would re-introduce the bias auto()'s
# bootstrap is supposed to avoid.
_TOPIC_DIVERSE_QUERIES: tuple[str, ...] = (
    "recipe",
    "symptoms",
    "travel guide",
    "tutorial",
    "best of 2024",
    "research study",
    "how to fix",
    "interview with",
    "diagnosis",
    "review",
    "conference",
    "lawsuit",
    "university",
    "album release",
    "workout",
    "budget",
    "election",
    "vaccine",
    "festival",
    "stock market",
    "restaurant",
    "fashion",
    "climate",
    "book review",
    "tournament",
    "scholarship",
    "startup",
)


__all__ = ["CorpusSource", "_TOPIC_DIVERSE_QUERIES"]
