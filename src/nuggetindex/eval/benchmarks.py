"""Benchmark loaders for :mod:`nuggetindex.eval`.

Three flavours:

* ``"sanity"`` — ships with the package, 10 hand-crafted temporal queries
  drawn from the canonical Google / Microsoft / Apple / Twitter / Facebook
  governance facts the rest of the codebase uses in tests. No external
  download, no network.
* ``"timeqa"`` — loads the TimeQA dataset via the optional ``datasets``
  extra. Raises :class:`ImportError` with a pointer to
  ``pip install nuggetindex[eval]`` when the extra is missing.
* ``"situatedqa"`` — same gating, same pattern.

The loaders all normalise onto :class:`BenchmarkQuery` so the harness
doesn't care which source a query came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class BenchmarkQuery:
    """A single benchmark query + its reference answer.

    Attributes
    ----------
    query:
        The natural-language question.
    expected_answer:
        The reference answer string (exact-match normaliser is applied
        on both sides during scoring).
    query_time:
        Optional temporal anchor. The sidecar uses this to filter
        validity-bounded facts; benchmarks without a time dimension
        leave it ``None``.
    metadata:
        Free-form metadata passed through from the underlying dataset.
    """

    query: str
    expected_answer: str
    query_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Sanity benchmark — always available, no external deps
# --------------------------------------------------------------------------- #


def _dt(year: int, month: int = 6, day: int = 15) -> datetime:
    """UTC-tagged datetime shortcut for the sanity fixtures."""
    return datetime(year, month, day, tzinfo=UTC)


_SANITY_QUERIES: tuple[BenchmarkQuery, ...] = (
    BenchmarkQuery(
        query="Who was Google's CEO in 2013?",
        expected_answer="Larry Page",
        query_time=_dt(2013),
        metadata={"topic": "google-ceo", "kind": "temporal"},
    ),
    BenchmarkQuery(
        query="Who was Google's CEO in 2018?",
        expected_answer="Sundar Pichai",
        query_time=_dt(2018),
        metadata={"topic": "google-ceo", "kind": "temporal"},
    ),
    BenchmarkQuery(
        query="Who was Microsoft's CEO in 2010?",
        expected_answer="Steve Ballmer",
        query_time=_dt(2010),
        metadata={"topic": "microsoft-ceo", "kind": "temporal"},
    ),
    BenchmarkQuery(
        query="Who was Microsoft's CEO in 2016?",
        expected_answer="Satya Nadella",
        query_time=_dt(2016),
        metadata={"topic": "microsoft-ceo", "kind": "temporal"},
    ),
    BenchmarkQuery(
        query="Who was Apple's CEO in 2010?",
        expected_answer="Steve Jobs",
        query_time=_dt(2010),
        metadata={"topic": "apple-ceo", "kind": "temporal"},
    ),
    BenchmarkQuery(
        query="Who was Apple's CEO in 2014?",
        expected_answer="Tim Cook",
        query_time=_dt(2014),
        metadata={"topic": "apple-ceo", "kind": "temporal"},
    ),
    BenchmarkQuery(
        query="What was Twitter renamed to?",
        expected_answer="X Corp.",
        query_time=_dt(2023),
        metadata={"topic": "twitter-rename", "kind": "rename"},
    ),
    BenchmarkQuery(
        query="Who was Facebook's CEO in 2012?",
        expected_answer="Mark Zuckerberg",
        query_time=_dt(2012),
        metadata={"topic": "facebook-ceo", "kind": "temporal"},
    ),
    BenchmarkQuery(
        query="What was Facebook renamed to?",
        expected_answer="Meta Platforms",
        query_time=_dt(2021),
        metadata={"topic": "facebook-rename", "kind": "rename"},
    ),
    BenchmarkQuery(
        query="Who was Twitter's CEO in 2020?",
        expected_answer="Jack Dorsey",
        query_time=_dt(2020),
        metadata={"topic": "twitter-ceo", "kind": "temporal"},
    ),
)


def load_sanity_benchmark() -> list[BenchmarkQuery]:
    """Return the 10-query sanity benchmark.

    Always available; never touches the network. Useful as a smoke test
    for new retrievers / sidecars and as a CI fixture for the eval
    harness itself.
    """
    return list(_SANITY_QUERIES)


# --------------------------------------------------------------------------- #
# External benchmark loaders — require the optional ``[eval]`` extra
# --------------------------------------------------------------------------- #


_EVAL_INSTALL_HINT = (
    "TimeQA / SituatedQA loaders require the 'datasets' package. "
    "Install with: pip install nuggetindex[eval]"
)


def _require_datasets() -> Any:
    """Import ``datasets`` or raise a helpful :class:`ImportError`."""
    try:
        import datasets  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover -- covered via gate
        raise ImportError(_EVAL_INSTALL_HINT) from exc
    return datasets


def load_timeqa(*, max_queries: int | None = None) -> list[BenchmarkQuery]:
    """Load TimeQA via HuggingFace Datasets.

    Returns a list of :class:`BenchmarkQuery`; raises
    :class:`ImportError` if the ``datasets`` package is absent. The
    loader is intentionally minimal: TimeQA has multiple splits and
    answer formats, so we pull the validation split and coerce the
    first reference answer to a string.
    """
    ds = _require_datasets()  # pragma: no cover -- external download gated
    raw = ds.load_dataset("timeqa", split="validation")  # pragma: no cover
    out: list[BenchmarkQuery] = []  # pragma: no cover
    for row in raw:  # pragma: no cover
        question = str(row.get("question", ""))
        answers = row.get("answers") or row.get("answer") or []
        if isinstance(answers, dict):
            answers = answers.get("text", [])
        answer = str(answers[0]) if answers else ""
        time_str = row.get("time") or row.get("question_time")
        qt = _parse_query_time(time_str)
        out.append(
            BenchmarkQuery(
                query=question,
                expected_answer=answer,
                query_time=qt,
                metadata={"dataset": "timeqa"},
            )
        )
        if max_queries is not None and len(out) >= max_queries:
            break
    return out


def load_situatedqa(*, max_queries: int | None = None) -> list[BenchmarkQuery]:
    """Load SituatedQA (temporal split) via HuggingFace Datasets.

    Same gating and coercion contract as :func:`load_timeqa`.
    """
    ds = _require_datasets()  # pragma: no cover
    raw = ds.load_dataset("situatedqa", "temporal", split="test")  # pragma: no cover
    out: list[BenchmarkQuery] = []  # pragma: no cover
    for row in raw:  # pragma: no cover
        question = str(row.get("question", ""))
        answers = row.get("any_answer") or row.get("answer") or []
        if isinstance(answers, dict):
            answers = answers.get("text", [])
        answer = str(answers[0]) if answers else ""
        time_str = row.get("date") or row.get("edit_time")
        qt = _parse_query_time(time_str)
        out.append(
            BenchmarkQuery(
                query=question,
                expected_answer=answer,
                query_time=qt,
                metadata={"dataset": "situatedqa"},
            )
        )
        if max_queries is not None and len(out) >= max_queries:
            break
    return out


def _parse_query_time(raw: Any) -> datetime | None:
    """Best-effort parse of a benchmark row's time column."""
    if raw is None:
        return None
    s = str(raw).strip()  # pragma: no cover -- external data only
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:  # pragma: no cover -- best-effort fallback
        # Year-only ("2013") -> mid-year anchor.
        try:
            year = int(s[:4])
            return datetime(year, 6, 15, tzinfo=UTC)
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Unified dispatcher
# --------------------------------------------------------------------------- #


def load_benchmark(
    name: str,
    *,
    max_queries: int | None = None,
) -> list[BenchmarkQuery]:
    """Dispatch on benchmark name; return a list of :class:`BenchmarkQuery`.

    Raises :class:`ValueError` for unknown names and :class:`ImportError`
    (with an install hint) when the caller asks for a dataset gated on
    the ``[eval]`` extra.
    """
    key = name.strip().lower()
    if key == "sanity":
        queries = load_sanity_benchmark()
    elif key == "timeqa":
        queries = load_timeqa(max_queries=max_queries)
    elif key == "situatedqa":
        queries = load_situatedqa(max_queries=max_queries)
    else:
        raise ValueError(
            f"Unknown benchmark {name!r}; expected one of 'sanity', 'timeqa', 'situatedqa'."
        )
    if max_queries is not None:
        return queries[:max_queries]
    return queries


__all__ = [
    "BenchmarkQuery",
    "load_benchmark",
    "load_sanity_benchmark",
    "load_situatedqa",
    "load_timeqa",
]
