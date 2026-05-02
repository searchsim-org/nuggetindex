"""Eval harness for proving sidecar value in metric form.

Runs a baseline retriever against a sidecar-augmented one on a small
benchmark and reports diff-style metrics (exact-match, F1, a list of
queries fixed or broken by the sidecar). The harness is offline by
default: the ``"sanity"`` benchmark ships with 10 hand-crafted queries so
the module can be exercised without any external downloads.

Public API::

    from nuggetindex.eval import (
        BenchmarkQuery, EvalResult, EvalReport, run_eval,
    )
"""

from __future__ import annotations

from nuggetindex.eval.benchmarks import (
    BenchmarkQuery,
    load_benchmark,
    load_sanity_benchmark,
)
from nuggetindex.eval.harness import EvalReport, EvalResult, run_eval
from nuggetindex.eval.metrics import exact_match, f1_score

__all__ = [
    "BenchmarkQuery",
    "EvalReport",
    "EvalResult",
    "exact_match",
    "f1_score",
    "load_benchmark",
    "load_sanity_benchmark",
    "run_eval",
]
