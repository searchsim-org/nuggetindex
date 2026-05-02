"""Evaluation utilities for nuggetindex.

Public surface:
    - ``attach_nugget_metadata``: enrich a Hugging Face ``Dataset`` with
      nugget-level governance metadata so that downstream Ragas metrics can
      score temporal/conflict behaviour.

Ragas-backed metrics (``TemporalFaithfulness``, ``ConflictTransparency``)
and the reproducibility scripts under ``evaluation.reproduce`` are
*lazily* importable — they live in modules guarded by ``_require_ragas``
so that callers who only need the dataset adapter don't have to pay the
Ragas install cost.
"""

from nuggetindex.evaluation.dataset_adapter import attach_nugget_metadata

__all__ = ["attach_nugget_metadata"]
