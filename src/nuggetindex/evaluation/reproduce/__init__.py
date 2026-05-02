"""Reproducibility scripts for the nuggetindex paper figures.

One script per research question:

- ``rq1`` — Nugget recall vs. passage recall on RAVine.
- ``rq2`` — Temporal correctness on TimeQA.
- ``rq3`` — Conflict rate under deliberate contradictions.
- ``rq4`` — Context token efficiency vs. raw-passage RAG.
- ``rq5`` — Multi-hop recall on HotpotQA-style queries.

Each script is invoked as ``python -m nuggetindex.evaluation.reproduce.rqN``.
v0.1 ships *skeletons*: they attempt to load the upstream dataset, print a
clear source/licence message if it isn't cached locally, and emit a
Markdown table (placeholder rows when no data is wired).
"""
