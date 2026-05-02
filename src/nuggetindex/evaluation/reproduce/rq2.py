"""RQ2 — Temporal correctness on TimeQA.

Reproduces Table 2: fraction of answers whose supporting nuggets remain
valid at the queried timestamp. Measured with ``TemporalFaithfulness``.
"""

from __future__ import annotations

from nuggetindex.evaluation.reproduce._common import (
    print_banner,
    print_markdown_table,
    try_load_dataset,
)


def main() -> int:
    print_banner(
        "RQ2",
        "Temporal correctness (TimeQA)",
        "Paper §6.2, Table 2",
    )

    ds = try_load_dataset(
        "timeqa/timeqa",
        source_url="https://github.com/wenhuchen/Time-Sensitive-QA",
        license_note="CC-BY-SA-4.0 (see upstream README)",
    )
    if ds is None:
        print("\n[v0.1 skeleton] No data wired — emitting placeholder table.")
        print_markdown_table(
            headers=("system", "temporal_faithfulness"),
            rows=[
                ("nuggetindex", "n/a"),
                ("passage-rag baseline", "n/a"),
            ],
        )
        return 0

    print("\n[v0.1 skeleton] Dataset loaded but pipeline is not yet wired.")
    print_markdown_table(
        headers=("system", "temporal_faithfulness"),
        rows=[
            ("nuggetindex", "pending"),
            ("passage-rag baseline", "pending"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
