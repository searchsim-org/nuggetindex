"""RQ1 — Nugget recall vs. passage recall on RAVine.

Reproduces Table 1 of the paper: how often does the retriever surface a
ground-truth nugget in top-k, compared to surfacing the passage it was
extracted from? The hypothesis is that nugget-level retrieval wins on
queries where the relevant fact is buried in a long passage.
"""

from __future__ import annotations

from nuggetindex.evaluation.reproduce._common import (
    print_banner,
    print_markdown_table,
    try_load_dataset,
)


def main() -> int:
    print_banner(
        "RQ1",
        "Nugget recall vs. passage recall (RAVine)",
        "Paper §6.1, Table 1",
    )

    ds = try_load_dataset(
        "ravine-bench/ravine",
        source_url="https://huggingface.co/datasets/ravine-bench/ravine",
        license_note="Apache-2.0 (check dataset card before redistribution)",
    )
    if ds is None:
        # Skeleton mode — emit placeholder rows so a downstream README can
        # still render. These are marked n/a, never synthetic numbers.
        print("\n[v0.1 skeleton] No data wired — emitting placeholder table.")
        print_markdown_table(
            headers=("k", "passage_recall", "nugget_recall"),
            rows=[
                (1, "n/a", "n/a"),
                (3, "n/a", "n/a"),
                (10, "n/a", "n/a"),
            ],
        )
        return 0

    # Real pipeline wiring is left for a follow-up phase — see the Phase 12
    # plan: "Don't bother actually downloading TimeQA / RAVine data in this
    # phase — that's a separate future task."
    print("\n[v0.1 skeleton] Dataset loaded but pipeline is not yet wired.")
    print_markdown_table(
        headers=("k", "passage_recall", "nugget_recall"),
        rows=[
            (1, "pending", "pending"),
            (3, "pending", "pending"),
            (10, "pending", "pending"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
