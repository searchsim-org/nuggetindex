"""RQ5 — Multi-hop recall on HotpotQA-style queries.

Reproduces Table 5: fraction of multi-hop questions for which *every*
supporting nugget is present in top-k. Highlights the recall gap vs.
single-hop passage retrieval.
"""

from __future__ import annotations

from nuggetindex.evaluation.reproduce._common import (
    print_banner,
    print_markdown_table,
    try_load_dataset,
)


def main() -> int:
    print_banner(
        "RQ5",
        "Multi-hop recall",
        "Paper §6.5, Table 5",
    )

    ds = try_load_dataset(
        "hotpot_qa",
        source_url="https://hotpotqa.github.io/",
        license_note="CC-BY-SA-4.0",
    )
    if ds is None:
        print("\n[v0.1 skeleton] No data wired — emitting placeholder table.")
        print_markdown_table(
            headers=("k", "all_hops_recalled"),
            rows=[
                (5, "n/a"),
                (10, "n/a"),
                (20, "n/a"),
            ],
        )
        return 0

    print("\n[v0.1 skeleton] Dataset loaded but pipeline is not yet wired.")
    print_markdown_table(
        headers=("k", "all_hops_recalled"),
        rows=[
            (5, "pending"),
            (10, "pending"),
            (20, "pending"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
