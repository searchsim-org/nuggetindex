"""RQ3 — Conflict rate under deliberate contradictions.

Reproduces Table 3: on a dataset with known contradictory claims, how
often does the system (a) retrieve the contested key, (b) surface the
disagreement transparently in the answer? Measured with
``ConflictTransparency``.
"""

from __future__ import annotations

from nuggetindex.evaluation.reproduce._common import (
    print_banner,
    print_markdown_table,
    try_load_dataset,
)


def main() -> int:
    print_banner(
        "RQ3",
        "Conflict rate under contradictions",
        "Paper §6.3, Table 3",
    )

    ds = try_load_dataset(
        "nuggetindex/synthetic-conflicts",
        source_url="TBD — synthesised from Wikidata diffs",
        license_note="CC-BY-SA-4.0 (matches upstream Wikidata)",
    )
    if ds is None:
        print("\n[v0.1 skeleton] No data wired — emitting placeholder table.")
        print_markdown_table(
            headers=("system", "conflict_detect_rate", "conflict_transparency"),
            rows=[
                ("nuggetindex", "n/a", "n/a"),
                ("passage-rag baseline", "n/a", "n/a"),
            ],
        )
        return 0

    print("\n[v0.1 skeleton] Dataset loaded but pipeline is not yet wired.")
    print_markdown_table(
        headers=("system", "conflict_detect_rate", "conflict_transparency"),
        rows=[
            ("nuggetindex", "pending", "pending"),
            ("passage-rag baseline", "pending", "pending"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
