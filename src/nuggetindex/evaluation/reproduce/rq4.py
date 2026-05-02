"""RQ4 — Context token efficiency vs. raw-passage RAG.

Reproduces Table 4: tokens-in-context per correct answer for nugget-level
retrieval vs. passage-level retrieval. The claim is that nugget retrieval
is sharper per token even when top-k is held constant.
"""

from __future__ import annotations

from nuggetindex.evaluation.reproduce._common import (
    print_banner,
    print_markdown_table,
    try_load_dataset,
)


def main() -> int:
    print_banner(
        "RQ4",
        "Context token efficiency",
        "Paper §6.4, Table 4",
    )

    ds = try_load_dataset(
        "ravine-bench/ravine",
        source_url="https://huggingface.co/datasets/ravine-bench/ravine",
        license_note="Apache-2.0 (check dataset card)",
    )
    if ds is None:
        print("\n[v0.1 skeleton] No data wired — emitting placeholder table.")
        print_markdown_table(
            headers=("system", "mean_tokens", "answer_f1"),
            rows=[
                ("nuggetindex", "n/a", "n/a"),
                ("passage-rag baseline", "n/a", "n/a"),
            ],
        )
        return 0

    print("\n[v0.1 skeleton] Dataset loaded but pipeline is not yet wired.")
    print_markdown_table(
        headers=("system", "mean_tokens", "answer_f1"),
        rows=[
            ("nuggetindex", "pending", "pending"),
            ("passage-rag baseline", "pending", "pending"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
