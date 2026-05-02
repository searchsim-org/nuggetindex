"""``nuggetindex auto`` CLI tests.

Covers the new ``--corpus-url`` / ``--corpus-name`` / ``--bootstrap`` /
``--sample-size`` surface. :class:`VespaCorpus` is monkey-patched to a
no-network stub so tests never hit a real cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from nuggetindex.cli.app import app
from nuggetindex.pipeline.constructor import Document

runner = CliRunner()


@pytest.fixture
def jsonl_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "docs.jsonl"
    lines = [
        json.dumps({"source_id": f"d{i}", "text": text})
        for i, text in enumerate(
            [
                "Larry Page became CEO of Google in 2011.",
                "Sundar Pichai became CEO of Google in 2015.",
                "Microsoft acquired LinkedIn for 26 billion.",
                "Microsoft acquired GitHub.",
                "Twitter was renamed to X in 2023.",
                "Facebook was renamed to Meta.",
            ]
        )
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _synthetic_docs(n: int) -> list[Document]:
    return [
        Document(
            source_id=f"v{i}",
            text=f"vespa doc {i}: research study on recipes and tutorials.",
        )
        for i in range(n)
    ]


def test_auto_cli_rejects_mismatched_corpus_flags(tmp_path: Path) -> None:
    """--corpus-url without --corpus-name (or vice versa) is an error."""
    result = runner.invoke(
        app,
        [
            "auto",
            "--corpus-url",
            "http://vespa.test",
            "--store-path",
            str(tmp_path / "store.db"),
            "--cache-path",
            str(tmp_path / "cache.db"),
        ],
    )
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--corpus-url" in combined and "--corpus-name" in combined


def test_auto_cli_rejects_caller_bootstrap_with_corpus(
    tmp_path: Path,
) -> None:
    """--corpus-url with default --bootstrap=caller is a clear error."""
    result = runner.invoke(
        app,
        [
            "auto",
            "--corpus-url",
            "http://vespa.test",
            "--corpus-name",
            "my-corpus",
            "--store-path",
            str(tmp_path / "store.db"),
            "--cache-path",
            str(tmp_path / "cache.db"),
        ],
    )
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "bootstrap" in combined


def test_auto_cli_builds_vespa_corpus_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full run: CLI wires VespaCorpus -> auto() with bootstrap sampling."""
    captured: dict[str, Any] = {}

    class StubCorpus:
        def __init__(self, *, base_url: str, corpus: str, **kwargs: Any) -> None:
            captured["base_url"] = base_url
            captured["corpus"] = corpus
            self.sample = AsyncMock(return_value=_synthetic_docs(6))
            self.search = AsyncMock(return_value=[])
            self.aclose = AsyncMock()

    # Monkey-patch the class in the adapters package so the CLI's
    # late-import `from nuggetindex.adapters import VespaCorpus` resolves
    # to our stub.
    import nuggetindex.adapters as adapters_pkg

    monkeypatch.setattr(adapters_pkg, "VespaCorpus", StubCorpus)

    result = runner.invoke(
        app,
        [
            "auto",
            "--corpus-url",
            "http://vespa.test",
            "--corpus-name",
            "my-corpus",
            "--bootstrap",
            "topic_diverse",
            "--sample-size",
            "6",
            "--budget",
            "3",
            "--store-path",
            str(tmp_path / "store.db"),
            "--cache-path",
            str(tmp_path / "cache.db"),
        ],
    )
    assert result.exit_code == 0, (
        (result.stdout or "") + "\n---stderr---\n" + (result.stderr or "")
    )
    assert captured == {"base_url": "http://vespa.test", "corpus": "my-corpus"}
    # The Markdown report dump includes "docs processed".
    assert "docs processed" in result.stdout


def test_auto_cli_rejects_unknown_bootstrap_value(tmp_path: Path) -> None:
    """Typos on --bootstrap are rejected with a helpful error."""
    result = runner.invoke(
        app,
        [
            "auto",
            "--corpus-url",
            "http://vespa.test",
            "--corpus-name",
            "my-corpus",
            "--bootstrap",
            "totally-diverse",
            "--store-path",
            str(tmp_path / "store.db"),
            "--cache-path",
            str(tmp_path / "cache.db"),
        ],
    )
    assert result.exit_code != 0


def test_auto_cli_still_supports_index_path(
    jsonl_corpus: Path, tmp_path: Path
) -> None:
    """Backward-compat: --index alone still drives the old pipeline."""
    # Silence MagicMock warnings and just drive the real tiny pipeline.
    result = runner.invoke(
        app,
        [
            "auto",
            "--index",
            str(jsonl_corpus),
            "--budget",
            "3",
            "--store-path",
            str(tmp_path / "store.db"),
            "--cache-path",
            str(tmp_path / "cache.db"),
        ],
    )
    assert result.exit_code == 0, (
        (result.stdout or "") + "\n---stderr---\n" + (result.stderr or "")
    )
    assert "docs processed" in result.stdout
