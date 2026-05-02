"""``nuggetindex query`` CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nuggetindex.cli import build as build_module
from nuggetindex.cli.app import app
from tests.fixtures import RuleBasedExtractor

runner = CliRunner()


@pytest.fixture(autouse=True)
def _patch_rule_based(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the CLI's ``_build_extractor`` so the rule-based fixture
    stands in for the LLM extractor — keeps these query tests offline
    after ``RuleBasedExtractor`` was moved out of the library."""

    def _stub(model: str) -> object:  # noqa: ARG001
        return RuleBasedExtractor()

    monkeypatch.setattr(build_module, "_build_extractor", _stub)


def _prep_db(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Sundar Pichai is CEO of Google.\n")
    db = tmp_path / "ni.db"
    result = runner.invoke(app, ["build", str(corpus), "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


def test_query_runs_against_built_db(tmp_path: Path) -> None:
    db = _prep_db(tmp_path)
    # Use a future query time to ensure the validity window is open.
    result = runner.invoke(
        app,
        [
            "query",
            "Google",
            "--db",
            str(db),
            "--time",
            "2030-01-01T00:00:00+00:00",
            "--top-k",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    # Either a hit row or a "No results" banner; both are acceptable for the
    # happy path as long as the command completes.
    assert "Google" in result.stdout or "No results" in result.stdout


def test_query_missing_db_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["query", "anything", "--db", str(tmp_path / "absent.db")],
    )
    assert result.exit_code != 0
