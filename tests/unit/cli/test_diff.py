"""``nuggetindex diff`` CLI tests."""

from __future__ import annotations

import shutil
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
    stands in for the LLM extractor — keeps these diff tests offline after
    ``RuleBasedExtractor`` was moved out of the library."""

    def _stub(model: str) -> object:  # noqa: ARG001
        return RuleBasedExtractor()

    monkeypatch.setattr(build_module, "_build_extractor", _stub)


def _build(tmp_path: Path, db_name: str, files: dict[str, str]) -> Path:
    corpus = tmp_path / f"corpus-{db_name}"
    corpus.mkdir()
    for name, text in files.items():
        (corpus / name).write_text(text)
    db = tmp_path / db_name
    result = runner.invoke(app, ["build", str(corpus), "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


def test_diff_reports_additions(tmp_path: Path) -> None:
    old = _build(tmp_path, "old.db", {"a.txt": "Sundar Pichai is CEO of Google.\n"})
    # Start new from the old db then ingest another file so we get an addition.
    new = tmp_path / "new.db"
    shutil.copy(old, new)
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "c.txt").write_text("Tim Cook is CEO of Apple.\n")
    runner.invoke(app, ["ingest", str(extra), "--db", str(new)])

    result = runner.invoke(app, ["diff", "--old", str(old), "--new", str(new)])
    assert result.exit_code == 0, result.output
    assert "added" in result.stdout
    assert "removed" in result.stdout


def test_diff_missing_old_errors(tmp_path: Path) -> None:
    new = _build(tmp_path, "new.db", {"a.txt": "x is y.\n"})
    result = runner.invoke(
        app,
        ["diff", "--old", str(tmp_path / "nope.db"), "--new", str(new)],
    )
    assert result.exit_code != 0
