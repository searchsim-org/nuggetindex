"""``nuggetindex audit`` CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nuggetindex.audit import api as audit_api
from nuggetindex.cli.app import app
from tests.fixtures import RuleBasedExtractor

runner = CliRunner()


def _write_context(tmp_path: Path) -> Path:
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("Sundar Pichai is CEO of Google.\n\nLarry Page is CEO of Google.\n")
    return ctx


@pytest.fixture
def _patch_rule_based(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the audit CLI's implicit extractor resolution always return the
    test-local rule-based fixture, so these tests stay offline after the
    real ``RuleBasedExtractor`` was moved out of the public library."""
    real_resolver = audit_api._resolve_extractor

    def _stub(extractor: object) -> object:
        from nuggetindex.extractors.base import BaseExtractor

        if isinstance(extractor, BaseExtractor):
            return real_resolver(extractor)
        # Any string (including the removed ``rule_based`` sentinel) routes
        # to the fixture extractor in-test.
        return RuleBasedExtractor()

    monkeypatch.setattr(audit_api, "_resolve_extractor", _stub)


def test_audit_json_output(tmp_path: Path, _patch_rule_based: None) -> None:
    ctx = _write_context(tmp_path)
    result = runner.invoke(
        app,
        [
            "audit",
            "--query",
            "Who is CEO of Google?",
            "--context",
            str(ctx),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "conflicts" in payload
    assert len(payload["conflicts"]) >= 1


def test_audit_console_output(tmp_path: Path, _patch_rule_based: None) -> None:
    ctx = _write_context(tmp_path)
    result = runner.invoke(
        app,
        [
            "audit",
            "--query",
            "q",
            "--context",
            str(ctx),
            "--format",
            "console",
        ],
    )
    assert result.exit_code == 0
    assert "Conflicts" in result.stdout or "Audit" in result.stdout


def test_audit_markdown_output(tmp_path: Path, _patch_rule_based: None) -> None:
    ctx = _write_context(tmp_path)
    result = runner.invoke(
        app,
        [
            "audit",
            "--query",
            "q",
            "--context",
            str(ctx),
            "--format",
            "markdown",
        ],
    )
    assert result.exit_code == 0
    assert "# Audit Report" in result.stdout


def test_audit_missing_context_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope.txt"
    result = runner.invoke(
        app,
        [
            "audit",
            "--query",
            "q",
            "--context",
            str(missing),
        ],
    )
    assert result.exit_code != 0


def test_audit_rule_based_model_delegates_to_trigger(
    tmp_path: Path,
) -> None:
    """Legacy ``--model rule_based`` now delegates to the trigger extractor
    with a deprecation warning (softened from the old hard rejection)."""
    ctx = _write_context(tmp_path)
    with pytest.warns(DeprecationWarning, match="rule_based"):
        result = runner.invoke(
            app,
            [
                "audit",
                "--query",
                "q",
                "--context",
                str(ctx),
                "--model",
                "rule_based",
            ],
        )
    assert result.exit_code == 0, result.output
