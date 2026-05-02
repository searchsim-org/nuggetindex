"""Tests for ``nuggetindex chain`` CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nuggetindex import NuggetStore
from nuggetindex.cli.app import app
from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)

runner = CliRunner()


def _google_ceo(obj: str, start: int, end: int | None) -> Nugget:
    vi = ValidityInterval(
        start=datetime(start, 1, 1, tzinfo=UTC),
        end=datetime(end, 1, 1, tzinfo=UTC) if end else None,
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            # Canonical predicate; CLI now canonicalises aliases at query time.
            predicate="chiefExecutiveOfficer",
            object=obj,
            text=f"{obj} is CEO",
        ),
        validity=vi,
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id=f"doc-{obj}",
                evidence_span=f"{obj} is CEO",
            ),
        ),
    )


@pytest.fixture
def succession_db(tmp_path: Path) -> Path:
    db = tmp_path / "chain.db"
    store = NuggetStore(db_path=db)
    try:
        for n in [
            _google_ceo("Schmidt", 2001, 2011),
            _google_ceo("Page", 2011, 2015),
            _google_ceo("Pichai", 2015, None),
        ]:
            store.add(n)
    finally:
        store.close()
    return db


def test_chain_succession_renders_table(succession_db: Path) -> None:
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "succession",
            "--subject",
            "Google",
            "--predicate",
            "ceo",
            "--db",
            str(succession_db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Schmidt" in result.stdout
    assert "Pichai" in result.stdout


def test_chain_missing_db_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "succession",
            "--subject",
            "X",
            "--predicate",
            "y",
            "--db",
            str(tmp_path / "absent.db"),
        ],
    )
    assert result.exit_code != 0


def test_chain_succession_missing_predicate_errors(succession_db: Path) -> None:
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "succession",
            "--subject",
            "Google",
            "--db",
            str(succession_db),
        ],
    )
    assert result.exit_code != 0


def test_chain_json_output(succession_db: Path) -> None:
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "succession",
            "--subject",
            "Google",
            "--predicate",
            "ceo",
            "--db",
            str(succession_db),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["chain_type"] == "succession"
    assert len(payload["nuggets"]) == 3


def test_chain_invalid_type_rejected(succession_db: Path) -> None:
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "nonsense",
            "--subject",
            "Google",
            "--predicate",
            "ceo",
            "--db",
            str(succession_db),
        ],
    )
    assert result.exit_code != 0


def test_chain_rename_empty_is_ok(succession_db: Path) -> None:
    # Subject has no rename edges in the fixture.
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "rename",
            "--subject",
            "Google",
            "--db",
            str(succession_db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No results" in result.stdout


def test_chain_join_requires_start_pair(succession_db: Path) -> None:
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "join",
            "--db",
            str(succession_db),
        ],
    )
    assert result.exit_code != 0


def test_chain_discover_lists_candidate_keys(tmp_path: Path) -> None:
    """When the exact-match chain is empty and --discover is set, render a
    Rich table of nearby (subject, predicate, scope) triples.
    """
    db = tmp_path / "discover.db"
    store = NuggetStore(db_path=db)
    try:
        store.add(_google_ceo("Schmidt", 2001, 2011))
    finally:
        store.close()

    # Query a SUBJECT that won't match (no nuggets for "Acme"), but ask
    # the discovery helper for anything with "google" in the subject.
    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "succession",
            "--subject",
            "google",  # lowercased substring
            "--predicate",
            "ceo",
            "--discover",
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    # The candidate-keys table should surface the real key.
    assert "Google" in result.stdout
    assert "chiefExecutiveOfficer" in result.stdout


def test_chain_no_discover_hint_on_empty(tmp_path: Path) -> None:
    """When the chain is empty and --discover is NOT set, print the hint."""
    db = tmp_path / "empty.db"
    store = NuggetStore(db_path=db)
    try:
        store.add(_google_ceo("Schmidt", 2001, 2011))
    finally:
        store.close()

    result = runner.invoke(
        app,
        [
            "chain",
            "--type",
            "succession",
            "--subject",
            "Nobody",
            "--predicate",
            "ceo",
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--discover" in result.stdout
