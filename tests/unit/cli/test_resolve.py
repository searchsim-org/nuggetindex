"""``nuggetindex resolve`` CLI tests.

The CLI is interactive (typer.prompt). We test it by:
1. Seeding a store with two contested rivals and one deprecated peer.
2. Driving the prompt by piping a stdin transcript that picks winner #1
   for the first key and skips the second (or vice-versa).
3. Verifying the post-conditions on the store via the public API
   (winner is Active+Preferred, loser is Deprecated+Deprecated).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from nuggetindex.cli.app import app
from nuggetindex.core.enums import (
    EpistemicRank,
    LifecycleStatus,
    NuggetKind,
)
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.store import NuggetStore

runner = CliRunner()


def _seed_contested(db: Path) -> tuple[str, str]:
    """Seed two Microsoft/acquiredFor rivals + one Active peer.
    Returns ``(winner_id, loser_id)`` in canonical (Reuters, Bloomberg)
    order so tests can assert on either side."""
    store = NuggetStore(db_path=db)
    winner = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Microsoft",
            predicate="acquiredFor",
            object="$26.2B",
            text="Microsoft paid $26.2B for LinkedIn",
        ),
        validity=ValidityInterval(start=datetime(2016, 6, 13, tzinfo=UTC)),
        epistemic=EpistemicState(status=LifecycleStatus.CONTESTED),
        provenance=(ProvenanceRecord(source_id="reuters", evidence_span="...for $26.2B."),),
    )
    loser = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Microsoft",
            predicate="acquiredFor",
            object="$26.4B",
            text="Microsoft paid $26.4B for LinkedIn",
        ),
        validity=ValidityInterval(start=datetime(2016, 6, 13, tzinfo=UTC)),
        epistemic=EpistemicState(status=LifecycleStatus.CONTESTED),
        provenance=(ProvenanceRecord(source_id="bloomberg", evidence_span="...for $26.4B."),),
    )
    store.add(winner)
    store.add(loser)
    store.close()
    return winner.id, loser.id


def test_resolve_dry_run_does_not_change_store(tmp_path: Path) -> None:
    db = tmp_path / "store.db"
    winner_id, loser_id = _seed_contested(db)
    result = runner.invoke(app, ["resolve", "--store", str(db), "--dry-run"])
    assert result.exit_code == 0, result.output
    # The two contested rivals are still both contested.
    store = NuggetStore(db_path=db)
    w = store.get(winner_id)
    los = store.get(loser_id)
    store.close()
    assert w is not None and w.epistemic.status is LifecycleStatus.CONTESTED
    assert los is not None and los.epistemic.status is LifecycleStatus.CONTESTED


def test_resolve_picks_winner_via_prompt(tmp_path: Path) -> None:
    """Drive the prompt: choose '1' (the lexicographically-first card)."""
    db = tmp_path / "store.db"
    a_id, b_id = _seed_contested(db)
    # Cards are sorted by validity.start (both 2016-06-13 here) then by
    # insertion order; the first is whichever the SQL returned first
    # for that key. We pick winner=1 and assert that exactly one of the
    # two ends up Active+Preferred and the other Deprecated+Deprecated.
    result = runner.invoke(app, ["resolve", "--store", str(db)], input="1\n")
    assert result.exit_code == 0, result.output

    store = NuggetStore(db_path=db)
    a = store.get(a_id)
    b = store.get(b_id)
    store.close()
    assert a is not None and b is not None
    epistemics = {a.epistemic.status: a, b.epistemic.status: b}
    assert LifecycleStatus.ACTIVE in epistemics
    assert LifecycleStatus.DEPRECATED in epistemics
    winner = epistemics[LifecycleStatus.ACTIVE]
    loser = epistemics[LifecycleStatus.DEPRECATED]
    assert winner.epistemic.rank is EpistemicRank.PREFERRED
    assert loser.epistemic.rank is EpistemicRank.DEPRECATED


def test_resolve_skip_leaves_state_alone(tmp_path: Path) -> None:
    db = tmp_path / "store.db"
    a_id, b_id = _seed_contested(db)
    result = runner.invoke(app, ["resolve", "--store", str(db)], input="skip\n")
    assert result.exit_code == 0, result.output
    store = NuggetStore(db_path=db)
    a = store.get(a_id)
    b = store.get(b_id)
    store.close()
    assert a is not None and a.epistemic.status is LifecycleStatus.CONTESTED
    assert b is not None and b.epistemic.status is LifecycleStatus.CONTESTED


def test_resolve_all_wrong_suppresses_every_candidate(tmp_path: Path) -> None:
    db = tmp_path / "store.db"
    a_id, b_id = _seed_contested(db)
    result = runner.invoke(app, ["resolve", "--store", str(db)], input="all-wrong\n")
    assert result.exit_code == 0, result.output
    store = NuggetStore(db_path=db)
    a = store.get(a_id)
    b = store.get(b_id)
    store.close()
    assert a is not None and a.epistemic.status is LifecycleStatus.DEPRECATED
    assert b is not None and b.epistemic.status is LifecycleStatus.DEPRECATED


def test_resolve_no_contested_keys_short_circuits(tmp_path: Path) -> None:
    """Empty store -> CLI exits cleanly without prompting."""
    db = tmp_path / "store.db"
    # Seed an Active-only nugget so the DB exists but has no contested rows.
    store = NuggetStore(db_path=db)
    store.add(
        Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject="Apple",
                predicate="ceo",
                object="Tim Cook",
                text="Tim Cook is CEO of Apple",
            ),
            validity=ValidityInterval(start=datetime(2011, 8, 24, tzinfo=UTC)),
            epistemic=EpistemicState(status=LifecycleStatus.ACTIVE),
            provenance=(ProvenanceRecord(source_id="x", evidence_span="x"),),
        )
    )
    store.close()
    result = runner.invoke(app, ["resolve", "--store", str(db)])
    assert result.exit_code == 0, result.output
    assert "No contested keys" in result.output


def test_resolve_invalid_choice_reprompts(tmp_path: Path) -> None:
    db = tmp_path / "store.db"
    _seed_contested(db)
    # First an invalid token, then a valid '1'. The prompt loop must
    # reject 'banana' and accept '1' on retry.
    result = runner.invoke(app, ["resolve", "--store", str(db)], input="banana\n1\n")
    assert result.exit_code == 0, result.output
    assert "Invalid choice" in result.output
