"""Freshness checking helper for the Sidecar runtime.

Given a (subject, predicate, scope) key and a threshold, decides whether the
store's evidence for that key is fresh enough. When stale, the Sidecar
invokes the fallback CorpusSource (typically WebSearchCorpus) to augment the
context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuggetindex.store import NuggetStore


@dataclass(frozen=True)
class FreshnessResult:
    is_fresh: bool
    latest: datetime | None
    age: timedelta | None
    reason: str


@dataclass
class FreshnessChecker:
    """Threshold-based freshness verdict over a store's evidence."""

    threshold: timedelta = timedelta(days=90)

    def is_fresh(
        self,
        *,
        latest: datetime | None,
        now: datetime | None = None,
    ) -> bool:
        return self.check(latest=latest, now=now).is_fresh

    def check(
        self,
        *,
        latest: datetime | None,
        now: datetime | None = None,
    ) -> FreshnessResult:
        now = now or datetime.now(tz=UTC)
        if latest is None:
            return FreshnessResult(
                is_fresh=False,
                latest=None,
                age=None,
                reason="no evidence for key",
            )
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=UTC)
        age = now - latest
        if age <= self.threshold:
            return FreshnessResult(
                is_fresh=True,
                latest=latest,
                age=age,
                reason=f"youngest evidence is {age.days}d old",
            )
        return FreshnessResult(
            is_fresh=False,
            latest=latest,
            age=age,
            reason=(f"youngest evidence is {age.days}d old (threshold {self.threshold.days}d)"),
        )

    async def check_store(
        self,
        store: NuggetStore,
        *,
        subject: str,
        predicate: str,
        scope: str = "global",
        now: datetime | None = None,
    ) -> FreshnessResult:
        """Return the youngest provenance created_at timestamp on key."""
        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        try:
            row = conn.execute(
                """
                SELECT MAX(json_extract(data, '$.provenance[0].created_at'))
                FROM nuggets
                WHERE subject = ? AND predicate = ? AND scope = ?
                """,
                (subject, predicate, scope),
            ).fetchone()
        finally:
            conn.close()
        latest: datetime | None = None
        if row and row[0]:
            try:
                raw = str(row[0]).replace("Z", "+00:00")
                latest = datetime.fromisoformat(raw)
            except (ValueError, AttributeError):
                latest = None
        return self.check(latest=latest, now=now)
