"""Rotating proxy pool with quarantine-based health tracking.

Supports HTTP / HTTPS / SOCKS5 proxies. Quarantines entries that return errors;
releases them back into rotation after ``quarantine_duration`` expires. Used by
both the SearXNG HTTP client and the Camoufox browser fallback.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import cycle


@dataclass
class ProxyEntry:
    url: str
    quarantined_until: datetime | None = None
    consecutive_failures: int = 0
    last_latency_ms: float | None = None


@dataclass
class ProxyPool:
    proxies: list[str | ProxyEntry]
    quarantine_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    max_consecutive_failures: int = 3
    _entries: list[ProxyEntry] = field(init=False, default_factory=list)
    _cursor: Iterator[int] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._entries = [
            p if isinstance(p, ProxyEntry) else ProxyEntry(url=p)
            for p in self.proxies
        ]
        self._cursor = cycle(range(len(self._entries))) if self._entries else None

    def next(self) -> ProxyEntry | None:
        """Return the next active proxy, or None when the pool is empty.
        Raises RuntimeError when every proxy is currently quarantined."""
        if not self._entries or self._cursor is None:
            return None
        now = datetime.now(tz=UTC)
        for _ in range(len(self._entries)):
            idx = next(self._cursor)
            entry = self._entries[idx]
            if entry.quarantined_until is None or entry.quarantined_until <= now:
                entry.quarantined_until = None
                return entry
        raise RuntimeError(
            "ProxyPool exhausted: all proxies are quarantined. "
            "Wait for quarantine_duration to expire, or supply fresh proxies."
        )

    def mark_failed(self, entry: ProxyEntry) -> None:
        entry.consecutive_failures += 1
        entry.quarantined_until = datetime.now(tz=UTC) + self.quarantine_duration

    def mark_success(self, entry: ProxyEntry, *, latency_ms: float | None = None) -> None:
        entry.consecutive_failures = 0
        entry.quarantined_until = None
        entry.last_latency_ms = latency_ms

    def __len__(self) -> int:
        return len(self._entries)

    def active_count(self) -> int:
        now = datetime.now(tz=UTC)
        return sum(
            1 for e in self._entries
            if e.quarantined_until is None or e.quarantined_until <= now
        )
