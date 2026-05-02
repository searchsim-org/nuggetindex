"""CAPTCHA and rate-limit classifier.

Given an HTTP response, decides whether the response represents an honest
result, a hard CAPTCHA challenge, a soft rate-limit signal, or a
SearXNG-specific empty-because-upstream-blocked state. Pure heuristic; never
hits the network. Extensible via ``extra_patterns``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_Category = Literal[
    "ok",
    "rate_limit",
    "cloudflare",
    "google_sorry",
    "ddg_anomaly",
    "bing_captcha",
    "searxng_silent_empty",
    "unknown_block",
]


@dataclass(frozen=True)
class DetectionResult:
    is_captcha: bool
    category: _Category
    reason: str


_PATTERNS: list[tuple[re.Pattern[str], _Category]] = [
    (re.compile(r"cf-browser-verification|Just a moment\.\.\.", re.I), "cloudflare"),
    (re.compile(r"/sorry/index|unusual traffic", re.I), "google_sorry"),
    (re.compile(r"anomaly-modal__title|bots use DuckDuckGo", re.I), "ddg_anomaly"),
    (re.compile(r"captchaform|captchaAnswer|/fd/ls/CAPT", re.I), "bing_captcha"),
]


@dataclass
class CaptchaDetector:
    extra_patterns: list[tuple[re.Pattern[str], str]] = field(default_factory=list)

    def classify(
        self,
        *,
        status_code: int,
        body: str,
        headers: dict[str, str],
        searxng_empty: bool = False,
        searxng_engines_failed: bool = False,
    ) -> DetectionResult:
        if status_code == 429:
            return DetectionResult(True, "rate_limit", "HTTP 429")
        if status_code in (403, 503):
            cat = self._match_body(body)
            if cat is not None:
                return DetectionResult(True, cat, f"HTTP {status_code} + {cat}")
            return DetectionResult(True, "unknown_block", f"HTTP {status_code}")
        cat = self._match_body(body)
        if cat is not None:
            return DetectionResult(True, cat, f"body matches {cat} pattern")
        if searxng_empty and searxng_engines_failed:
            return DetectionResult(
                True,
                "searxng_silent_empty",
                "SearXNG returned 200 with zero results and engine errors",
            )
        return DetectionResult(False, "ok", "no CAPTCHA signals")

    def _match_body(self, body: str) -> _Category | None:
        for pattern, cat in _PATTERNS:
            if pattern.search(body):
                return cat  # type: ignore[return-value]
        for pattern, cat in self.extra_patterns:
            if pattern.search(body):
                return cat  # type: ignore[return-value]
        return None
