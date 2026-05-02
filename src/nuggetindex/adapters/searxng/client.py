"""HTTP client for a self-hosted SearXNG instance.

SearXNG proxies queries to Google, Bing, DuckDuckGo, Brave, Qwant, etc. and
returns a unified JSON. When upstream engines rate-limit or block SearXNG,
responses manifest as 429s, 503s with CF challenges, or silent empty results.
This client detects all three and escalates via proxy rotation; when rotation
is exhausted, it returns a ``SearxngResponse`` with ``was_captcha=True`` so
the caller can fall back to ``CamoufoxBackend``.
"""
from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass, field
from typing import Any

from nuggetindex.adapters.searxng.detect import CaptchaDetector, DetectionResult
from nuggetindex.adapters.searxng.proxy import ProxyPool


@dataclass(frozen=True)
class SearxngResponse:
    results: list[dict]
    was_captcha: bool
    captcha_category: str
    attempts: int
    elapsed_ms: float


@dataclass
class SearxngClient:
    base_url: str
    proxy_pool: ProxyPool | None = None
    detector: CaptchaDetector = field(default_factory=CaptchaDetector)
    timeout: float = 20.0
    http_client: Any | None = None  # inject httpx.AsyncClient for tests / custom transport

    async def search(self, query: str, *, limit: int) -> SearxngResponse:
        started = time.monotonic()
        attempts = 0
        last_detection: DetectionResult | None = None
        max_attempts = max(1, len(self.proxy_pool) if self.proxy_pool else 1)

        # Loop one extra iteration so the non-proxy happy path still fires once.
        for _ in range(max_attempts + 1):
            attempts += 1
            proxy_entry = self._pick_proxy()
            try:
                status, body, headers = await self._get(query, limit, proxy_entry)
            except Exception as exc:  # noqa: BLE001 -- transport flake, move on
                if proxy_entry is not None and self.proxy_pool is not None:
                    self.proxy_pool.mark_failed(proxy_entry)
                last_detection = DetectionResult(True, "unknown_block", f"transport: {exc}")
                continue

            try:
                payload = _json.loads(body or "")
            except _json.JSONDecodeError:
                payload = {"results": []}

            searxng_empty = not payload.get("results")
            searxng_engines_failed = bool(payload.get("unresponsive_engines"))

            detection = self.detector.classify(
                status_code=status, body=body or "", headers=headers,
                searxng_empty=searxng_empty,
                searxng_engines_failed=searxng_engines_failed,
            )
            last_detection = detection

            if not detection.is_captcha:
                if proxy_entry is not None and self.proxy_pool is not None:
                    self.proxy_pool.mark_success(proxy_entry)
                elapsed = (time.monotonic() - started) * 1000
                return SearxngResponse(
                    results=payload.get("results", []),
                    was_captcha=False,
                    captcha_category="ok",
                    attempts=attempts,
                    elapsed_ms=elapsed,
                )
            if proxy_entry is not None and self.proxy_pool is not None:
                self.proxy_pool.mark_failed(proxy_entry)

        elapsed = (time.monotonic() - started) * 1000
        return SearxngResponse(
            results=[],
            was_captcha=True,
            captcha_category=last_detection.category if last_detection else "unknown_block",
            attempts=attempts,
            elapsed_ms=elapsed,
        )

    def _pick_proxy(self):
        if self.proxy_pool is None:
            return None
        try:
            return self.proxy_pool.next()
        except RuntimeError:
            return None

    async def _get(self, query: str, limit: int, proxy_entry):
        params = {"q": query, "format": "json", "safesearch": "0"}
        if self.http_client is not None:
            # Tests inject an httpx.AsyncClient with a MockTransport.
            resp = await self.http_client.get(
                f"{self.base_url}/search", params=params, timeout=self.timeout,
            )
        else:
            import httpx
            client_kwargs: dict[str, Any] = {"timeout": self.timeout}
            if proxy_entry is not None:
                client_kwargs["proxy"] = proxy_entry.url
            async with httpx.AsyncClient(**client_kwargs) as c:
                resp = await c.get(f"{self.base_url}/search", params=params)
        return resp.status_code, resp.text, dict(resp.headers)
