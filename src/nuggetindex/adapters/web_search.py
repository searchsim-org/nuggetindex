"""Web-search CorpusSource.

Two backends:

* ``searxng`` -- HTTP against a self-hosted SearXNG instance. Fast, cheap,
  zero browser. Survives rate limits via ProxyPool rotation.
* ``searxng+camoufox`` -- same, but when SearXNG's response is CAPTCHA-flagged
  (HTTP 429, CF challenge, Google sorry, DDG anomaly, Bing captcha, or
  SearXNG silent-empty), the backend escalates to Camoufox (stealth Firefox
  via Playwright) and scrapes the engine directly. Each escalation rotates
  proxies.

No paid APIs. Self-hostable. Covers the three canonical blocks:
  1. HTTP rate limit (429 / 503)
  2. Challenge pages (CF, Google sorry, DDG anomaly, Bing captcha)
  3. Silent empty responses (200 + zero results + engines failed upstream)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nuggetindex.adapters.searxng.camoufox_backend import CamoufoxBackend
from nuggetindex.adapters.searxng.client import SearxngClient
from nuggetindex.adapters.searxng.detect import CaptchaDetector
from nuggetindex.adapters.searxng.proxy import ProxyPool

_Backend = Literal["searxng", "searxng+camoufox"]


@dataclass
class WebSearchCorpus:
    searxng_url: str
    backend: _Backend = "searxng"
    proxy_pool: ProxyPool | None = None
    camoufox_engine: str = "google"  # google | bing | ddg
    camoufox_opts: dict[str, Any] = field(default_factory=dict)
    detector: CaptchaDetector = field(default_factory=CaptchaDetector)
    timeout: float = 20.0
    http_client: Any = None

    _client: SearxngClient = field(init=False)
    _camoufox: CamoufoxBackend | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._client = SearxngClient(
            base_url=self.searxng_url,
            proxy_pool=self.proxy_pool,
            detector=self.detector,
            timeout=self.timeout,
            http_client=self.http_client,
        )
        if self.backend == "searxng+camoufox":
            self._camoufox = CamoufoxBackend(
                engine=self.camoufox_engine,
                proxy_pool=self.proxy_pool,
                detector=self.detector,
                launch_options=self.camoufox_opts,
            )

    async def search(self, query: str, *, limit: int):
        response = await self._client.search(query, limit=limit)
        if not response.was_captcha:
            return [self._to_document(h) for h in response.results[:limit]]
        if self._camoufox is not None:
            hits = await self._camoufox.search(query, limit=limit)
            return [self._to_document(h) for h in hits[:limit]]
        return []

    async def sample(self, *, mode, n):
        from nuggetindex.adapters.base import _TOPIC_DIVERSE_QUERIES
        if mode == "random_ids":
            raise NotImplementedError(
                "WebSearchCorpus does not support mode='random_ids'."
            )
        pool = _TOPIC_DIVERSE_QUERIES if mode == "topic_diverse" else ("news today",)
        per = max(1, n // len(pool)) + 1
        seen: set[str] = set()
        out: list = []
        for q in pool:
            if len(out) >= n:
                break
            for h in await self.search(q, limit=per):
                if h.source_id not in seen:
                    seen.add(h.source_id)
                    out.append(h)
                    if len(out) >= n:
                        break
        return out

    def _to_document(self, hit: dict):
        from nuggetindex.pipeline.constructor import Document
        url = hit.get("url") or ""
        title = hit.get("title") or ""
        content = hit.get("content") or hit.get("snippet") or ""
        text = f"{title}\n{content}".strip() or title or content
        return Document(source_id=url, text=text, uri=url, source_date=None)
