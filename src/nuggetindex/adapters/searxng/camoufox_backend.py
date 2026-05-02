"""Camoufox stealth-browser fallback.

Launches a patched Firefox (Camoufox) per query with anti-fingerprinting and
geoip-aware locale settings, scrapes Google / Bing / DuckDuckGo result HTML
directly, and parses into the same result shape ``SearxngClient`` returns.

Shares the ``ProxyPool`` with the HTTP client: each query gets a fresh proxy,
and proxies that land on a sorry-page are quarantined like any other failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

from nuggetindex.adapters.searxng.detect import CaptchaDetector
from nuggetindex.adapters.searxng.proxy import ProxyPool

try:
    # Lazy at module level: core package imports without the [web] extra.
    from camoufox.async_api import AsyncCamoufox  # type: ignore
except ImportError:  # pragma: no cover -- exercised via monkeypatch in tests
    AsyncCamoufox = None  # type: ignore


_ENGINE_URLS = {
    "google": "https://www.google.com/search?q={q}&num={n}&hl=en",
    "bing":   "https://www.bing.com/search?q={q}&count={n}",
    "ddg":    "https://duckduckgo.com/html/?q={q}",
}


@dataclass
class CamoufoxBackend:
    engine: str = "google"
    proxy_pool: ProxyPool | None = None
    detector: CaptchaDetector = field(default_factory=CaptchaDetector)
    geoip: bool = True
    max_attempts: int = 3
    launch_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine not in _ENGINE_URLS:
            raise ValueError(
                f"unsupported engine: {self.engine!r}. "
                f"Supported: {sorted(_ENGINE_URLS)}"
            )

    async def search(self, query: str, *, limit: int) -> list[dict]:
        if AsyncCamoufox is None:
            raise RuntimeError(
                "Camoufox is not installed. Install the [web] extra: "
                "pip install 'nuggetindex[web]' && python -m playwright install firefox"
            )
        url = _ENGINE_URLS[self.engine].format(q=quote_plus(query), n=limit)

        attempts_left = max(
            self.max_attempts,
            len(self.proxy_pool) if self.proxy_pool else 1,
        )
        while attempts_left > 0:
            attempts_left -= 1
            proxy_entry = None
            if self.proxy_pool is not None:
                try:
                    proxy_entry = self.proxy_pool.next()
                except RuntimeError:
                    break

            opts: dict[str, Any] = {"geoip": self.geoip, **self.launch_options}
            if proxy_entry is not None:
                opts["proxy"] = {"server": proxy_entry.url}

            try:
                async with AsyncCamoufox(**opts) as browser:
                    page = await browser.new_page()
                    await page.goto(url, wait_until="domcontentloaded")
                    html = await page.content()
                    await page.close()
            except Exception:  # noqa: BLE001 -- browser / proxy flake; rotate
                if proxy_entry is not None and self.proxy_pool is not None:
                    self.proxy_pool.mark_failed(proxy_entry)
                continue

            detection = self.detector.classify(status_code=200, body=html, headers={})
            if detection.is_captcha:
                if proxy_entry is not None and self.proxy_pool is not None:
                    self.proxy_pool.mark_failed(proxy_entry)
                continue

            if proxy_entry is not None and self.proxy_pool is not None:
                self.proxy_pool.mark_success(proxy_entry)
            return self._parse(html, limit)

        return []

    def _parse(self, html: str, limit: int) -> list[dict]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        results: list[dict] = []
        if self.engine == "google":
            for g in soup.select("div.g"):
                a = g.select_one("a[href]")
                h3 = g.select_one("h3")
                snippet = g.select_one("div.VwiC3b, .IsZvec")
                if not (a and h3):
                    continue
                results.append({
                    "url": a.get("href", ""),
                    "title": h3.get_text(" ", strip=True),
                    "content": snippet.get_text(" ", strip=True) if snippet else "",
                })
                if len(results) >= limit:
                    break
        elif self.engine == "bing":
            for li in soup.select("li.b_algo"):
                a = li.select_one("h2 a[href]")
                snippet = li.select_one(".b_caption p")
                if not a:
                    continue
                results.append({
                    "url": a.get("href", ""),
                    "title": a.get_text(" ", strip=True),
                    "content": snippet.get_text(" ", strip=True) if snippet else "",
                })
                if len(results) >= limit:
                    break
        elif self.engine == "ddg":
            for r in soup.select("div.result"):
                a = r.select_one("a.result__a[href]")
                snippet = r.select_one(".result__snippet")
                if not a:
                    continue
                results.append({
                    "url": a.get("href", ""),
                    "title": a.get_text(" ", strip=True),
                    "content": snippet.get_text(" ", strip=True) if snippet else "",
                })
                if len(results) >= limit:
                    break
        return results
