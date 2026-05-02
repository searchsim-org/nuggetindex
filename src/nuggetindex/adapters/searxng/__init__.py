"""SearXNG + Camoufox web-search backend primitives.

Stubs for ``ProxyPool``, ``ProxyEntry``. Additional components
(``CaptchaDetector``, ``SearxngClient``, ``CamoufoxBackend``) ship in later
tasks within the same package.
"""

from nuggetindex.adapters.searxng.camoufox_backend import CamoufoxBackend
from nuggetindex.adapters.searxng.client import SearxngClient, SearxngResponse
from nuggetindex.adapters.searxng.detect import CaptchaDetector, DetectionResult
from nuggetindex.adapters.searxng.proxy import ProxyEntry, ProxyPool

__all__ = [
    "CamoufoxBackend",
    "CaptchaDetector",
    "DetectionResult",
    "ProxyEntry",
    "ProxyPool",
    "SearxngClient",
    "SearxngResponse",
]
