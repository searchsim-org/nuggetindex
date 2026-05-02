"""Stable short-hash helper used for content-addressed nugget IDs."""

import hashlib


def stable_short_hash(content: str) -> str:
    """Return the first 16 hex chars of a SHA-256 over UTF-8-encoded content.

    Used for nugget IDs, session-cache keys, and extraction cache keys.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
