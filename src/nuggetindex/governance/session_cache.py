"""Session-cache helpers for the Tier-1 governance postprocessor.

The session cache is a per-process on-disk NuggetStore that accumulates
nuggets across queries so cross-document conflict detection becomes possible
over the lifetime of a session. Two primitives live here:

* :func:`default_cache_path` — where the cache lives. Respects the
  ``NUGGETINDEX_CACHE_DIR`` env var and falls back to
  ``~/.nuggetindex/sessions/``. The filename is content-hashed over the
  extractor config + schema hash, so different setups don't collide on disk.
* :func:`passage_hash` — the content-addressed hash we use to deduplicate
  extraction work across passages whose text we have already processed.
"""

from __future__ import annotations

import os
from pathlib import Path

from nuggetindex.utils.hashing import stable_short_hash


def default_cache_path(*, extractor_config: str, schema_hash: str) -> Path:
    """Resolve the default session-cache path.

    Precedence: ``NUGGETINDEX_CACHE_DIR`` env var > ``~/.nuggetindex/sessions/``.
    The filename is content-hashed over ``extractor_config`` and
    ``schema_hash`` so different configurations don't collide on disk.
    """
    env = os.environ.get("NUGGETINDEX_CACHE_DIR")
    base = Path(env) if env else Path.home() / ".nuggetindex" / "sessions"
    base.mkdir(parents=True, exist_ok=True)
    h = stable_short_hash(f"{extractor_config}|{schema_hash}")
    return base / f"{h}.db"


def passage_hash(text: str) -> str:
    """Content-addressed hash of passage text for the extraction cache."""
    return stable_short_hash(text)
