"""Haystack integration test conftest.

Currently just a collection-time guard: if ``haystack-ai`` isn't installed,
skip the whole package cleanly. The individual tests set up their own
stores synchronously (see ``test_retriever._seed_store``) because Haystack
components call ``asyncio.run`` inside ``run()`` and so can't live inside
an async fixture — nested loops raise.
"""
from __future__ import annotations

import pytest

pytest.importorskip("haystack")
