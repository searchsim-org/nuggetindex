"""Smoke tests for the ``reproduce.rqN`` scripts.

Each script should import cleanly and its ``main()`` should return 0 even
when the underlying dataset isn't available — v0.1 ships skeletons that
print "no data wired" placeholder tables.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("rq", ["rq1", "rq2", "rq3", "rq4", "rq5"])
def test_reproduce_script_runs_as_skeleton(rq: str, capsys: pytest.CaptureFixture) -> None:
    mod = importlib.import_module(f"nuggetindex.evaluation.reproduce.{rq}")
    assert hasattr(mod, "main")
    rc = mod.main()
    assert rc == 0
    captured = capsys.readouterr()
    # Each script must print its banner and a Markdown table (``|`` cells).
    assert rq.upper() in captured.out
    assert "|" in captured.out
