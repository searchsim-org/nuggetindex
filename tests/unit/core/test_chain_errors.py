"""Tests for the chain-specific error hierarchy."""

import pytest

from nuggetindex.core.errors import (
    ChainAmbiguousError,
    ChainCycleDetected,
    ChainDepthExceeded,
    NuggetIndexError,
)


def test_chain_ambiguous_inherits_from_root() -> None:
    assert issubclass(ChainAmbiguousError, NuggetIndexError)


def test_chain_cycle_inherits_from_root() -> None:
    assert issubclass(ChainCycleDetected, NuggetIndexError)


def test_chain_depth_inherits_from_root() -> None:
    assert issubclass(ChainDepthExceeded, NuggetIndexError)


def test_chain_ambiguous_carries_attrs() -> None:
    err = ChainAmbiguousError(subject="X", candidates=[1, 2], step=3)
    assert err.subject == "X"
    assert err.candidates == [1, 2]
    assert err.step == 3


def test_chain_ambiguous_message_mentions_resolver() -> None:
    err = ChainAmbiguousError(subject="Acme", candidates=[1, 2, 3], step=0)
    assert "Acme" in str(err)
    assert "resolver" in str(err).lower()


def test_errors_reexported_from_top_level() -> None:
    from nuggetindex import (
        ChainAmbiguousError as CA,
    )
    from nuggetindex import (
        ChainCycleDetected as CC,
    )
    from nuggetindex import (
        ChainDepthExceeded as CD,
    )

    assert CA is ChainAmbiguousError
    assert CC is ChainCycleDetected
    assert CD is ChainDepthExceeded


def test_chain_ambiguous_is_raisable() -> None:
    with pytest.raises(ChainAmbiguousError) as ei:
        raise ChainAmbiguousError(subject="Y", candidates=[], step=-1)
    assert ei.value.subject == "Y"
