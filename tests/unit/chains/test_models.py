"""Unit tests for chain data models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from nuggetindex.chains.models import ChainEdge, ChainEdgeType, NuggetChain


def test_chain_edge_is_frozen() -> None:
    e = ChainEdge(
        from_idx=0,
        to_idx=1,
        edge_type=ChainEdgeType.SUCCEEDS,
        gap=timedelta(days=30),
    )
    with pytest.raises(ValidationError):
        e.from_idx = 5  # type: ignore[misc]


def test_chain_head_tail(sample_nuggets: list) -> None:
    chain = NuggetChain(
        nuggets=tuple(sample_nuggets),
        edges=tuple(
            ChainEdge(
                from_idx=i,
                to_idx=i + 1,
                edge_type=ChainEdgeType.SUCCEEDS,
                gap=None,
            )
            for i in range(len(sample_nuggets) - 1)
        ),
        chain_type="succession",
    )
    assert chain.head == sample_nuggets[0]
    assert chain.tail == sample_nuggets[-1]


def test_empty_chain_head_raises() -> None:
    chain = NuggetChain(nuggets=(), edges=(), chain_type="succession")
    with pytest.raises(ValueError):
        _ = chain.head


def test_empty_chain_tail_raises() -> None:
    chain = NuggetChain(nuggets=(), edges=(), chain_type="succession")
    with pytest.raises(ValueError):
        _ = chain.tail


def test_window_returns_nugget_valid_at_time(sample_nuggets: list) -> None:
    chain = NuggetChain(
        nuggets=tuple(sample_nuggets),
        edges=(),
        chain_type="succession",
    )
    mid = sample_nuggets[1].validity.start + timedelta(days=1)
    assert chain.window(mid) == sample_nuggets[1]


def test_window_returns_none_when_no_match(sample_nuggets: list) -> None:
    chain = NuggetChain(nuggets=tuple(sample_nuggets), edges=(), chain_type="succession")
    assert chain.window(datetime(1990, 1, 1, tzinfo=UTC)) is None


def test_chain_type_literal_validated() -> None:
    with pytest.raises(ValidationError):
        NuggetChain(nuggets=(), edges=(), chain_type="bogus")  # type: ignore[arg-type]


def test_chain_is_frozen(sample_nuggets: list) -> None:
    chain = NuggetChain(nuggets=tuple(sample_nuggets), edges=(), chain_type="succession")
    with pytest.raises(ValidationError):
        chain.truncated = True  # type: ignore[misc]


def test_chain_truncated_defaults_false() -> None:
    chain = NuggetChain(nuggets=(), edges=(), chain_type="succession")
    assert chain.truncated is False
    assert chain.as_of is None


def test_chain_edge_types_stringify() -> None:
    assert str(ChainEdgeType.SUCCEEDS) == "succeeds"
    assert str(ChainEdgeType.RENAMES_TO) == "renames_to"
    assert str(ChainEdgeType.OBJECT_IS_SUBJECT) == "object_is_subject"
