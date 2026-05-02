"""Tests for the public ``NuggetStore.backend`` property and the
deprecation shim on the old ``_backend`` dunder-prefixed access."""
import warnings

import pytest

from nuggetindex import NuggetStore
from nuggetindex.store.base import MetadataBackend  # noqa: F401  (re-exported symbol check)


def test_backend_property_returns_metadata_backend(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    # Public: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend = store.backend
    # Implements the protocol.
    assert hasattr(backend, "aupsert")
    assert hasattr(backend, "afilter")
    assert hasattr(backend, "abm25_search")


def test_private_backend_emits_deprecation_warning(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    with pytest.warns(DeprecationWarning, match="store.backend"):
        _ = store._backend


def test_public_and_private_return_same_object(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert store.backend is store._backend
