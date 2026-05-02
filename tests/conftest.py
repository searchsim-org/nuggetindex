import contextlib
import os

import pytest

# Force a known, stable OpenMP runtime before any native extension (torch, faiss,
# chromadb, etc.) is loaded. On macOS, torch + chromadb + faiss each ship their
# own OpenMP/BLAS — when they load in an arbitrary order across tests, the C
# runtime can abort with "Fatal Python error: Aborted". This workaround lets
# duplicates coexist and stops the native abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import faiss up-front (if installed) so that its BLAS is initialised before
# chromadb/torch load theirs. Tests that don't use faiss are unaffected.
with contextlib.suppress(ImportError):
    import faiss  # noqa: F401


@pytest.fixture
def tmp_db_path(tmp_path):
    """Temp SQLite path for store tests."""
    return tmp_path / "test.db"
