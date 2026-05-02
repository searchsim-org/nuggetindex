"""Tests for the per-thread SQLite ``_ConnectionPool``.

The pool is an internal helper used by ``SQLiteBackend`` to give each
executor thread its own ``sqlite3.Connection``. SQLite in WAL mode allows
concurrent readers as long as each reader uses a different connection, so a
per-thread pool is essential for exploiting executor parallelism without
tripping ``check_same_thread`` or serialising reads behind a lock.
"""

from __future__ import annotations

import threading

from nuggetindex.store.backends.sqlite import _ConnectionPool


def test_pool_returns_same_connection_for_same_thread(tmp_path):
    pool = _ConnectionPool(tmp_path / "p.db")
    c1 = pool.get()
    c2 = pool.get()
    assert c1 is c2
    pool.close_all()


def test_pool_returns_different_connection_for_different_thread(tmp_path):
    pool = _ConnectionPool(tmp_path / "p.db")
    main = pool.get()
    other: list = []
    t = threading.Thread(target=lambda: other.append(pool.get()))
    t.start()
    t.join()
    assert main is not other[0]
    pool.close_all()


def test_pool_wal_and_foreign_keys_enabled(tmp_path):
    pool = _ConnectionPool(tmp_path / "p.db")
    conn = pool.get()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    pool.close_all()


def test_pool_close_all_is_idempotent(tmp_path):
    pool = _ConnectionPool(tmp_path / "p.db")
    pool.get()
    pool.close_all()
    # Second close should not raise even though connections are already closed
    # and the internal list is empty.
    pool.close_all()


def test_pool_row_factory_is_sqlite_row(tmp_path):
    """Callers rely on column access by name (e.g. r["id"])."""
    import sqlite3

    pool = _ConnectionPool(tmp_path / "p.db")
    conn = pool.get()
    assert conn.row_factory is sqlite3.Row
    pool.close_all()
