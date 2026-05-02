import sqlite3

from nuggetindex.store.backends.sqlite import SQLiteBackend


def test_fresh_db_creates_all_tables(tmp_db_path):
    SQLiteBackend(tmp_db_path)  # constructor runs migrations
    conn = sqlite3.connect(tmp_db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    }
    conn.close()
    assert "nuggets" in tables
    assert "provenance" in tables
    assert "passages" in tables
    assert "nuggets_fts" in tables


def test_wal_mode_enabled(tmp_db_path):
    SQLiteBackend(tmp_db_path)
    conn = sqlite3.connect(tmp_db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"


def test_indexes_created(tmp_db_path):
    SQLiteBackend(tmp_db_path)
    conn = sqlite3.connect(tmp_db_path)
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    conn.close()
    assert "idx_nuggets_key" in indexes
    assert "idx_nuggets_validity" in indexes
    assert "idx_nuggets_status" in indexes
