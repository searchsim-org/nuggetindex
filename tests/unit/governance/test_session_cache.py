"""Session cache helpers: default path resolution + content-addressed passage hash."""

from pathlib import Path

from nuggetindex.governance.session_cache import default_cache_path, passage_hash


def test_default_cache_in_home(monkeypatch):
    # Ensure env var is not set so we get the ~/.nuggetindex/sessions/ default.
    monkeypatch.delenv("NUGGETINDEX_CACHE_DIR", raising=False)
    p = default_cache_path(extractor_config="openai:gpt-4o-mini", schema_hash="abc123")
    assert p.parent == Path.home() / ".nuggetindex" / "sessions"
    assert p.name.endswith(".db")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NUGGETINDEX_CACHE_DIR", str(tmp_path))
    p = default_cache_path(extractor_config="x", schema_hash="y")
    assert p.parent == tmp_path


def test_env_override_creates_dir(monkeypatch, tmp_path):
    target = tmp_path / "not_yet_existing"
    monkeypatch.setenv("NUGGETINDEX_CACHE_DIR", str(target))
    p = default_cache_path(extractor_config="x", schema_hash="y")
    assert p.parent == target
    assert target.exists()


def test_same_config_same_hash_same_path(monkeypatch, tmp_path):
    monkeypatch.setenv("NUGGETINDEX_CACHE_DIR", str(tmp_path))
    a = default_cache_path(extractor_config="x", schema_hash="y")
    b = default_cache_path(extractor_config="x", schema_hash="y")
    assert a == b


def test_different_configs_different_hashes(monkeypatch, tmp_path):
    monkeypatch.setenv("NUGGETINDEX_CACHE_DIR", str(tmp_path))
    a = default_cache_path(extractor_config="x", schema_hash="y")
    b = default_cache_path(extractor_config="z", schema_hash="y")
    assert a != b


def test_different_schemas_different_hashes(monkeypatch, tmp_path):
    monkeypatch.setenv("NUGGETINDEX_CACHE_DIR", str(tmp_path))
    a = default_cache_path(extractor_config="x", schema_hash="y")
    b = default_cache_path(extractor_config="x", schema_hash="z")
    assert a != b


def test_passage_hash_deterministic():
    assert passage_hash("hello") == passage_hash("hello")


def test_passage_hash_differs_for_different_inputs():
    assert passage_hash("hello") != passage_hash("goodbye")


def test_passage_hash_is_short_hex():
    h = passage_hash("hello world")
    assert len(h) == 16
    int(h, 16)  # raises if not hex
