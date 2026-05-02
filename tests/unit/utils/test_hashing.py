from nuggetindex.utils.hashing import stable_short_hash


def test_deterministic():
    assert stable_short_hash("abc") == stable_short_hash("abc")


def test_different_inputs_different_hashes():
    assert stable_short_hash("abc") != stable_short_hash("abd")


def test_length_is_16():
    assert len(stable_short_hash("abc")) == 16


def test_hex_only():
    h = stable_short_hash("xyz")
    assert all(c in "0123456789abcdef" for c in h)
