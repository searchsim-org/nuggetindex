import nuggetindex


def test_version_is_set():
    assert nuggetindex.__version__.startswith("0.")


def test_package_is_importable():
    # Sanity check: the public namespace exists and can be imported.
    assert hasattr(nuggetindex, "__version__")
