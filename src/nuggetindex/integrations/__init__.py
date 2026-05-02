"""Namespace package for third-party framework integrations.

Intentionally empty so ``import nuggetindex.integrations`` never fails when
optional extras like ``[langchain]`` / ``[llamaindex]`` / ``[haystack]``
aren't installed. Re-exports live one level deeper in each framework's own
``__init__``; importing those is what triggers the import-guard error that
points users at the right ``pip install nuggetindex[...]`` extra.
"""
