"""Store module: NuggetStore public class + pluggable backends."""
from nuggetindex.store.base import (
    AddResult,
    DiffReport,
    IngestResult,
    MetadataBackend,
    NuggetStore,
    ViewMode,
)

__all__ = [
    "AddResult",
    "DiffReport",
    "IngestResult",
    "MetadataBackend",
    "NuggetStore",
    "ViewMode",
]
