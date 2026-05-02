from datetime import UTC

import pytest
from pydantic import ValidationError

from nuggetindex.core.models import ProvenanceRecord


def test_provenance_defaults():
    p = ProvenanceRecord(source_id="doc-1", evidence_span="Pichai is CEO")
    assert p.char_start == 0
    assert p.char_end == 0
    assert p.created_at.tzinfo == UTC


def test_provenance_is_frozen():
    p = ProvenanceRecord(source_id="doc-1", evidence_span="x")
    with pytest.raises(ValidationError):
        p.source_id = "doc-2"
