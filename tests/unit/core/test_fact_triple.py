import pytest
from pydantic import ValidationError

from nuggetindex.core.models import FactTriple


def test_fact_triple_requires_all_fields():
    ft = FactTriple(
        subject="Google",
        predicate="chiefExecutiveOfficer",
        object="Sundar Pichai",
        text="Sundar Pichai is CEO of Google",
    )
    assert ft.subject == "Google"


def test_fact_triple_is_frozen():
    ft = FactTriple(subject="A", predicate="b", object="C", text="A b C")
    with pytest.raises(ValidationError):
        ft.subject = "X"


def test_fact_triple_rejects_empty_subject():
    with pytest.raises(ValidationError):
        FactTriple(subject="", predicate="b", object="C", text="t")


def test_fact_triple_json_round_trip():
    ft = FactTriple(subject="A", predicate="b", object="C", text="A b C")
    dumped = ft.model_dump_json()
    restored = FactTriple.model_validate_json(dumped)
    assert restored == ft
