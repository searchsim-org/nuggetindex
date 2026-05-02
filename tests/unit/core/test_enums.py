from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind


def test_nugget_kind_values():
    assert NuggetKind.SEMANTIC_FACT == "semantic_fact"
    assert NuggetKind.EPISODIC_EVENT == "episodic_event"
    assert NuggetKind.INSTRUCTION == "instruction"
    assert NuggetKind.USER_PREFERENCE == "user_preference"


def test_lifecycle_status_values():
    assert LifecycleStatus.ACTIVE == "active"
    assert LifecycleStatus.DEPRECATED == "deprecated"
    assert LifecycleStatus.CONTESTED == "contested"


def test_epistemic_rank_values():
    assert EpistemicRank.PREFERRED == "preferred"
    assert EpistemicRank.NORMAL == "normal"
    assert EpistemicRank.DEPRECATED == "deprecated"


def test_enums_are_str_enums():
    """Enums should serialize as plain strings for SQLite/JSON compatibility."""
    assert isinstance(NuggetKind.SEMANTIC_FACT, str)
    assert isinstance(LifecycleStatus.ACTIVE, str)
    assert isinstance(EpistemicRank.NORMAL, str)
