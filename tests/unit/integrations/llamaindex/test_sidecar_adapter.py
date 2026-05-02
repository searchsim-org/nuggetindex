"""Tests for ``nuggetindex.integrations.llamaindex.sidecar``.

The whole module is skipped if LlamaIndex isn't installed. Tests confirm
the adapter:

* Prepends a synthetic ``nuggetindex-governance`` ``TextNode`` when the
  sidecar router opts in (temporal / functional-predicate query).
* Passes the input node list through unchanged on a noise query.

Tests are deliberately sync: the postprocessor's ``_postprocess_nodes``
calls the sync :meth:`Sidecar.handle` wrapper which uses ``asyncio.run``
internally. The store is seeded via a single ``asyncio.run`` helper,
matching the idiom used by the existing Haystack integration test.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import (  # noqa: E402
    NodeWithScore,
    QueryBundle,
    TextNode,
)

from nuggetindex import NuggetStore  # noqa: E402
from nuggetindex.core.enums import (  # noqa: E402
    EpistemicRank,
    LifecycleStatus,
    NuggetKind,
)
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.integrations.llamaindex import (  # noqa: E402
    NuggetSidecarNodePostprocessor,
)
from nuggetindex.sidecar import Sidecar  # noqa: E402


def _seed_store(db_path: Path) -> NuggetStore:
    async def _inner() -> NuggetStore:
        store = NuggetStore(db_path=db_path)
        now = datetime.now(tz=UTC)
        n = Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(
                subject="Google",
                predicate="chiefExecutiveOfficer",
                object="Larry Page",
                text="Larry Page served as Google CEO from 2011 through 2015",
            ),
            validity=ValidityInterval(
                start=datetime(2011, 4, 4, tzinfo=UTC),
                end=datetime(2015, 10, 2, tzinfo=UTC),
            ),
            epistemic=EpistemicState(
                status=LifecycleStatus.ACTIVE,
                rank=EpistemicRank.NORMAL,
                confidence=0.9,
            ),
            provenance=(
                ProvenanceRecord(
                    source_id="wiki",
                    evidence_span="Larry Page served as CEO from 2011 to 2015.",
                    char_start=0,
                    char_end=50,
                    created_at=now,
                ),
            ),
            extraction_confidence=0.9,
        )
        await store.aadd(n)
        return store

    return asyncio.run(_inner())


def _node(node_id: str, text: str, score: float = 0.5) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text), score=score)


def test_llamaindex_sidecar_postprocessor_inserts_governance_node(
    tmp_path: Path,
) -> None:
    store = _seed_store(tmp_path / "s.db")
    sidecar = Sidecar(store=store, mode="offline-curated")
    pp = NuggetSidecarNodePostprocessor(sidecar=sidecar)
    nodes = [_node("n1", "some retrieved text")]
    out = pp._postprocess_nodes(
        nodes, query_bundle=QueryBundle(query_str="who was Google's CEO in 2013?")
    )
    ids = [n.node.node_id for n in out]
    assert ids[0] == "nuggetindex-governance"
    gov = out[0].node
    assert "Larry Page" in gov.get_content()
    assert "n1" in ids


def test_llamaindex_sidecar_postprocessor_passthrough(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "s.db")
    sidecar = Sidecar(store=store, mode="offline-curated")
    pp = NuggetSidecarNodePostprocessor(sidecar=sidecar)
    nodes = [_node("n1", "some retrieved text")]
    out = pp._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="the sky is blue"))
    ids = [n.node.node_id for n in out]
    assert "nuggetindex-governance" not in ids
    assert ids == ["n1"]
