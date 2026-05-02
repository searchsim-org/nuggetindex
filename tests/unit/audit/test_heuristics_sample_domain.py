"""Tests for ``stratify_by='domain'`` in ``stratified_sample`` (Fix 4)."""

from __future__ import annotations

import pytest

from nuggetindex.audit.heuristics import stratified_sample
from nuggetindex.pipeline.constructor import Document


def _host_of(uri: str | None) -> str:
    """Local helper mirroring the sampler's ``_domain_of`` contract."""
    if not uri:
        return "unknown_domain"
    from urllib.parse import urlparse

    return urlparse(uri).netloc or "unknown_domain"


@pytest.mark.asyncio
async def test_domain_stratification_covers_hosts() -> None:
    """10 docs per host across 5 hosts; a 20-doc sample should hit at least 4 hosts."""
    hosts = [
        "a.example.com",
        "b.example.com",
        "c.example.com",
        "d.example.com",
        "e.example.com",
    ]
    docs: list[Document] = []
    for host in hosts:
        for i in range(10):
            docs.append(
                Document(
                    source_id=f"{host}-{i:02d}",
                    text=f"filler text for {host} doc {i}",
                    uri=f"https://{host}/article/{i}",
                    source_date=None,
                )
            )

    sampled, n_total = await stratified_sample(
        docs, sample_size=20, stratify_by="domain", rng_seed=0
    )
    assert n_total == 50
    assert len(sampled) == 20
    hosts_hit = {_host_of(d.uri) for d in sampled}
    assert len(hosts_hit) >= 4, f"expected >=4 hosts, got {sorted(hosts_hit)}"


@pytest.mark.asyncio
async def test_domain_unknown_bucket() -> None:
    """Docs with ``uri=None`` all fall into one synthetic bucket; the sampler still returns all 10."""
    docs = [
        Document(
            source_id=f"u{i:03d}",
            text=f"orphan {i}",
            uri=None,
            source_date=None,
        )
        for i in range(10)
    ]
    sampled, n_total = await stratified_sample(
        docs, sample_size=10, stratify_by="domain", rng_seed=0
    )
    assert n_total == 10
    assert len(sampled) == 10
    assert {d.source_id for d in sampled} == {d.source_id for d in docs}
