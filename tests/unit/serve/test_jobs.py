import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from nuggetindex.serve import create_app
from nuggetindex.serve.jobs import JobRegistry


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "store.db")
    with TestClient(app) as c:
        yield c


def test_job_submit_returns_job_id(client):
    r = client.post(
        "/v1/auto",
        json={
            "corpus_type": "jsonl",
            "corpus_url": None,
            "corpus_name": None,
            "budget": 10,
            "sample_size": 20,
            "mode": "offline-curated",
            "extractor": "trigger",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]
    assert body["state"] in {"pending", "running", "succeeded"}


def test_job_status_returns_404_for_unknown(client):
    r = client.get("/v1/auto/does-not-exist")
    assert r.status_code == 404


def test_job_eventually_succeeds(client):
    r = client.post(
        "/v1/auto",
        json={
            "corpus_type": "jsonl",
            "budget": 5,
            "sample_size": 10,
            "mode": "offline-curated",
            "extractor": "trigger",
        },
    )
    job_id = r.json()["job_id"]
    # Poll until terminal state or timeout
    import time

    for _ in range(50):
        time.sleep(0.1)
        status = client.get(f"/v1/auto/{job_id}").json()
        if status["state"] in {"succeeded", "failed"}:
            break
    assert status["state"] == "succeeded"
    assert status["progress"] == 1.0
    assert status["result"] is not None


@pytest.mark.asyncio
async def test_job_registry_bounded_concurrency():
    registry = JobRegistry(max_concurrency=2)
    observed_concurrency: list[int] = []
    active = {"n": 0}

    async def slow(params, job):
        active["n"] += 1
        observed_concurrency.append(active["n"])
        await asyncio.sleep(0.05)
        active["n"] -= 1
        return {"ok": True}

    jobs = [await registry.submit(slow, {}) for _ in range(5)]
    # Wait for all to finish.
    while any(j.state not in {"succeeded", "failed"} for j in jobs):
        await asyncio.sleep(0.02)
    assert max(observed_concurrency) <= 2


@pytest.mark.asyncio
async def test_failed_job_records_error():
    registry = JobRegistry(max_concurrency=1)

    async def blow_up(params, job):
        raise RuntimeError("kaboom")

    job = await registry.submit(blow_up, {})
    # Wait until terminal.
    for _ in range(50):
        if job.state in {"succeeded", "failed"}:
            break
        await asyncio.sleep(0.02)
    assert job.state == "failed"
    assert job.error is not None
    assert "kaboom" in job.error
