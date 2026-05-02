import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from nuggetindex.serve import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "store.db")
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_stats_reports_empty_store(client):
    r = client.get("/v1/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_nuggets"] == 0
    assert body["contested_nuggets"] == 0
    assert body["rename_edges"] == 0


def test_ingest_then_stats(client):
    r = client.post(
        "/v1/ingest",
        json={
            "source_id": "d1",
            "text": "Tim Cook became CEO of Apple in 2011.",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["document_id"]
    # nuggets_added is extractor-dependent; just assert integer shape.
    assert isinstance(body["nuggets_added"], int)
    assert isinstance(body["conflicts_detected"], int)

    r = client.get("/v1/stats")
    assert r.status_code == 200
    stats = r.json()
    # Non-negative integer totals after ingest.
    assert stats["total_nuggets"] >= 0


def test_query_returns_shape(client):
    # Seed one known fact so the sidecar's offline-curated path has something.
    client.post(
        "/v1/ingest",
        json={
            "source_id": "d1",
            "text": "Tim Cook became CEO of Apple in 2011.",
        },
    )
    r = client.post("/v1/query", json={"query": "who runs Apple?"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "who runs Apple?"
    assert "context_block" in body
    assert isinstance(body["nuggets"], list)
    assert "use_nugget" in body
    assert "reason" in body


def test_doctor_runs_fast_mode_on_empty_store(client):
    r = client.post("/v1/doctor", json={"sample_size": 10, "mode": "fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["sample_mode"] == "fast"
    assert isinstance(body["scores"], list)
    assert isinstance(body["verdict"], str)
    assert "rendered_markdown" in body


def test_query_validation_rejects_invalid_top_k(client):
    r = client.post("/v1/query", json={"query": "x", "top_k": 500})
    assert r.status_code == 422


def test_ingest_rejects_missing_fields(client):
    r = client.post("/v1/ingest", json={"text": "orphan"})
    assert r.status_code == 422
