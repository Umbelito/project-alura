from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "customer-segmentation-api"
    assert body["status"] in {"ok", "degraded"}
    assert "scores_available" in body


def test_segment_missing_table_is_503_or_404() -> None:
    response = client.get("/segments/unknown-customer")
    assert response.status_code in {404, 503}
