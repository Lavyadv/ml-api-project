"""Health endpoint: the thing monitoring hits."""


def test_health_returns_200_and_expected_shape(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]  # non-empty string


def test_health_carries_a_request_id_header(client):
    """Every response should be traceable back to a log line."""
    response = client.get("/api/v1/health")
    assert response.headers.get("x-request-id")
