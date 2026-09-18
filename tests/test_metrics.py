"""Prometheus endpoint and ML-specific metric regression coverage."""
from prometheus_client import REGISTRY


def _row() -> dict[str, float]:
    return {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def test_metrics_exposes_http_and_successful_prediction_metrics(client):
    """A scrape sees both instrumentator and application-level measurements."""
    before = REGISTRY.get_sample_value(
        "iris_predictions_total", {"api_version": "v1", "predicted_class": "setosa"}
    ) or 0.0

    assert client.post("/api/v1/predict", json=_row()).status_code == 200
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_request_duration_highr_seconds" in response.text
    assert "iris_predictions_total" in response.text
    assert REGISTRY.get_sample_value(
        "iris_predictions_total", {"api_version": "v1", "predicted_class": "setosa"}
    ) == before + 1
