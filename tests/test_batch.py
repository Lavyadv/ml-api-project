"""Batch prediction, including the configured size limit."""
import pytest


def _row(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2):
    return {
        "sepal_length": sepal_length, "sepal_width": sepal_width,
        "petal_length": petal_length, "petal_width": petal_width,
    }


@pytest.mark.parametrize("size", [1, 2, 50, 100])
def test_batch_handles_valid_sizes(client, size):
    """Task requirement: 1-100 inputs must all work."""
    response = client.post("/api/v1/predict-batch", json={"items": [_row()] * size})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == size
    assert len(body["predictions"]) == size


def test_batch_preserves_input_order(client):
    """Result i must correspond to input i, or clients can't match them up."""
    items = [
        _row(5.1, 3.5, 1.4, 0.2),   # setosa
        _row(7.7, 3.0, 6.1, 2.3),   # virginica
        _row(5.1, 3.5, 1.4, 0.2),   # setosa again
    ]
    predictions = client.post("/api/v1/predict-batch", json={"items": items}).json()["predictions"]

    assert predictions[0]["prediction"] == "setosa"
    assert predictions[1]["prediction"] == "virginica"
    assert predictions[2]["prediction"] == "setosa"


def test_batch_agrees_with_single_predict(client, valid_payload):
    """Batching must not change the answer — only how fast it arrives."""
    single = client.post("/api/v1/predict", json=valid_payload).json()
    batched = client.post("/api/v1/predict-batch", json={"items": [valid_payload]}).json()

    assert batched["predictions"][0]["prediction"] == single["prediction"]
    assert batched["predictions"][0]["confidence"] == single["confidence"]


def test_batch_over_the_limit_is_rejected(client, max_batch_size):
    """413, and the message should say what the limit actually is."""
    oversized = [_row()] * (max_batch_size + 1)
    response = client.post("/api/v1/predict-batch", json={"items": oversized})

    assert response.status_code == 413
    assert str(max_batch_size) in response.json()["detail"]


def test_empty_batch_is_rejected(client):
    response = client.post("/api/v1/predict-batch", json={"items": []})
    assert response.status_code == 422


def test_batch_rejects_a_bad_row(client, valid_payload):
    """One malformed row invalidates the request — no partial scoring."""
    response = client.post("/api/v1/predict-batch", json={
        "items": [valid_payload, {"sepal_length": -1}],
    })

    assert response.status_code == 422
    locations = [".".join(str(p) for p in err["loc"]) for err in response.json()["detail"]]
    assert any("items.1" in loc for loc in locations), "the error should point at row 1"
