"""Model metadata endpoint."""


def test_model_info_returns_real_metadata(client):
    response = client.get("/api/v1/model-info")

    assert response.status_code == 200
    body = response.json()

    expected_keys = {
        "model_version", "model_type", "trained_at", "features",
        "classes", "sklearn_version", "test_accuracy", "n_training_samples",
    }
    assert expected_keys <= set(body)


def test_model_info_matches_what_the_model_was_trained_on(client):
    """Guards against the metadata drifting into decorative placeholder text."""
    body = client.get("/api/v1/model-info").json()

    assert body["model_type"] == "RandomForestClassifier"
    assert body["features"] == [
        "sepal_length", "sepal_width", "petal_length", "petal_width",
    ]
    assert body["classes"] == ["setosa", "versicolor", "virginica"]
    assert 0.0 <= body["test_accuracy"] <= 1.0
    assert body["n_training_samples"] > 0
