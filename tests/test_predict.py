"""Single prediction: the happy path and every way it can be sent wrong."""
import pytest

VALID_SPECIES = {"setosa", "versicolor", "virginica"}


def test_predict_valid_input_returns_sensible_prediction(client, valid_payload):
    response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in VALID_SPECIES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["request_id"]
    assert body["model_version"]


def test_predict_known_setosa_is_classified_correctly(client, valid_payload):
    """A textbook setosa should come back as setosa, confidently.

    This is the test that would catch the features being fed to the model
    in the wrong order — the shapes would all still be valid, but the
    answer would be wrong.
    """
    body = client.post("/api/v1/predict", json=valid_payload).json()

    assert body["prediction"] == "setosa"
    assert body["confidence"] > 0.8


def test_predict_probabilities_cover_every_class_and_sum_to_one(client, valid_payload):
    body = client.post("/api/v1/predict", json=valid_payload).json()

    assert set(body["probabilities"]) == VALID_SPECIES
    assert body["probabilities"][body["prediction"]] == pytest.approx(body["confidence"])
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_ignores_json_key_order(client, valid_payload):
    """Feature order must come from the model, not from the request body."""
    reversed_payload = dict(reversed(list(valid_payload.items())))

    first = client.post("/api/v1/predict", json=valid_payload).json()
    second = client.post("/api/v1/predict", json=reversed_payload).json()

    assert first["prediction"] == second["prediction"]
    assert first["confidence"] == second["confidence"]


def test_different_inputs_give_different_predictions(client):
    """Proves the model is actually being consulted, not a constant returned."""
    setosa = client.post("/api/v1/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2,
    }).json()
    virginica = client.post("/api/v1/predict", json={
        "sepal_length": 7.7, "sepal_width": 3.0, "petal_length": 6.1, "petal_width": 2.3,
    }).json()

    assert setosa["prediction"] != virginica["prediction"]


# --- validation failures: every one of these must be 422, never a 500 ------
@pytest.mark.parametrize(
    "bad_payload, reason",
    [
        ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}, "missing field"),
        ({"sepal_length": "long", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "wrong type"),
        ({"sepal_length": -5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "negative value"),
        ({"sepal_length": 0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "zero value"),
        ({"sepal_length": 510, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "above max"),
        ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "colour": "blue"}, "unknown field"),
        ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 0.2, "petal_width": 1.4}, "petal values swapped"),
        ({}, "empty body"),
    ],
)
def test_predict_rejects_bad_input_with_422(client, bad_payload, reason):
    response = client.post("/api/v1/predict", json=bad_payload)

    assert response.status_code == 422, f"{reason} should be rejected, got {response.status_code}"
    body = response.json()
    assert body["detail"], "a 422 must explain what was wrong"
    assert body["request_id"]


def test_validation_error_names_the_offending_field(client):
    """A 422 is only useful if it says which field failed."""
    response = client.post("/api/v1/predict", json={
        "sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2,
    })

    locations = [".".join(str(p) for p in err["loc"]) for err in response.json()["detail"]]
    assert any("sepal_length" in loc for loc in locations)


def test_predict_never_leaks_a_traceback(client):
    """Error bodies must not contain internals, whatever went wrong."""
    response = client.post("/api/v1/predict", json={"sepal_length": "nope"})

    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert "/Applications/" not in response.text


def test_every_error_response_carries_a_request_id(client):
    """The shape promise: detail + request_id, whatever the status code."""
    responses = [
        client.post("/api/v1/predict", json={}),                     # 422 validation
        client.post("/api/v1/predict-batch", json={"items": [{
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2}] * 500}),       # 413 HTTPException
        client.get("/api/v1/no-such-route"),                          # 404 from Starlette
    ]

    for response in responses:
        body = response.json()
        assert "detail" in body, f"{response.status_code} body missing 'detail'"
        assert body.get("request_id"), f"{response.status_code} body missing 'request_id'"
