"""Proof — in code, not prose — that v2 does not break v1 clients.

Task 14's real question is "would anything you just did break a client
depending on v1's exact response shape?". A comment claiming "no" is
worth nothing. These tests fail loudly the moment it stops being true.
"""

SAME_INPUT = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

# The exact contract v1 promised. Pinned deliberately: if anyone adds,
# removes, or renames a v1 field, this test fails and forces the question
# "should this have been v3 instead?"
V1_FIELDS = {"prediction", "confidence", "probabilities", "model_version", "request_id"}
V2_FIELDS = {"prediction", "probability", "ranked", "model_version", "trained_at", "request_id"}


def test_v1_response_shape_is_unchanged(client):
    """The frozen contract, asserted exactly."""
    body = client.post("/api/v1/predict", json=SAME_INPUT).json()

    assert set(body) == V1_FIELDS
    assert isinstance(body["confidence"], float)
    assert isinstance(body["probabilities"], dict)


def test_v2_response_shape_is_the_new_contract(client):
    body = client.post("/api/v2/predict", json=SAME_INPUT).json()

    assert set(body) == V2_FIELDS
    assert isinstance(body["probability"], float)
    assert isinstance(body["ranked"], list)


def test_both_versions_answer_the_same_input_differently_but_correctly(client):
    """The mini challenge: same input, two valid-but-different shapes."""
    v1 = client.post("/api/v1/predict", json=SAME_INPUT)
    v2 = client.post("/api/v2/predict", json=SAME_INPUT)

    assert v1.status_code == v2.status_code == 200
    v1_body, v2_body = v1.json(), v2.json()

    # Shapes genuinely differ...
    assert set(v1_body) != set(v2_body)

    # ...in exactly the ways documented as breaking:
    assert "confidence" in v1_body and "confidence" not in v2_body   # renamed
    assert "probability" in v2_body and "probability" not in v1_body
    assert isinstance(v1_body["probabilities"], dict)                 # reshaped
    assert isinstance(v2_body["ranked"], list)
    assert "trained_at" in v2_body and "trained_at" not in v1_body    # added

    # ...while the underlying model answer is identical. Versioning
    # changed the presentation, not the prediction.
    assert v1_body["prediction"] == v2_body["prediction"]
    assert v1_body["confidence"] == v2_body["probability"]
    assert v1_body["model_version"] == v2_body["model_version"]


def test_v2_ranked_is_sorted_most_likely_first(client):
    """The ordering guarantee v1's dict could not express."""
    ranked = client.post("/api/v2/predict", json=SAME_INPUT).json()["ranked"]

    probabilities = [entry["probability"] for entry in ranked]
    assert probabilities == sorted(probabilities, reverse=True)
    assert len(ranked) == 3


def test_v2_top_ranked_entry_matches_the_prediction(client):
    body = client.post("/api/v2/predict", json=SAME_INPUT).json()

    assert body["ranked"][0]["species"] == body["prediction"]
    assert body["ranked"][0]["probability"] == body["probability"]


def test_both_versions_share_the_same_input_validation(client):
    """The request contract did not change, so bad input fails identically."""
    bad = {"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}

    assert client.post("/api/v1/predict", json=bad).status_code == 422
    assert client.post("/api/v2/predict", json=bad).status_code == 422


def test_unversioned_root_advertises_both_versions(client):
    body = client.get("/").json()
    assert "/api/v1" in body["api_versions"]
    assert "/api/v2" in body["api_versions"]
