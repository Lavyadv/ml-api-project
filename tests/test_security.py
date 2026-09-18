"""Authentication and strict-input regression tests."""


def test_predict_rejects_missing_api_key(client, valid_payload):
    response = client.post("/api/v1/predict", json=valid_payload, headers={"X-API-Key": ""})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_predict_rejects_invalid_api_key(client, valid_payload):
    response = client.post("/api/v1/predict", json=valid_payload, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_predict_rejects_unexpected_extra_field(client, valid_payload):
    response = client.post("/api/v1/predict", json={**valid_payload, "unexpected": True})
    assert response.status_code == 422
    assert any(error["type"] == "extra_forbidden" for error in response.json()["detail"])
