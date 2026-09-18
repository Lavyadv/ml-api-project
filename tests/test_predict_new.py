import pytest

from ml.predict_new import predict_new_customer


def test_predict_new_customer_returns_prediction_for_valid_input(valid_payload):
    result = predict_new_customer(valid_payload, "ml/saved_model/model.joblib")
    assert result["prediction"] == "setosa"
    assert 0 <= result["probability"] <= 1


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"sepal_length": 5.1}, "Missing required fields"),
        ({"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "greater than 0"),
        ({"sepal_length": "long", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "numeric"),
    ],
)
def test_predict_new_customer_fails_clearly_for_bad_input(payload, message):
    with pytest.raises(ValueError, match=message):
        predict_new_customer(payload, "ml/saved_model/model.joblib")
