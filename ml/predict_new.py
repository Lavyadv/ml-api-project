"""Validated prediction entry point for one new Iris flower.

The saved sklearn Pipeline owns scaling and classification. This module owns
the boundary check: reject malformed raw data before it reaches the model.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:  # Supports both `python ml/predict_new.py` and `python -m ml.predict_new`.
    from ml.train import FEATURE_COLUMNS
except ModuleNotFoundError:  # pragma: no cover - exercised by direct-script use
    from train import FEATURE_COLUMNS

DEFAULT_PIPELINE_PATH = Path("ml/saved_model/iris_pipeline.pkl")
MIN_CM = 0.0
MAX_CM = 30.0


def _one_row(raw_customer: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a dict or exactly-one-row DataFrame."""
    if isinstance(raw_customer, pd.DataFrame):
        if len(raw_customer) != 1:
            raise ValueError("Expected exactly one customer row")
        row = raw_customer.copy()
    elif isinstance(raw_customer, Mapping):
        row = pd.DataFrame([dict(raw_customer)])
    else:
        raise TypeError("Expected a dictionary or a single-row pandas DataFrame")

    missing = [name for name in FEATURE_COLUMNS if name not in row.columns]
    unexpected = [name for name in row.columns if name not in FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected fields: {unexpected}")

    try:
        values = row[FEATURE_COLUMNS].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All measurements must be numeric") from exc

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Measurements must be finite numbers")
    invalid = [
        name for name, value in values.iloc[0].items() if not MIN_CM < value <= MAX_CM
    ]
    if invalid:
        raise ValueError(f"Measurements must be greater than 0 and at most {MAX_CM}: {invalid}")
    if values.iloc[0]["petal_width"] > values.iloc[0]["petal_length"]:
        raise ValueError("petal_width cannot exceed petal_length")
    return values


def predict_new_customer(
    raw_customer: Mapping[str, Any] | pd.DataFrame,
    model_path: str | Path = DEFAULT_PIPELINE_PATH,
) -> dict[str, float | str]:
    """Return the predicted Iris species and its probability for validated raw input."""
    row = _one_row(raw_customer)
    try:
        bundle = joblib.load(model_path)
        pipeline = bundle["pipeline"]
        target_names = bundle["target_names"]
        # train.py fitted on a NumPy matrix, so preserve that input shape and
        # avoid pretending the artifact was trained with DataFrame names.
        probabilities = pipeline.predict_proba(row.to_numpy())[0]
    except FileNotFoundError as exc:
        raise RuntimeError(f"Saved pipeline not found at {model_path}. Run python ml/train.py first.") from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("Saved pipeline artifact is invalid or cannot score this input") from exc

    index = int(np.argmax(probabilities))
    return {"prediction": str(target_names[index]), "probability": float(probabilities[index])}


if __name__ == "__main__":
    sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    print(predict_new_customer(sample))
