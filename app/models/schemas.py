"""Pydantic schemas: the contract for what goes in and what comes out.

FastAPI reads these classes to do three jobs at once:
  * validate and parse incoming JSON before the endpoint function runs,
  * shape and filter the outgoing JSON,
  * generate the /docs page.

A request that doesn't satisfy PredictionInput never reaches the model —
FastAPI rejects it with a 422 and a message naming the offending field.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Every Iris measurement is a length in centimetres, so the same two
# bounds apply to all four. gt=0 rejects zero and negatives (a flower
# part with no length is not a measurement, it's a mistake); le=30 is a
# generous sanity ceiling — the largest petal in the training data is
# 6.9 cm, so 30 catches unit mix-ups (millimetres, inches) and typos
# like 55.0 without rejecting an unusually large but plausible flower.
MIN_CM = 0
MAX_CM = 30


# ==========================================================================
# V1 SCHEMAS — frozen contract
#
# Nothing in this section may change in a breaking way. Renaming a field,
# removing one, or tightening a constraint here would break every client
# already calling /api/v1. Additive, optional fields are the only safe
# edit. Breaking changes go in the V2 section at the bottom instead.
# ==========================================================================


class PredictionInput(BaseModel):
    """The four measurements the model was trained on."""

    # extra="forbid" is a deliberate choice: an unknown field means the
    # caller thinks it is sending something we will use. Silently
    # ignoring `sepal_lenght` (typo) would produce a confident,
    # wrong-for-the-wrong-reason answer. Better to say so, with a 422.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        },
    )

    sepal_length: float = Field(
        ..., gt=MIN_CM, le=MAX_CM, description="Sepal length in cm (must be positive)"
    )
    sepal_width: float = Field(
        ..., gt=MIN_CM, le=MAX_CM, description="Sepal width in cm (must be positive)"
    )
    petal_length: float = Field(
        ..., gt=MIN_CM, le=MAX_CM, description="Petal length in cm (must be positive)"
    )
    petal_width: float = Field(
        ..., gt=MIN_CM, le=MAX_CM, description="Petal width in cm (must be positive)"
    )

    @model_validator(mode="after")
    def petal_must_be_longer_than_wide(self) -> "PredictionInput":
        """Domain rule that no single-field constraint can express.

        Iris petals are always longer than they are wide — in all 150
        training rows the ratio is at least 2.1. A body where petal_width
        exceeds petal_length is almost certainly two values swapped, which
        the model would happily accept and answer confidently. Field-level
        constraints can't catch this because it depends on two fields at
        once, which is exactly what a model_validator is for.
        """
        if self.petal_width > self.petal_length:
            raise ValueError(
                "petal_width cannot exceed petal_length "
                f"(got petal_width={self.petal_width}, petal_length={self.petal_length}) "
                "— these two values may have been swapped"
            )
        return self


class PredictionOutput(BaseModel):
    """The shape every successful /predict response takes."""

    prediction: str = Field(..., description="Predicted species name")
    confidence: float = Field(
        ..., ge=0, le=1, description="Model probability for the predicted class"
    )
    probabilities: dict[str, float] = Field(
        ..., description="Probability assigned to every class"
    )
    model_version: str = Field(..., description="Identifies the exact model artifact used")
    request_id: str = Field(..., description="Correlates this response with the server logs")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prediction": "setosa",
                    "confidence": 1.0,
                    "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
                    "model_version": "sha256:1a2b3c4d5e6f",
                    "request_id": "0f8fad5b-d9cb-469f-a165-70867728950e",
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    """Answer to "is this service actually able to serve predictions?"."""

    status: str = Field(..., description="'ok' when the service can predict, else 'degraded'")
    model_loaded: bool = Field(..., description="Whether a model is loaded in memory")
    model_version: str | None = Field(None, description="Version of the loaded model, if any")


class ErrorResponse(BaseModel):
    """The shape every error response takes, so clients can rely on it."""

    detail: object = Field(..., description="Safe, human-readable description of what went wrong")
    request_id: str = Field(..., description="Quote this when reporting a problem")


class PredictionBatchInput(BaseModel):
    """Many rows in one request.

    The upper bound is NOT declared here as `max_length`, because the
    limit is configurable (MAX_BATCH_SIZE) and a schema constant would
    freeze it at import time. The endpoint enforces it from settings and
    returns 413 instead — see app/routers/v1.py.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "items": [
                        {
                            "sepal_length": 5.1,
                            "sepal_width": 3.5,
                            "petal_length": 1.4,
                            "petal_width": 0.2,
                        },
                        {
                            "sepal_length": 6.0,
                            "sepal_width": 2.7,
                            "petal_length": 5.1,
                            "petal_width": 1.6,
                        },
                    ]
                }
            ]
        },
    )

    items: list[PredictionInput] = Field(
        ..., min_length=1, description="One or more sets of measurements to score"
    )


class PredictionBatchOutput(BaseModel):
    """Results in the same order as the inputs that produced them."""

    predictions: list[PredictionOutput] = Field(
        ..., description="One result per input item, in the order submitted"
    )
    count: int = Field(..., description="How many rows were scored")
    request_id: str = Field(..., description="Correlates this response with the server logs")


class ModelInfoResponse(BaseModel):
    """What model is currently serving traffic.

    Every value here is read from the bundle that train.py saved, not
    written by hand — a hardcoded 'version: 1.0' would go stale the first
    time anyone retrained and would then actively mislead.
    """

    model_version: str = Field(..., description="Fingerprint of the loaded artifact")
    model_type: str = Field(..., description="Estimator class name, e.g. RandomForestClassifier")
    trained_at: str | None = Field(None, description="UTC timestamp of training")
    features: list[str] = Field(..., description="Feature names, in the order the model expects")
    classes: list[str] = Field(..., description="Class labels the model can output")
    sklearn_version: str | None = Field(None, description="scikit-learn version used to train")
    test_accuracy: float | None = Field(None, description="Hold-out accuracy at training time")
    n_training_samples: int | None = Field(None, description="Rows used to fit the model")


# ==========================================================================
# V2 SCHEMAS — the breaking change
#
# What makes v2 necessary rather than an edit to v1:
#   * `confidence` is RENAMED to `probability`      -> removal for v1 clients
#   * `probabilities` (dict) becomes `ranked` (list) -> shape change
#   * `trained_at` is ADDED                          -> additive, safe alone
#
# Only the first two force a new version. An additive optional field could
# have shipped in v1 without breaking anyone; a rename cannot, because a
# client doing response["confidence"] raises KeyError the moment it lands.
#
# What is deliberately NOT duplicated: PredictionInput is shared, because
# the *request* contract did not change, and the inference itself lives in
# IrisModel. Only the response assembly differs between versions.
# ==========================================================================


class ClassProbability(BaseModel):
    """One class and the probability assigned to it."""

    species: str = Field(..., description="Class label")
    probability: float = Field(..., ge=0, le=1, description="Probability for this class")


class PredictionOutputV2(BaseModel):
    """v2 response. Compare field-by-field with PredictionOutput above."""

    prediction: str = Field(..., description="Predicted species name")
    probability: float = Field(
        ..., ge=0, le=1, description="Probability of the predicted class (v1 called this 'confidence')"
    )
    ranked: list[ClassProbability] = Field(
        ..., description="All classes, most likely first (v1 sent an unordered dict)"
    )
    model_version: str = Field(..., description="Identifies the exact model artifact used")
    trained_at: str | None = Field(None, description="When the serving model was trained (new in v2)")
    request_id: str = Field(..., description="Correlates this response with the server logs")
