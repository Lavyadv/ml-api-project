"""Loading and running the trained Iris model.

Kept separate from main.py so the web layer (routes, HTTP, JSON) and the
ML layer (joblib files, numpy arrays) don't tangle together. main.py
never needs to know what shape scikit-learn wants; this module never
needs to know what a request is.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.logging_config import kv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Resolved from this file's location, not the current working directory,
# so the app finds its model no matter where uvicorn was started from.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "saved_model" / "model.joblib"


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted but no model is available."""


class InferenceError(RuntimeError):
    """Raised when the model fails on input that passed validation.

    Typically a shape mismatch: the request was well-formed JSON with the
    right types, but the array handed to scikit-learn didn't have the
    number of columns the pipeline was fitted on.
    """


@dataclass(frozen=True)
class PredictionResult:
    """What one prediction produced, in plain Python types."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class IrisModel:
    """Owns the loaded pipeline and everything saved alongside it.

    Wrapping the model in an object (rather than assigning a bare module
    -level `model = None` global) means the "is it loaded?" question has
    an honest answer at any moment, and the state that must stay in sync
    — pipeline, class names, feature order, version — travels together
    instead of as four globals that can drift apart.
    """

    def __init__(self, model_path: str | os.PathLike[str] | None = None) -> None:
        self.model_path = Path(model_path or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH)
        self._pipeline = None
        self.target_names: list[str] = []
        self.features: list[str] = []
        self.version: str | None = None

    @property
    def is_loaded(self) -> bool:
        """True once load() has succeeded. Never raises, so /health can ask freely."""
        return self._pipeline is not None

    def load(self) -> None:
        """Read the model bundle from disk. Called once, at app startup."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No model file at {self.model_path}. Run `python ml/train.py` to create it."
            )

        bundle = joblib.load(self.model_path)

        # train.py saves a dict so the class names and feature order travel
        # with the weights. Without them the API would be guessing at both.
        missing = [key for key in ("pipeline", "target_names", "features") if key not in bundle]
        if missing:
            raise ValueError(f"Model bundle at {self.model_path} is missing keys: {missing}")

        self._pipeline = bundle["pipeline"]
        self.target_names = list(bundle["target_names"])
        self.features = list(bundle["features"])
        self.version = bundle.get("version") or self._fingerprint()

        logger.info(
            kv(
                event="model_loaded",
                path=str(self.model_path),
                version=self.version,
                features=",".join(self.features),
                classes=",".join(self.target_names),
                supports_proba=hasattr(self._pipeline, "predict_proba"),
            )
        )

    def unload(self) -> None:
        """Drop the reference at shutdown so the memory can be reclaimed."""
        self._pipeline = None

    def _fingerprint(self) -> str:
        """Identify the exact artifact serving predictions.

        The bundle carries no explicit version, so hash the file. Two
        deployments returning different answers can then be told apart
        from the response body alone.
        """
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        return f"sha256:{digest[:12]}"

    def _to_array(self, payload: dict[str, float]) -> np.ndarray:
        """Turn a dict of named features into the 2D array sklearn expects.

        The column order comes from `self.features` — the order saved by
        train.py — NOT from the order the keys happen to appear in the
        request. Reordering the JSON body therefore cannot silently feed
        petal width into the sepal length column.

        A feature the model wants but the payload lacks is a loud error
        rather than a zero-filled column, because a wrong prediction is
        worse than no prediction.
        """
        try:
            row = [float(payload[name]) for name in self.features]
        except KeyError as exc:
            missing_name = exc.args[0]
            raise InferenceError(
                f"Input is missing the feature {missing_name!r}; "
                f"this model expects exactly {self.features}"
            ) from exc

        # shape (1, n_features): one row, because we predict for one flower.
        return np.asarray([row], dtype=float)

    def predict(self, payload: dict[str, float]) -> PredictionResult:
        """Run one prediction. Raises ModelNotLoadedError / InferenceError."""
        if self._pipeline is None:
            raise ModelNotLoadedError("Model is not loaded")

        features = self._to_array(payload)

        try:
            class_index = int(self._pipeline.predict(features)[0])
            probabilities = self._pipeline.predict_proba(features)[0]
        except ValueError as exc:
            # sklearn raises ValueError for a wrong number of columns.
            raise InferenceError(f"Model rejected the input array: {exc}") from exc

        return PredictionResult(
            label=self.target_names[class_index],
            confidence=float(probabilities[class_index]),
            probabilities={
                name: round(float(p), 6) for name, p in zip(self.target_names, probabilities)
            },
        )
