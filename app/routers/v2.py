"""Version 2 of the API — a deliberately breaking response shape.

v1 is NOT edited to produce this. That is the whole point: a client
pinned to /api/v1/predict keeps getting byte-identical responses while
v2 exists alongside it. tests/test_versioning.py proves it with code
rather than asserting it in a comment.

What changed, and why each change needs a version bump:

    v1                              v2
    ------------------------------  ---------------------------------
    confidence: float               probability: float      RENAMED
    probabilities: {str: float}     ranked: [{species,...}]  RESHAPED
    (absent)                        trained_at: str|null     ADDED

Only the rename and the reshape are breaking. A client doing
`response["confidence"]` raises KeyError against v2, and one doing
`response["probabilities"]["setosa"]` gets a list it cannot subscript
by name. The added field breaks nobody — a well-behaved client ignores
fields it does not know, which is why `trained_at` alone would not have
justified a new version.

HOW THIS FILE STAYS SMALL — the answer to "where would you duplicate?":
inference is not reimplemented here. IrisModel.predict() is shared, and
so is PredictionInput, because the *request* contract is unchanged. Only
response assembly differs, which is the only thing that actually changed.
If v3 ever needs a different request shape too, that is when a separate
input schema earns its place — not before.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_model, get_request_id, require_api_key
from app.logging_config import kv
from app.metrics import PREDICTIONS_TOTAL
from app.ml_model import InferenceError, IrisModel, ModelNotLoadedError
from app.models.schemas import (
    ClassProbability,
    ErrorResponse,
    PredictionInput,
    PredictionOutputV2,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(require_api_key)])


@router.post(
    "/predict",
    response_model=PredictionOutputV2,
    summary="Predict the Iris species (v2 response shape)",
    responses={
        422: {"model": ErrorResponse, "description": "Input failed validation"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def predict_v2(
    payload: PredictionInput,
    model: IrisModel = Depends(get_model),
    request_id: str = Depends(get_request_id),
) -> PredictionOutputV2:
    """Same model, same input, different response contract from v1."""
    try:
        result = model.predict(payload.model_dump())
    except (ModelNotLoadedError, InferenceError):
        raise
    except Exception as exc:
        logger.exception(kv(event="prediction_failed", error=type(exc).__name__))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed"
        ) from exc

    # Most likely class first — the ordering v1's dict could not express.
    ranked = sorted(
        (ClassProbability(species=name, probability=p) for name, p in result.probabilities.items()),
        key=lambda c: c.probability,
        reverse=True,
    )

    logger.info(
        kv(
            event="prediction_success",
            api_version="v2",
            prediction=result.label,
            probability=round(result.confidence, 4),
            model_version=model.version,
        )
    )
    PREDICTIONS_TOTAL.labels(api_version="v2", predicted_class=result.label).inc()

    return PredictionOutputV2(
        prediction=result.label,
        probability=result.confidence,
        ranked=ranked,
        model_version=model.version or "unknown",
        trained_at=model.info().get("trained_at"),
        request_id=request_id,
    )
