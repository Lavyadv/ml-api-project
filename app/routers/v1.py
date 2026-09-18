"""Version 1 of the API — the frozen contract.

Everything here is reachable under /api/v1/... . Once a client depends
on these shapes, they must keep working: fixes and additive optional
fields are fine, renames and removals are not. Those go to v2.py.

Versioning is a promise, not a URL prefix. The prefix is just how the
promise is made visible.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.config import settings
from app.dependencies import get_model, get_request_id, require_api_key
from app.logging_config import kv
from app.metrics import PREDICTIONS_TOTAL
from app.ml_model import InferenceError, IrisModel, ModelNotLoadedError
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionBatchInput,
    PredictionBatchOutput,
    PredictionInput,
    PredictionOutput,
)

logger = logging.getLogger(__name__)

# The prefix lives on the router, not on each route, so every path this
# file defines is versioned automatically — including ones added later.
router = APIRouter(prefix="/api/v1", tags=["v1"], dependencies=[Depends(require_api_key)])

ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Input failed validation"},
    500: {"model": ErrorResponse, "description": "Prediction failed"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request, response: Response) -> HealthResponse:
    """Cheap liveness/readiness check for monitoring.

    Deliberately does not use the get_model dependency: that raises when
    the model is missing, and this endpoint's whole job is to *report*
    that state rather than fail on it.
    """
    model = getattr(request.app.state, "model", None)
    loaded = model is not None and model.is_loaded

    # A monitor keys off the status code, not the body.
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=model.version if loaded else None,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info(model: IrisModel = Depends(get_model)) -> ModelInfoResponse:
    """Report what model is actually serving traffic.

    Every field comes from the bundle train.py saved. When someone asks
    "why did production give a different answer than staging?", this is
    the endpoint that answers it.
    """
    return ModelInfoResponse(**model.info())


@router.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict the Iris species from four measurements",
    responses=ERROR_RESPONSES,
)
def predict(
    payload: PredictionInput,
    model: IrisModel = Depends(get_model),
    request_id: str = Depends(get_request_id),
) -> PredictionOutput:
    """Score one set of measurements.

    Defined with `def` rather than `async def` on purpose: inference is
    blocking CPU work, so FastAPI runs this in a threadpool and the event
    loop stays free for other requests.
    """
    try:
        result = model.predict(payload.model_dump())
    except (ModelNotLoadedError, InferenceError):
        raise  # dedicated handlers in main.py log and shape these
    except Exception as exc:
        logger.exception(kv(event="prediction_failed", error=type(exc).__name__))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed"
        ) from exc

    logger.info(
        kv(
            event="prediction_success",
            api_version="v1",
            prediction=result.label,
            confidence=round(result.confidence, 4),
            model_version=model.version,
        )
    )
    PREDICTIONS_TOTAL.labels(api_version="v1", predicted_class=result.label).inc()

    return PredictionOutput(
        prediction=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_version=model.version or "unknown",
        request_id=request_id,
    )


@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput,
    summary="Score many rows in a single call",
    responses={
        **ERROR_RESPONSES,
        413: {"model": ErrorResponse, "description": "Batch larger than MAX_BATCH_SIZE"},
    },
)
def predict_batch(
    payload: PredictionBatchInput,
    model: IrisModel = Depends(get_model),
    request_id: str = Depends(get_request_id),
) -> PredictionBatchOutput:
    """Score a list of rows in one vectorised call into scikit-learn.

    The size limit comes from settings, not a literal, so it can be
    tuned per environment without a code change. 413 (Payload Too Large)
    rather than 422: the request is perfectly well-formed, there is just
    more of it than this service agreed to accept.
    """
    batch_size = len(payload.items)
    if batch_size > settings.max_batch_size:
        logger.warning(
            kv(event="batch_rejected_too_large", batch_size=batch_size,
               limit=settings.max_batch_size)
        )
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Batch of {batch_size} exceeds the limit of "
                f"{settings.max_batch_size}. Split it into smaller requests."
            ),
        )

    start = time.perf_counter()
    try:
        # One call for the whole batch — see IrisModel.predict_batch.
        results = model.predict_batch([item.model_dump() for item in payload.items])
    except (ModelNotLoadedError, InferenceError):
        raise
    except Exception as exc:
        logger.exception(
            kv(event="batch_prediction_failed", batch_size=batch_size,
               error=type(exc).__name__)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed"
        ) from exc
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        kv(
            event="batch_prediction_success",
            api_version="v1",
            batch_size=batch_size,
            duration_ms=round(duration_ms, 2),
            ms_per_row=round(duration_ms / batch_size, 3),
            model_version=model.version,
        )
    )
    for result in results:
        PREDICTIONS_TOTAL.labels(api_version="v1", predicted_class=result.label).inc()

    return PredictionBatchOutput(
        predictions=[
            PredictionOutput(
                prediction=r.label,
                confidence=r.confidence,
                probabilities=r.probabilities,
                model_version=model.version or "unknown",
                request_id=request_id,
            )
            for r in results
        ],
        count=len(results),
        request_id=request_id,
    )
