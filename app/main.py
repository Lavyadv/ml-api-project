"""FastAPI application: routes, startup, logging middleware, error handling.

Request lifecycle, end to end:

    request
      -> log_requests middleware   (assign request_id, start the clock)
      -> FastAPI + Pydantic        (parse & validate body -> 422 if bad)
      -> endpoint function         (shape array, call model)
      -> response_model            (filter & shape the outgoing JSON)
      -> log_requests middleware   (log method, path, status, duration)
    response

Anything raised along the way lands in one of the exception handlers
below, which turn it into a documented JSON body — never a traceback.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging_config import configure_logging, kv, request_id_var
from app.ml_model import InferenceError, IrisModel, ModelNotLoadedError
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionInput,
    PredictionOutput,
)

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once when the server boots, and once when it shuts down.

    Loading the model here — rather than inside the endpoint — means the
    joblib file is read from disk exactly once for the life of the
    process. Loading it per request would re-read and re-deserialize the
    whole model on every call, turning a millisecond of inference into
    hundreds of milliseconds of disk and CPU.

    Everything before `yield` is startup; everything after is shutdown.
    """
    configure_logging()
    logger.info(kv(event="startup_begin", api_version=API_VERSION))

    model = IrisModel()
    try:
        model.load()
    except Exception as exc:
        # Deliberate choice: start anyway, in a degraded state, rather
        # than refusing to boot. That keeps /health reachable so a
        # monitor can *report* "up but cannot predict" instead of just
        # seeing a dead port. The cost is that the failure is quieter —
        # so it is logged at ERROR, /health answers 503, and /predict
        # answers 503 rather than pretending.
        logger.error(
            kv(event="model_load_failed", path=str(model.model_path), error=str(exc)),
            exc_info=True,
        )

    # app.state is FastAPI's designated place for objects shared across
    # requests. Endpoints reach it via request.app.state, so nothing has
    # to reach for a module-level global that tests can't substitute.
    app.state.model = model

    logger.info(kv(event="startup_complete", model_loaded=model.is_loaded))
    yield
    logger.info(kv(event="shutdown", model_loaded=model.is_loaded))
    model.unload()
    app.state.model = None


app = FastAPI(
    title="Iris Classifier API",
    description="Serves a scikit-learn Iris species classifier over HTTP.",
    version=API_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request, and give each one a traceable id.

    The id is stored in two places on purpose:
      * request.state.request_id — so the endpoint can read it and put it
        in the response body;
      * the request_id ContextVar — so *any* log line emitted while this
        request is being handled is stamped with it automatically.
    """
    # Honour a caller-supplied id so a trace can span several services.
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    token = request_id_var.set(request_id)

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            kv(
                event="request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )
        )
        raise  # let the exception handlers below produce the response
    else:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            kv(
                event="request_complete",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
        )
        # Hand the id back so the caller can quote it in a bug report.
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


# --------------------------------------------------------------------------
# Exception handlers — the only things that decide what an error looks like
# --------------------------------------------------------------------------
def _request_id_of(request: Request) -> str:
    """Read the id without assuming the middleware got to run."""
    return getattr(request.state, "request_id", "-")


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError) -> JSONResponse:
    """Handle a model that failed on structurally valid input.

    The realistic cause is a shape mismatch — the schema and the model
    disagree about the feature set, e.g. someone renamed a field or
    retrained with a different number of columns. The real reason goes
    to the log; the client gets a fixed, safe sentence.
    """
    logger.error(
        kv(event="inference_error", path=request.url.path, error=str(exc)), exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Prediction failed", "request_id": _request_id_of(request)},
    )


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    """503, not 500: the service is fine, it just has nothing to serve with."""
    logger.error(kv(event="model_not_loaded", path=request.url.path))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Model is not loaded; the service cannot serve predictions",
            "request_id": _request_id_of(request),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's own 422, plus the request_id.

    The per-field detail list is kept exactly as FastAPI builds it — that
    is the genuinely useful part for whoever sent the bad request.
    """
    logger.warning(
        kv(event="validation_failed", path=request.url.path, errors=len(exc.errors()))
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": jsonable_encoder(exc.errors()),
            "request_id": _request_id_of(request),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence, so no traceback ever reaches a client.

    Without this, an unexpected error would render a stack trace naming
    local file paths and internals — unprofessional, and a real
    information leak.
    """
    logger.exception(kv(event="unhandled_exception", path=request.url.path))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": _request_id_of(request)},
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is alive", "docs": "/docs", "version": API_VERSION}


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request, response: Response) -> HealthResponse:
    """Cheap liveness/readiness check for monitoring.

    Deliberately does not touch the prediction path — a monitor hitting
    this every few seconds must not burn CPU on inference. It also must
    never crash: `getattr` with a default covers the case where startup
    failed before app.state.model was ever assigned.
    """
    model = getattr(request.app.state, "model", None)
    loaded = model is not None and model.is_loaded

    # A monitor keys off the status code, not the body, so an unusable
    # service must not answer 200.
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=model.version if loaded else None,
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["inference"],
    summary="Predict the Iris species from four measurements",
    responses={
        422: {"model": ErrorResponse, "description": "Input failed validation"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def predict(payload: PredictionInput, request: Request) -> PredictionOutput:
    """Run the loaded model against one set of measurements.

    Note what is *absent*: there is no joblib.load() here. The model was
    loaded once at startup and is only looked up.

    Defined with `def` rather than `async def` on purpose. Inference is
    blocking CPU work; a plain `def` endpoint is run by FastAPI in a
    threadpool, so a slow prediction doesn't stall the event loop and
    freeze every other in-flight request.
    """
    request_id = _request_id_of(request)
    model: IrisModel | None = getattr(request.app.state, "model", None)

    if model is None or not model.is_loaded:
        raise ModelNotLoadedError("Model is not loaded")

    try:
        result = model.predict(payload.model_dump())
    except (ModelNotLoadedError, InferenceError):
        # Both have dedicated handlers above that log and shape the
        # response; re-raise rather than flattening them into a generic 500.
        raise
    except Exception as exc:
        # Anything genuinely unforeseen: record the full detail server-side,
        # send the client a sentence that reveals nothing about internals.
        logger.exception(kv(event="prediction_failed", error=type(exc).__name__))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed"
        ) from exc

    logger.info(
        kv(
            event="prediction_success",
            prediction=result.label,
            confidence=round(result.confidence, 4),
            model_version=model.version,
        )
    )

    return PredictionOutput(
        prediction=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_version=model.version or "unknown",
        request_id=request_id,
    )
