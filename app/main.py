"""FastAPI application wiring.

This file deliberately defines no business routes. Its job is to build
the app and attach the cross-cutting concerns that every version shares:

    * lifespan        — load the model once, at startup
    * middleware      — request ids, timing, access logging
    * exception       — turn failures into documented JSON, never tracebacks
      handlers
    * include_router  — mount each API version

Routes live in app/routers/. Adding /api/v3 later means adding one file
and one include_router() line; nothing in here needs rewriting. That is
what "adding a version won't require rewriting existing code" buys you.

Request lifecycle, end to end:

    request
      -> log_requests middleware   (assign request_id, start the clock)
      -> FastAPI + Pydantic        (parse & validate body -> 422 if bad)
      -> router endpoint           (shape array, call model)
      -> response_model            (filter & shape the outgoing JSON)
      -> log_requests middleware   (log method, path, status, duration)
    response
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.logging_config import configure_logging, kv, request_id_var
from app.ml_model import InferenceError, IrisModel, ModelNotLoadedError
from app.routers import v1, v2

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once when the server boots, and once when it shuts down.

    Loading the model here — rather than inside an endpoint — means the
    joblib file is read from disk exactly once for the life of the
    process. Loading it per request would re-read and re-deserialize the
    whole model on every call, turning a millisecond of inference into
    hundreds of milliseconds of disk and CPU.

    Everything before `yield` is startup; everything after is shutdown.
    """
    configure_logging()
    logger.info(kv(event="startup_begin", api_version=settings.api_version))

    model = IrisModel(settings.model_path)
    try:
        model.load()
    except Exception as exc:
        # Deliberate choice: start anyway, in a degraded state, rather
        # than refusing to boot. That keeps /health reachable so a
        # monitor can *report* "up but cannot predict" instead of just
        # seeing a dead port. The cost is that the failure is quieter —
        # so it is logged at ERROR, and both /health and /predict answer
        # 503 rather than pretending.
        logger.error(
            kv(event="model_load_failed", path=str(model.model_path), error=str(exc)),
            exc_info=True,
        )

    # app.state is FastAPI's designated place for objects shared across
    # requests. The get_model dependency reads it, so no route reaches
    # for a module-level global that tests can't substitute.
    app.state.model = model

    logger.info(kv(event="startup_complete", model_loaded=model.is_loaded))
    yield
    logger.info(kv(event="shutdown", model_loaded=model.is_loaded))
    model.unload()
    app.state.model = None


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
)

# This adds scrapeable HTTP request counters and latency histograms at the
# public /metrics endpoint. It intentionally sits outside the API-key-protected
# versioned routers: Prometheus scrapes it with network-level access control.
Instrumentator().instrument(app).expose(app, include_in_schema=False)

# Browser clients are restricted to explicitly configured origins.  The
# default is deliberately deny-by-default, rather than a permissive "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request, and give each one a traceable id.

    The id is stored in two places on purpose:
      * request.state.request_id — so endpoints can read it (via the
        get_request_id dependency) and put it in the response body;
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
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": jsonable_encoder(exc.errors()),
            "request_id": _request_id_of(request),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Give deliberately-raised HTTPExceptions the same body shape as everything else.

    FastAPI's built-in handler returns {"detail": ...} with no request_id,
    which would make 413s and 404s the odd ones out. A client should never
    have to branch on status code just to find the correlation id.
    """
    logger.warning(
        kv(
            event="http_exception",
            path=request.url.path,
            status=exc.status_code,
            detail=str(exc.detail),
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": _request_id_of(request)},
        headers=getattr(exc, "headers", None),
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
# Routers — one line per API version
# --------------------------------------------------------------------------
app.include_router(v1.router)
app.include_router(v2.router)


@app.get("/", tags=["ops"])
def root() -> dict[str, object]:
    """Unversioned entry point: says what versions exist and where docs are.

    Kept off the version prefix on purpose — a client that doesn't yet
    know which versions you serve has to be able to ask something.
    """
    return {
        "message": "ML API is alive",
        "docs": "/docs",
        "app_version": settings.api_version,
        "api_versions": ["/api/v1", "/api/v2"],
    }
