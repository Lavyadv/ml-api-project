"""Shared FastAPI dependencies.

These exist so v1 and v2 don't each grow their own copy of "find the
model" and "find the request id". A dependency is just a function
FastAPI calls before your endpoint and passes the result into it — the
natural place for logic every route needs but none of them own.
"""
from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings
from app.ml_model import IrisModel, ModelNotLoadedError

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Reject unauthenticated API requests without revealing key details."""
    configured_key = settings.api_key.get_secret_value() if settings.api_key else None
    if not configured_key or not api_key or not secrets.compare_digest(api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def get_model(request: Request) -> IrisModel:
    """Return the model loaded at startup, or refuse the request.

    Raises ModelNotLoadedError, which main.py's exception handler turns
    into a 503. Endpoints therefore never have to check for themselves —
    by the time one runs, a usable model is guaranteed.
    """
    model = getattr(request.app.state, "model", None)
    if model is None or not model.is_loaded:
        raise ModelNotLoadedError("Model is not loaded")
    return model


def get_request_id(request: Request) -> str:
    """The id the logging middleware assigned to this request."""
    return getattr(request.state, "request_id", "-")
