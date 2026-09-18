"""Shared pytest fixtures.

TestClient calls the app in-process — no server, no port, no network.
That is what makes these tests fast enough to run on every save.
"""
import os

import pytest
from fastapi.testclient import TestClient

# Set before app.config is imported, keeping the app fail-closed outside tests.
os.environ.setdefault("API_KEY", "test-api-key")
from app.config import settings
from app.main import app


@pytest.fixture(scope="session")
def client():
    """A client with the app's lifespan actually run.

    The `with` block matters: TestClient only triggers startup/shutdown
    events when used as a context manager. Without it the model would
    never load and every test would see a 503.
    """
    with TestClient(app) as test_client:
        test_client.headers.update({"X-API-Key": "test-api-key"})
        yield test_client


@pytest.fixture
def valid_payload():
    """A realistic setosa measurement."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


@pytest.fixture
def max_batch_size():
    """Read the limit from config so the test can't drift from the app."""
    return settings.max_batch_size
