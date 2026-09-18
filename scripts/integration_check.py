#!/usr/bin/env python3
"""Run HTTP checks against a live Compose API, never an in-process TestClient."""
from __future__ import annotations

import os
import sys

import httpx


BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "replace-with-a-long-random-secret")
SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def check(response: httpx.Response, expected: int, label: str) -> None:
    if response.status_code != expected:
        raise AssertionError(f"{label}: expected {expected}, got {response.status_code}: {response.text}")
    print(f"PASS {label} ({response.status_code})")


def main() -> int:
    headers = {"X-API-Key": API_KEY}
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        check(client.get("/api/v1/health", headers=headers), 200, "health")
        check(client.post("/api/v1/predict", headers=headers, json=SAMPLE), 200, "predict")
        check(client.post("/api/v1/predict-batch", headers=headers, json={"items": [SAMPLE, SAMPLE]}), 200, "predict-batch")
        metrics = client.get("/metrics")
        check(metrics, 200, "metrics")
        if "iris_predictions_total" not in metrics.text:
            raise AssertionError("metrics: custom iris_predictions_total metric is missing")
        print("PASS custom prediction metric")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (httpx.HTTPError, AssertionError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
