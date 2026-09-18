#!/usr/bin/env python3
"""Small dependency-free concurrent HTTP load check for the live API."""
from __future__ import annotations

import asyncio
import os
import statistics
import time

import httpx


BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "replace-with-a-long-random-secret")
REQUESTS = int(os.getenv("LOAD_TEST_REQUESTS", "100"))
CONCURRENCY = int(os.getenv("LOAD_TEST_CONCURRENCY", "25"))
SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


async def main() -> None:
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        async def one_request() -> tuple[int, float]:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post("/api/v1/predict", headers={"X-API-Key": API_KEY}, json=SAMPLE)
                return response.status_code, (time.perf_counter() - started) * 1000

        results = await asyncio.gather(*(one_request() for _ in range(REQUESTS)), return_exceptions=True)

    completed = [result for result in results if not isinstance(result, Exception)]
    failures = len(results) - len(completed) + sum(status != 200 for status, _ in completed)
    latencies = [duration for _, duration in completed]
    print(f"requests={REQUESTS} concurrency={CONCURRENCY} successes={len(completed) - sum(s != 200 for s, _ in completed)} failures={failures}")
    if latencies:
        print(f"latency_ms mean={statistics.mean(latencies):.2f} p95={sorted(latencies)[max(0, int(len(latencies) * .95) - 1)]:.2f} max={max(latencies):.2f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
