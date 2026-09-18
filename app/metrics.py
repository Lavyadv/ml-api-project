"""Application-specific Prometheus metrics.

The instrumentator supplies HTTP request count, duration, and response-size
metrics.  This module holds metrics that only the ML application understands.
"""
from prometheus_client import Counter


PREDICTIONS_TOTAL = Counter(
    "iris_predictions_total",
    "Number of successfully produced Iris predictions.",
    labelnames=("api_version", "predicted_class"),
)
