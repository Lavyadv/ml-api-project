# End-to-end and load testing

These checks deliberately talk to the running Docker container over HTTP. They
complement the in-process pytest suite; they do not replace it.

## Repeatable procedure

```bash
cp .env.example .env             # if it does not already exist
docker compose up --build -d
docker compose ps                # wait until api is healthy
API_KEY='your-value-from-.env' python scripts/integration_check.py
API_KEY='your-value-from-.env' LOAD_TEST_REQUESTS=100 LOAD_TEST_CONCURRENCY=25 \
  python scripts/load_test.py
curl -s http://localhost:8000/metrics | grep -E 'iris_predictions_total|http_request'
docker compose down
```

The integration script checks `/api/v1/health`, `/api/v1/predict`,
`/api/v1/predict-batch`, and `/metrics`. It also verifies that the custom
`iris_predictions_total` metric appears after successful inference.

The load script sends 100 prediction requests with at most 25 in flight. It
reports successful/failed requests plus mean, p95, and maximum client-observed
latency, and exits non-zero if any request fails. Change the two environment
variables to test a different load level.

## What was fixed

The API originally had no operational visibility: a client could get an error
without a time-series view of request volume, status codes, or latency. The
Prometheus instrumentator now exports those HTTP metrics at `/metrics`, and the
application adds `iris_predictions_total{api_version, predicted_class}` only
after a prediction succeeds. This makes both traffic and model-output mix
observable without recording request payloads.

Prometheus is included in Compose and scrapes the API every five seconds. Open
<http://localhost:9090/targets> and confirm that `iris-api` is `UP`; query
`iris_predictions_total` in its graph UI. The API healthcheck uses `/metrics`
because that endpoint is public for scraper access and does not require an API
key.
