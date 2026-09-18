# ML API Project — Iris Classifier

## What this is
A small REST API that serves a machine learning model. The goal of this
project is not model complexity — it's proving I can take a trained model
and wrap it in a real, well-structured service.

## Dataset / Problem
**Dataset:** the classic Iris dataset, stored in this repo at `data/iris.csv`
(150 rows, 50 per class — identical to scikit-learn's built-in `load_iris`).
**Problem:** Multi-class classification — predict the species of an iris
flower (setosa / versicolor / virginica) from 4 numeric measurements
(sepal length, sepal width, petal length, petal width).

## API Contract (plain English)
- **Endpoint:** `POST /predict`
- **Input:** JSON object with 4 floats — `sepal_length`, `sepal_width`,
  `petal_length`, `petal_width` (all in cm).
- **Output:** JSON object with the predicted species name (string) and
  the model's confidence for that prediction (float, 0–1).
- **Errors:** if a field is missing or not a number, the API returns a
  `422 Unprocessable Entity` with a message pointing at the bad field
  (FastAPI + Pydantic handle this automatically once validation is added).

## Request → Validation → Model → Response flow
1. Client sends a POST request to `/predict` with 4 measurements as JSON.
2. FastAPI parses the body against a Pydantic schema — if a field is
   missing, the wrong type, or out of a sane range, the request is
   rejected before it ever reaches the model.
3. The validated data is turned into the exact shape the model expects
   (same order of features used during training) and passed to the
   loaded `model.joblib`.
4. The model returns a class prediction + probability. The API wraps
   that in a clean JSON response and sends it back.

## Endpoints
All business routes are namespaced under a version prefix. Only `/` and
`/docs` sit outside it — a client that doesn't yet know which versions
you serve has to be able to ask something.

| Method | Path                       | Purpose                                     |
|--------|----------------------------|---------------------------------------------|
| GET    | `/`                        | Lists available API versions                |
| GET    | `/docs`                    | Interactive Swagger UI                      |
| GET    | `/api/v1/health`           | 200 when a model is loaded, 503 when not    |
| GET    | `/api/v1/model-info`       | What model is serving traffic               |
| POST   | `/api/v1/predict`          | Score one flower                            |
| POST   | `/api/v1/predict-batch`    | Score up to `MAX_BATCH_SIZE` rows at once   |
| POST   | `/api/v2/predict`          | Same model, new response shape              |

### v1 vs v2 — what changed and why it needed a new version
| v1                            | v2                              | Breaking? |
|-------------------------------|---------------------------------|-----------|
| `confidence: float`           | `probability: float`            | Yes — rename |
| `probabilities: {str: float}` | `ranked: [{species, probability}]` | Yes — reshape, now sorted |
| *(absent)*                    | `trained_at: str \| null`       | No — additive |

Only the first two forced the version bump. A client doing
`response["confidence"]` raises `KeyError` against v2; one doing
`response["probabilities"]["setosa"]` gets a list it can't subscript by
name. The added field breaks nobody — a well-behaved client ignores
fields it doesn't recognise, which is why `trained_at` on its own would
not have justified v2.

`tests/test_versioning.py` asserts this with code rather than claiming it
in prose: it pins v1's exact field set, so any future edit to the v1
contract fails the suite and forces the question "should this have been
v3?"

### Response shapes
Success (`200`, v1):
```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
  "model_version": "sha256:5894f9a8604a",
  "request_id": "0f8fad5b-d9cb-469f-a165-70867728950e"
}
```
Every error — `404`, `413`, `422`, `500`, `503` — has the same two keys,
`detail` and `request_id`, so a client never has to branch on status code
just to find the correlation id. Quote the `request_id` (also returned as
the `X-Request-ID` header) to find that exact request in `logs/api.log`.

## Configuration
Nothing in the app reads `os.getenv()` directly; everything goes through
`app/config.py`. Copy `.env.example` to `.env` and edit — or set real
environment variables, which take precedence:

```bash
MAX_BATCH_SIZE=25 LOG_LEVEL=DEBUG uvicorn app.main:app
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `API_TITLE` | Iris Classifier API | Shown on `/docs` |
| `API_VERSION` | 1.0.0 | App version (distinct from the API's `/v1`, `/v2`) |
| `MODEL_PATH` | `ml/saved_model/model.joblib` | Relative paths resolve against the project root |
| `LOG_LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| `LOG_DIR` | `logs` | Where `api.log` is written |
| `LOG_MAX_BYTES` | 1000000 | Rotate at this size |
| `LOG_BACKUP_COUNT` | 5 | How many rotated files to keep |
| `MAX_BATCH_SIZE` | 100 | Largest accepted batch; over this returns 413 |

`.env` is gitignored and `.env.example` is committed, so the repo
documents what's configurable without carrying anyone's local values.

## Run with Docker Compose

Docker Compose is the normal one-command way to start the complete API:

```bash
cp .env.example .env
# Edit .env and replace API_KEY with a long random secret.
docker compose up --build
```

Open <http://localhost:8000/docs>. The API routes require the
`X-API-Key` header, for example:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace-with-the-value-from-your-.env' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Use `docker compose down` to stop it. The model artifact is copied into the
image, which keeps Compose portable on Docker Desktop without requiring a
host-directory sharing rule. After retraining, rebuild the API image with
`docker compose up --build` to serve the new artifact.

The Dockerfile runs Uvicorn with `--host 0.0.0.0`: that listens on all
container network interfaces, so Docker can forward host port 8000 into the
container. Binding to `127.0.0.1` would make the server reachable only from
inside that container.

### Monitoring

Compose also starts Prometheus at <http://localhost:9090>. It scrapes the API's
public `GET /metrics` endpoint every five seconds; open
<http://localhost:9090/targets> to see the `iris-api` target become `UP`.
`/metrics` returns Prometheus text format with request counts, status codes,
and latency histograms supplied by `prometheus-fastapi-instrumentator`.

The application additionally exports
`iris_predictions_total{api_version, predicted_class}`. It increments only
after a successful single or batch prediction, so it is a safe way to observe
model output volume and class mix without exposing input values. `/metrics` is
intentionally not API-key protected: limit access at the network/ingress layer
in a real deployment, or restrict it to Prometheus's network.

## Example requests

All versioned endpoints require `X-API-Key` (shown below as `$API_KEY`):

```bash
export API_KEY='replace-with-the-value-from-your-.env'
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/model-info
curl -X POST http://localhost:8000/api/v1/predict -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
curl -X POST http://localhost:8000/api/v1/predict-batch -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" -d '{"items":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]}'
curl -X POST http://localhost:8000/api/v2/predict -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
curl http://localhost:8000/metrics
```

## Security

All versioned API routes require `X-API-Key`; the value comes only from
`API_KEY` in the environment or `.env`, never from source code. Requests
without the matching key receive `401 Unauthorized`. Browser access is also
limited to the comma-separated `CORS_ORIGINS` configured in `.env`; there is
no wildcard origin. Request schemas use `extra="forbid"`, positive bounded
measurements, and a cross-field petal-size check so malformed data is
rejected with a useful `422` before model inference.

## Testing
```bash
source venv/bin/activate
pytest -v
```
36 tests covering the happy path, every validation failure, batch sizing
and ordering, model metadata, and cross-version compatibility. They use
FastAPI's `TestClient`, which calls the app in-process — no server, no
port, no network.

The suite is meaningful, not decorative: disabling the petal validator
makes `test_predict_rejects_bad_input_with_422[petal values swapped]`
fail, and the captured log shows why it matters — the swapped input
doesn't error, it returns a confident-looking `versicolor` prediction.
Silent wrongness is exactly what that validator prevents.

For container-level checks and a basic concurrent load test, see
[TESTING.md](TESTING.md). These scripts use real HTTP against Compose rather
than `TestClient`; the load test defaults to 100 requests with 25 concurrent
requests and fails if any response is unsuccessful.

## Model explanations and reusable prediction

The saved artifact is a fitted scikit-learn `Pipeline` containing both the
`StandardScaler` and `RandomForestClassifier`; training writes the API bundle
(`model.joblib`) and portable hand-off artifact (`iris_pipeline.pkl`). Create
both and generate the explainability outputs with:

```bash
source venv/bin/activate
python ml/train.py
python ml/explain_model.py
```

This writes a global Random Forest feature-importance chart and two LIME
explanations (HTML and PNG) to `ml/explanations/`, plus a generated
plain-English interpretation in `ml/explanations/README.md`. The samples are
two predicted Iris species because this project classifies flowers rather than
customers or churn.

For validated, non-HTTP use, call `predict_new_customer()` from
`ml/predict_new.py`, or run its standalone example:

```bash
python ml/predict_new.py
```

It accepts a feature dictionary or one-row DataFrame and rejects missing,
extra, non-numeric, non-finite, out-of-range, and internally inconsistent
measurements with a clear `ValueError` before it invokes the saved pipeline.

## Design decisions worth knowing
**The model loads once, at startup.** `app/main.py` uses a `lifespan`
context manager, so the joblib file is read exactly once per process
rather than on every request. Loading inside the endpoint would turn a
~1 ms prediction into a much slower disk-and-deserialize round trip.

**Feature order comes from the model, not the request.** `train.py`
saves the feature list alongside the pipeline, and `IrisModel._to_array`
builds the array by looking up each name in that saved order. Reordering
the JSON keys therefore cannot silently feed petal width into the sepal
length column, and a name the model expects but the payload lacks raises
`InferenceError` instead of quietly defaulting to zero.

**A missing model file does not stop the server from booting.** It
starts in a degraded state: the failure is logged at ERROR with a full
traceback, `/health` answers 503 with `model_loaded: false`, and
`/predict` answers 503. The trade-off is deliberate — a monitor can then
report "up but unable to predict" rather than just seeing a dead port.
The alternative (crash on boot) surfaces the problem more loudly; if
this were deployed behind an orchestrator that restarts unhealthy
containers, crashing would be the better choice.

**Errors never leak internals.** Exception handlers in `main.py` convert
every failure into a fixed, safe sentence. The real exception and its
traceback go to the log, never to the client.

**Batching calls scikit-learn once, not in a loop.** `/predict-batch`
builds one `(n, 4)` matrix and makes a single `predict()` call. The
per-call overhead — validation, dispatch, walking all 100 trees — is
then paid once instead of n times. Measured on this model:

| rows | one batch call | loop of n calls | |
|------|----------------|-----------------|--|
| 10   | 8.8 ms         | 53.5 ms         | 6x |
| 100  | 6.5 ms         | 487.2 ms        | 75x |

**`response_model` doubles as an output filter.** `IrisModel.info()`
returns `model_path`, but `ModelInfoResponse` doesn't declare that field,
so it's stripped before the response leaves. Declaring the output shape
keeps a server filesystem path from reaching clients — the same discipline
that stops tracebacks leaking.

**Known limitation.** `RotatingFileHandler` is not safe across multiple
processes, so running uvicorn with `--workers N` could interleave or
clobber rotations. Single-process is fine for now.

## Project layout
```
ml-api-project/
├── data/
│   └── iris.csv              # training data
├── app/
│   ├── main.py               # FastAPI app, lifespan, middleware, handlers, routes
│   ├── ml_model.py           # loads model.joblib, shapes arrays, runs inference
│   ├── logging_config.py     # console + rotating-file logging, request_id plumbing
│   ├── config.py             # pydantic-settings: all config in one place
│   ├── dependencies.py       # shared Depends(): get_model, get_request_id
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response schemas (v1 + v2)
│   └── routers/
│       ├── v1.py             # /api/v1 — frozen contract
│       └── v2.py             # /api/v2 — breaking response shape
├── ml/
│   ├── train.py              # reads data/iris.csv, trains + saves the model
│   ├── verify_load.py        # proves the saved model reloads
│   └── saved_model/          # model.joblib lives here
├── logs/                     # api.log (gitignored, created at startup)
├── tests/                    # pytest suite (36 tests)
├── .env                      # local config (gitignored)
├── .env.example              # documents every variable (committed)
├── pytest.ini
├── requirements.txt
└── .gitignore
```

## Status
- [x] Task 1 — Plan + dataset chosen
- [x] Task 2 — Environment + folder structure
- [x] Task 3 — Model trained and saved
- [x] Task 4 — Bare-bones FastAPI app running
- [x] Task 5 — Real model loaded once at startup via `lifespan`
- [x] Task 6 — Pydantic input validation (422s instead of crashes)
- [x] Task 7 — Milestone: core API assembled (`/predict` + `/health`)
- [x] Task 8 — Response model, status codes, exception handlers
- [x] Task 9 — Structured logging to console + rotating file
- [x] Task 10 — API versioning (`/api/v1` via `APIRouter`)
- [x] Task 11 — Batch prediction + model metadata endpoints
- [x] Task 12 — Configuration via `pydantic-settings` + `.env`
- [x] Task 13 — Automated test suite with pytest
- [x] Task 14 — Milestone: `/api/v2/predict` breaking change, v1 untouched
- [x] Task 15 — Docker packaging
- [x] Task 16 — Docker Compose
- [x] Task 17 — API key, CORS, and input hardening
- [x] Task 18 — Prometheus metrics and custom prediction counter
- [x] Task 19 — Repeatable Compose HTTP integration and load checks

## What I learned

Serving a model is more than returning a prediction. Validation protects the
model from malformed inputs, versioned routes preserve client contracts, and
request IDs make an individual failure traceable. Container-level tests catch
problems that unit tests cannot, while Prometheus turns request volume,
latency, failures, and successful prediction classes into evidence that the
service is healthy.

## Independent Extension: Prometheus monitoring

Beyond basic endpoint instrumentation, this project adds a model-aware,
labelled prediction counter and a Prometheus service wired into Compose. The
counter distinguishes API version and predicted Iris class, which makes it
possible to spot a class-distribution shift alongside operational metrics.
