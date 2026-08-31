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
| Method | Path       | Purpose                                              |
|--------|------------|------------------------------------------------------|
| GET    | `/`        | Liveness ping                                        |
| GET    | `/health`  | 200 when a model is loaded, 503 when not             |
| POST   | `/predict` | Species prediction + confidence                      |
| GET    | `/docs`    | Interactive Swagger UI                               |

### Response shapes
Success (`200`):
```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
  "model_version": "sha256:9ee4cdf60dd8",
  "request_id": "0f8fad5b-d9cb-469f-a165-70867728950e"
}
```
Every error (`422`, `500`, `503`) has the same two keys — `detail` and
`request_id` — so a client never has to guess what an error looks like.
Quote the `request_id` (also returned as the `X-Request-ID` header) to
find that exact request in `logs/api.log`.

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
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response schemas
│   └── routers/              # route files (as the API grows)
├── ml/
│   ├── train.py              # reads data/iris.csv, trains + saves the model
│   ├── verify_load.py        # proves the saved model reloads
│   └── saved_model/          # model.joblib lives here
├── logs/                     # api.log (gitignored, created at startup)
├── tests/
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
- [ ] Task 10 — API versioning
