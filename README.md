# ML API Project — Iris Classifier

## What this is
A small REST API that serves a machine learning model. The goal of this
project is not model complexity — it's proving I can take a trained model
and wrap it in a real, well-structured service.

## Dataset / Problem
**Dataset:** scikit-learn's built-in `load_iris`
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

## Project layout
```
ml-api-project/
├── app/
│   ├── main.py           # FastAPI app + routes
│   ├── models/           # Pydantic schemas
│   └── routers/          # route files (as the API grows)
├── ml/
│   ├── train.py          # trains + saves the model
│   └── saved_model/      # model.joblib lives here
├── tests/
├── requirements.txt
└── .gitignore
```

## Status
- [x] Task 1 — Plan + dataset chosen
- [x] Task 2 — Environment + folder structure
- [x] Task 3 — Model trained and saved
- [x] Task 4 — Bare-bones FastAPI app running
