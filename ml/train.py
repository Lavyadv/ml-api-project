"""
Train a simple classifier on the Iris dataset and save it to disk.

Reads the dataset from data/iris.csv so the exact data the model was
trained on lives in the repo alongside the code.

Uses a scikit-learn Pipeline (scaler + model) so that the exact same
preprocessing used at training time is automatically reapplied at
prediction time — no separate scaler to track by hand.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/iris.csv"
MODEL_PATH = "ml/saved_model/model.joblib"

# Order matters: the API must send features in this exact order at predict time.
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"


def load_data(path=DATA_PATH):
    """Load the dataset from CSV and return features, integer labels, class names."""
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if df.isna().any().any():
        raise ValueError(f"{path} contains missing values")

    # Sorted so the label -> index mapping is stable across runs.
    target_names = sorted(df[TARGET_COLUMN].unique())

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].map({name: i for i, name in enumerate(target_names)}).to_numpy()
    return X, y, target_names


def main():
    X, y, target_names = load_data()
    print(f"Loaded {len(X)} rows from {DATA_PATH} ({len(target_names)} classes)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline bundles preprocessing + model together so both are saved
    # and reloaded as a single unit.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    joblib.dump(
        {"pipeline": pipeline, "target_names": list(target_names), "features": FEATURE_COLUMNS},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
