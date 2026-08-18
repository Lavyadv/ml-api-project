"""
Train a simple classifier on the Iris dataset and save it to disk.

Uses a scikit-learn Pipeline (scaler + model) so that the exact same
preprocessing used at training time is automatically reapplied at
prediction time — no separate scaler to track by hand.
"""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "ml/saved_model/model.joblib"


def main():
    data = load_iris()
    X, y = data.data, data.target
    target_names = data.target_names

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

    joblib.dump({"pipeline": pipeline, "target_names": list(target_names)}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
