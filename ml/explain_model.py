"""Generate global and local explanations for the saved Iris pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import gettempdir

import joblib

# This workspace's home-directory cache is read-only. Keep Matplotlib's
# generated cache in the OS temp area instead of emitting noisy warnings.
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "ml-api-matplotlib"))
import matplotlib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

try:  # Supports both `python ml/explain_model.py` and module execution.
    from ml.train import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH
except ModuleNotFoundError:  # pragma: no cover - exercised by direct-script use
    from train import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path("ml/explanations")


def main() -> None:
    """Save a feature-importance chart and two local LIME explanations."""
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    class_names = list(bundle["target_names"])
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "feature_importances_"):
        raise TypeError("The selected classifier does not expose tree feature importances")
    importances = classifier.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 4.5))
    plt.barh(np.asarray(FEATURE_COLUMNS)[order], importances[order], color="#3274a1")
    plt.xlabel("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_feature_importance.png", dpi=160)
    plt.close()

    explainer = LimeTabularExplainer(
        X,
        feature_names=FEATURE_COLUMNS,
        class_names=class_names,
        mode="classification",
        random_state=42,
    )
    predictions = pipeline.predict(X)
    # Pick two different predicted classes, rather than assuming a binary churn label.
    selected = [int(np.where(predictions == label)[0][0]) for label in (0, 2)]
    summaries = ["# Model explanations", "", "## Plain-language interpretation"]
    for index in selected:
        probabilities = pipeline.predict_proba(X[index : index + 1])[0]
        predicted_index = int(np.argmax(probabilities))
        explanation = explainer.explain_instance(
            X[index], pipeline.predict_proba, labels=(predicted_index,), num_features=len(FEATURE_COLUMNS)
        )
        label = class_names[predicted_index]
        slug = f"customer_{index}_{label}"
        explanation.save_to_file(str(OUTPUT_DIR / f"lime_{slug}.html"))
        figure = explanation.as_pyplot_figure(label=predicted_index)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"lime_{slug}.png", dpi=160)
        plt.close(figure)
        factors = explanation.as_list(label=predicted_index)[:2]
        summaries.extend([
            f"### Sample {index}: predicted **{label}** ({probabilities[predicted_index]:.1%})",
            f"LIME's strongest local signals were `{factors[0][0]}` and `{factors[1][0]}`.",
            (
                f"In plain business language: for this individual flower, those measurements are the "
                f"main reasons the model favours **{label}**. This explanation describes this one "
                "prediction, not a universal rule for every flower."
            ),
            "",
        ])
    (OUTPUT_DIR / "README.md").write_text("\n".join(summaries), encoding="utf-8")
    print(f"Saved explanations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
