"""Proves that the saved model can be reloaded and used without retraining."""
import joblib

bundle = joblib.load("ml/saved_model/model.joblib")
pipeline = bundle["pipeline"]
target_names = bundle["target_names"]

# A sample measurement (roughly a setosa)
sample = [[5.1, 3.5, 1.4, 0.2]]
pred_idx = pipeline.predict(sample)[0]
proba = pipeline.predict_proba(sample)[0]

print(f"Predicted class: {target_names[pred_idx]}")
print(f"Confidence: {proba[pred_idx]:.4f}")
