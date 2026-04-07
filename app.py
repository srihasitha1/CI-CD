"""
app.py - Flask REST API for serving ML model predictions.

Endpoints:
  GET  /health          → liveness check
  GET  /model/info      → model metadata
  POST /predict         → single prediction
  POST /predict/batch   → batch predictions
"""

import os
import pickle
import numpy as np
from flask import Flask, jsonify, request

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
MODEL_PATH = os.path.join("model", "model.pkl")

# ── Load model once at startup ────────────────────────────────────────────────
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run train.py first to generate model/model.pkl"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


bundle = load_model(MODEL_PATH)
MODEL = bundle["model"]
SCALER = bundle["scaler"]

N_FEATURES = MODEL.n_features_in_
CLASS_NAMES = {0: "Class 0", 1: "Class 1"}

print(f"✅ Model loaded — expects {N_FEATURES} features")

# ── Helper ────────────────────────────────────────────────────────────────────
def validate_features(features: list) -> np.ndarray:
    """Validate and reshape a flat list of feature values."""
    if len(features) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, got {len(features)}."
        )
    return np.array(features, dtype=float).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok", "model_loaded": MODEL is not None}), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return metadata about the loaded model."""
    return jsonify(
        {
            "model_type": type(MODEL).__name__,
            "n_features": N_FEATURES,
            "n_classes": len(MODEL.classes_.tolist()),
            "classes": MODEL.classes_.tolist(),
            "n_estimators": getattr(MODEL, "n_estimators", None),
            "max_depth": getattr(MODEL, "max_depth", None),
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single prediction.

    Request body (JSON):
      { "features": [f1, f2, ..., f20] }

    Response:
      {
        "prediction": 0 or 1,
        "label": "Class 0" or "Class 1",
        "probabilities": {"Class 0": 0.12, "Class 1": 0.88}
      }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Request body must contain a 'features' key."}), 400

    try:
        X = validate_features(data["features"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    X_scaled = SCALER.transform(X)
    pred = int(MODEL.predict(X_scaled)[0])
    proba = MODEL.predict_proba(X_scaled)[0].tolist()

    return jsonify(
        {
            "prediction": pred,
            "label": CLASS_NAMES.get(pred, str(pred)),
            "probabilities": {
                CLASS_NAMES.get(i, str(i)): round(p, 4)
                for i, p in enumerate(proba)
            },
        }
    ), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction.

    Request body (JSON):
      { "instances": [[f1...f20], [f1...f20], ...] }

    Response:
      { "predictions": [{"prediction": 1, "label": "Class 1", ...}, ...] }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "instances" not in data:
        return jsonify({"error": "Request body must contain an 'instances' key."}), 400

    instances = data["instances"]
    if not isinstance(instances, list) or len(instances) == 0:
        return jsonify({"error": "'instances' must be a non-empty list."}), 400

    try:
        X = np.array(instances, dtype=float)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(
                f"Each instance must have exactly {N_FEATURES} features."
            )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    X_scaled = SCALER.transform(X)
    preds = MODEL.predict(X_scaled).tolist()
    probas = MODEL.predict_proba(X_scaled).tolist()

    results = [
        {
            "prediction": p,
            "label": CLASS_NAMES.get(p, str(p)),
            "probabilities": {
                CLASS_NAMES.get(i, str(i)): round(pr, 4)
                for i, pr in enumerate(pb)
            },
        }
        for p, pb in zip(preds, probas)
    ]

    return jsonify({"predictions": results, "count": len(results)}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Starting Flask API on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
