"""
app.py - Sprint 3: Flask REST API for Wine Model
==================================================
Serves the trained XGBoost Wine classifier via a REST API.

Endpoints:
  GET  /health          → liveness check
  GET  /model/info      → model metadata & hyperparameters
  GET  /metrics         → latest evaluation metrics from metrics.json
  POST /predict         → single-sample prediction  (13 features)
  POST /predict/batch   → multi-sample predictions   (N × 13 features)
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, jsonify, request

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Wine dataset class names (matching sklearn.datasets.load_wine)
CLASS_NAMES = {0: "class_0", 1: "class_1", 2: "class_2"}

# Wine dataset feature names (13 features)
FEATURE_NAMES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline",
]


# ── Load model once at startup ───────────────────────────────────────────────
def load_model_bundle(path: str):
    """Load the pickled model + scaler bundle."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run train.py first to generate model/model.pkl"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


bundle = load_model_bundle(MODEL_PATH)
MODEL = bundle["model"]
SCALER = bundle["scaler"]
N_FEATURES = MODEL.n_features_in_  # Expected: 13

print(f"✅ Model loaded — expects {N_FEATURES} features")


# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_features(features: list) -> np.ndarray:
    """Validate and reshape a flat list of feature values."""
    if not isinstance(features, list):
        raise ValueError("'features' must be a list of numbers.")
    if len(features) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, got {len(features)}. "
            f"Features: {FEATURE_NAMES}"
        )
    try:
        return np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        raise ValueError("All feature values must be numeric.")


def load_metrics():
    """Read metrics.json from disk. Returns dict or None."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def format_prediction(pred, probabilities):
    """Format a single prediction result."""
    return {
        "prediction": int(pred),
        "label": CLASS_NAMES.get(int(pred), str(pred)),
        "probabilities": {
            CLASS_NAMES.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — confirms the API and model are operational."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
    }), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return metadata about the loaded model."""
    return jsonify({
        "model_type": type(MODEL).__name__,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "n_classes": len(MODEL.classes_.tolist()),
        "classes": MODEL.classes_.tolist(),
        "class_names": list(CLASS_NAMES.values()),
        "hyperparameters": {
            "n_estimators": getattr(MODEL, "n_estimators", None),
            "max_depth": getattr(MODEL, "max_depth", None),
            "learning_rate": getattr(MODEL, "learning_rate", None),
        },
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return the latest evaluation metrics from metrics.json."""
    data = load_metrics()
    if data is None:
        return jsonify({
            "error": "metrics.json not found. Run evaluate.py first."
        }), 404
    return jsonify(data), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single prediction.

    Request body (JSON):
      { "features": [f1, f2, ..., f13] }

    Response:
      {
        "prediction": 0,
        "label": "class_0",
        "probabilities": {"class_0": 0.92, "class_1": 0.05, "class_2": 0.03}
      }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "features" not in data:
        return jsonify({
            "error": "Request body must contain a 'features' key."
        }), 400

    try:
        X = validate_features(data["features"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    X_scaled = SCALER.transform(X)
    pred = MODEL.predict(X_scaled)[0]
    proba = MODEL.predict_proba(X_scaled)[0]

    return jsonify(format_prediction(pred, proba)), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch predictions.

    Request body (JSON):
      { "instances": [[f1...f13], [f1...f13], ...] }

    Response:
      {
        "predictions": [ {prediction, label, probabilities}, ... ],
        "count": N
      }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "instances" not in data:
        return jsonify({
            "error": "Request body must contain an 'instances' key."
        }), 400

    instances = data["instances"]
    if not isinstance(instances, list) or len(instances) == 0:
        return jsonify({
            "error": "'instances' must be a non-empty list."
        }), 400

    try:
        X = np.array(instances, dtype=float)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(
                f"Each instance must have exactly {N_FEATURES} features."
            )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    X_scaled = SCALER.transform(X)
    preds = MODEL.predict(X_scaled)
    probas = MODEL.predict_proba(X_scaled)

    results = [
        format_prediction(p, pb)
        for p, pb in zip(preds, probas)
    ]

    return jsonify({"predictions": results, "count": len(results)}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Starting Flask API on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
