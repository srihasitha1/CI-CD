"""
tests/test_pipeline.py - Sprint 3: Pytest Test Suite
======================================================
Six tests validating the ML pipeline end-to-end:
  1. Model file exists on disk
  2. Model loads correctly (contains model + scaler)
  3. Prediction returns a valid Wine class [0, 1, 2]
  4. API /health endpoint returns 200
  5. Valid /predict request returns 200 with correct schema
  6. Invalid /predict request returns 400
"""

import os
import sys
import pickle
import pytest
import numpy as np

# ── Ensure project root is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "model.pkl")
N_FEATURES = 13   # Wine dataset has 13 features
VALID_CLASSES = [0, 1, 2]

# A realistic sample from the Wine dataset (class 0 — first row of load_wine)
SAMPLE_FEATURES = [
    14.23, 1.71, 2.43, 15.6, 127.0,
    2.80, 3.06, 0.28, 2.29, 5.64,
    1.04, 3.92, 1065.0,
]


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def model_bundle():
    """Load and return the saved model bundle."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Test 1: Model file exists ────────────────────────────────────────────────
def test_model_file_exists():
    """Verify that model/model.pkl exists on disk after training."""
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found at {MODEL_PATH}. Run train.py first."
    )


# ── Test 2: Model loads correctly ────────────────────────────────────────────
def test_model_loads_correctly(model_bundle):
    """Verify the saved bundle contains a model and a scaler."""
    assert "model" in model_bundle, "Bundle missing 'model' key."
    assert "scaler" in model_bundle, "Bundle missing 'scaler' key."

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    # Model should have the standard sklearn interface
    assert hasattr(model, "predict"), "Model lacks predict() method."
    assert hasattr(model, "predict_proba"), "Model lacks predict_proba() method."
    assert model.n_features_in_ == N_FEATURES, (
        f"Model expects {model.n_features_in_} features, expected {N_FEATURES}."
    )

    # Scaler should be fitted
    assert hasattr(scaler, "mean_"), "Scaler is not fitted."
    assert len(scaler.mean_) == N_FEATURES, (
        f"Scaler fitted on {len(scaler.mean_)} features, expected {N_FEATURES}."
    )


# ── Test 3: Prediction returns a valid class ─────────────────────────────────
def test_prediction_valid_class(model_bundle):
    """Verify the model predicts valid Wine classes [0, 1, 2]."""
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    X = np.array(SAMPLE_FEATURES).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])

    assert prediction in VALID_CLASSES, (
        f"Prediction {prediction} not in valid classes {VALID_CLASSES}."
    )

    # Also verify probabilities sum to ~1
    proba = model.predict_proba(X_scaled)[0]
    assert abs(sum(proba) - 1.0) < 1e-6, (
        f"Probabilities sum to {sum(proba)}, expected ~1.0."
    )


# ── Test 4: API /health returns 200 ──────────────────────────────────────────
def test_health_endpoint(client):
    """Verify GET /health returns 200 with correct payload."""
    response = client.get("/health")

    assert response.status_code == 200, (
        f"/health returned {response.status_code}, expected 200."
    )

    data = response.get_json()
    assert data["status"] == "ok", f"Expected status 'ok', got '{data['status']}'."
    assert data["model_loaded"] is True, "Model should be loaded."


# ── Test 5: Valid /predict returns 200 ────────────────────────────────────────
def test_predict_valid_input(client):
    """Verify POST /predict with valid 13-feature input returns 200."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})

    assert response.status_code == 200, (
        f"/predict returned {response.status_code}, expected 200."
    )

    data = response.get_json()

    # Response must contain required keys
    assert "prediction" in data, "Response missing 'prediction' key."
    assert "label" in data, "Response missing 'label' key."
    assert "probabilities" in data, "Response missing 'probabilities' key."

    # Prediction must be a valid class
    assert data["prediction"] in VALID_CLASSES, (
        f"Prediction {data['prediction']} not in {VALID_CLASSES}."
    )

    # Must have probabilities for all 3 classes
    assert len(data["probabilities"]) == 3, (
        f"Expected 3 probability entries, got {len(data['probabilities'])}."
    )


# ── Test 6: Invalid /predict returns 400 ─────────────────────────────────────
def test_predict_invalid_input(client):
    """Verify POST /predict with wrong feature count returns 400."""
    # Send only 5 features instead of 13
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]})

    assert response.status_code == 400, (
        f"/predict returned {response.status_code}, expected 400 for invalid input."
    )

    data = response.get_json()
    assert "error" in data, "Error response should contain an 'error' key."

    # Also test missing features key entirely
    response_no_key = client.post("/predict", json={"data": [1.0]})
    assert response_no_key.status_code == 400, (
        "Missing 'features' key should return 400."
    )
def test_intentional_failure():
    assert 1 == 2, "Testing pipeline test failure!"
