"""
evaluate.py - Sprint 2: Model Evaluation & Rollback
=====================================================
Loads the trained model from model/model.pkl, evaluates it on the Wine
test set, saves metrics to model/metrics.json, and enforces an accuracy
gate with automatic rollback if the threshold is not met.

Accuracy Gate:
    >= 0.82  →  PASS  (exit 0)
    <  0.82  →  ROLLBACK previous model, delete current  (exit 1)
"""

import os
import sys
import json
import shutil
import pickle
import numpy as np
from datetime import datetime, timezone
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Configuration ─────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.82
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PREV_MODEL_PATH = os.path.join(MODEL_DIR, "model_previous.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


# ── Helper functions ──────────────────────────────────────────────────────────

def load_model(path):
    """Load a pickled model+scaler bundle from disk."""
    if not os.path.exists(path):
        print(f"❌ {path} not found. Did you run train.py first?")
        sys.exit(1)

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    print(f"📂 Model loaded from {path}")
    return bundle["model"], bundle["scaler"]


def recreate_test_set():
    """
    Reproduce the exact same test split used during training.
    Uses identical RANDOM_STATE and TEST_SIZE as train.py.
    """
    print("📦 Loading Wine dataset (same seed as training)...")
    data = load_wine()
    X, y = data.data, data.target

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"   Test set size: {X_test.shape[0]} samples")
    return X_test, y_test


def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1 (all weighted)."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_weighted": round(
            precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "recall_weighted": round(
            recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "f1_weighted": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
    }
    return metrics


def save_metrics(metrics, path):
    """Save metrics dict to a JSON file with a timestamp."""
    metrics["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics["threshold"] = ACCURACY_THRESHOLD

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Metrics saved → {path}")


def print_report(metrics, y_true, y_pred, class_names):
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 55)
    print("           MODEL EVALUATION REPORT")
    print("=" * 55)
    print(f"\n  Accuracy          : {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision (wt)    : {metrics['precision_weighted']:.4f}")
    print(f"  Recall (wt)       : {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score (wt)     : {metrics['f1_weighted']:.4f}")
    print(f"  Accuracy Gate     : {ACCURACY_THRESHOLD:.2f}  ({ACCURACY_THRESHOLD * 100:.0f}%)")

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    # Print with class headers
    header = "       " + "  ".join(f"{name:>8}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>8}" for val in row)
        print(f"  {class_names[i]:>5} {row_str}")

    print("=" * 55)


def rollback():
    """
    Rollback: delete current model and restore the previous one.
    If no previous model exists, just delete the bad model.
    """
    # Delete the failing model
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"🗑️  Deleted failing model: {MODEL_PATH}")

    # Restore previous model if available
    if os.path.exists(PREV_MODEL_PATH):
        shutil.copy2(PREV_MODEL_PATH, MODEL_PATH)
        print(f"♻️  Restored previous model: {PREV_MODEL_PATH} → {MODEL_PATH}")
    else:
        print("⚠️  No previous model found to restore.")


# ── Main execution ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load the trained model
    model, scaler = load_model(MODEL_PATH)

    # 2. Recreate the test set (same split as training)
    X_test, y_test = recreate_test_set()

    # 3. Scale features using the saved scaler
    X_test_scaled = scaler.transform(X_test)

    # 4. Predict
    y_pred = model.predict(X_test_scaled)

    # 5. Compute metrics
    metrics = compute_metrics(y_test, y_pred)

    # 6. Get class names from the dataset
    class_names = load_wine().target_names.tolist()

    # 7. Print evaluation report
    print_report(metrics, y_test, y_pred, class_names)

    # 8. Save metrics to JSON
    save_metrics(metrics, METRICS_PATH)

    # 9. Accuracy gate with rollback logic
    if metrics["accuracy"] >= ACCURACY_THRESHOLD:
        print(f"\n✅ PASSED — accuracy {metrics['accuracy']:.4f} >= threshold {ACCURACY_THRESHOLD}")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED — accuracy {metrics['accuracy']:.4f} < threshold {ACCURACY_THRESHOLD}")
        print("🔄 ROLLBACK TRIGGERED — restored previous model")
        rollback()
        sys.exit(1)
