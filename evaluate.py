"""
evaluate.py - Evaluate the trained model and enforce the accuracy gate.
Exits with code 1 (failing the CI pipeline) if accuracy < 0.8.
"""

import os
import pickle
import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.80
RANDOM_STATE = 42
MODEL_PATH = os.path.join("model", "model.pkl")

# ── Load model bundle ─────────────────────────────────────────────────────────
print("📂 Loading model from", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    print("❌ model.pkl not found. Did you run train.py first?")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
print("   Model loaded successfully.")

# ── Recreate the same test split ─────────────────────────────────────────────
print("\n📦 Re-generating dataset with the same seed...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=RANDOM_STATE,
)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

X_test_scaled = scaler.transform(X_test)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("         MODEL EVALUATION REPORT")
print("=" * 50)
print(f"\n  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"  Threshold: {ACCURACY_THRESHOLD:.2f}  ({ACCURACY_THRESHOLD * 100:.0f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
print("=" * 50)

# ── Gate logic ────────────────────────────────────────────────────────────────
if accuracy >= ACCURACY_THRESHOLD:
    print(f"\n✅ PASSED — accuracy {accuracy:.4f} >= threshold {ACCURACY_THRESHOLD}")
    sys.exit(0)
else:
    print(f"\n❌ FAILED — accuracy {accuracy:.4f} < threshold {ACCURACY_THRESHOLD}")
    print("   Pipeline halted. Improve the model before merging.")
    sys.exit(1)
