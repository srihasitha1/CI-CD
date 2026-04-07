"""
train.py - Train a classification model using scikit-learn
Generates synthetic data and trains a RandomForest classifier.
Saves the trained model to model/model.pkl
"""

import os
import pickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Generate synthetic dataset ────────────────────────────────────────────────
print("📦 Generating synthetic classification dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=RANDOM_STATE,
)
print(f"   Dataset shape: X={X.shape}, y={y.shape}")
print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\n✂️  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── Feature scaling ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Train model ───────────────────────────────────────────────────────────────
print("\n🌲 Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)
print("   Training complete.")

# ── Quick accuracy check ──────────────────────────────────────────────────────
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
print(f"\n📊 Train accuracy : {train_acc:.4f}")
print(f"📊 Test  accuracy : {test_acc:.4f}")

# ── Persist model + scaler ────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

bundle = {"model": model, "scaler": scaler}
model_path = os.path.join("model", "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"\n✅ Model saved to {model_path}")
