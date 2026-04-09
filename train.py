"""
train.py - Sprint 1: CI/CD ML Pipeline
========================================
Trains an XGBoost classifier on the Wine dataset with hyperparameter
tuning via GridSearchCV. Saves the best model + scaler bundle to
model/model.pkl, backing up any existing model to model/model_previous.pkl.
"""

import os
import shutil
import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PREV_MODEL_PATH = os.path.join(MODEL_DIR, "model_previous.pkl")

# Hyperparameter search space
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
}

np.random.seed(RANDOM_STATE)


def load_data():
    """Load the Wine dataset and return features and labels."""
    print("📦 Loading Wine dataset...")
    data = load_wine()
    X, y = data.data, data.target
    print(f"   Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Classes: {len(np.unique(y))}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def split_data(X, y):
    """Perform stratified 80/20 train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n✂️  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Fit StandardScaler on training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("📐 Feature scaling applied (StandardScaler).")
    return scaler, X_train_scaled, X_test_scaled


def tune_and_train(X_train, y_train):
    """
    Run GridSearchCV over XGBClassifier with the defined param grid.
    Returns the best estimator found.
    """
    print(f"\n🔍 Starting GridSearchCV ({CV_FOLDS}-fold CV)...")
    print(f"   Param grid: {PARAM_GRID}")

    base_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"\n🏆 Best parameters: {grid_search.best_params_}")
    print(f"🏆 Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate(model, X_test, y_test):
    """Print accuracy and full classification report on the test set."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 Test accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    return acc


def save_model(model, scaler):
    """
    Save model + scaler bundle to MODEL_PATH.
    If a previous model exists, back it up to PREV_MODEL_PATH first.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Back up existing model ────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, PREV_MODEL_PATH)
        print(f"\n📂 Previous model backed up → {PREV_MODEL_PATH}")

    # ── Save new model bundle ─────────────────────────────────────────────────
    bundle = {"model": model, "scaler": scaler}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ Model + scaler saved → {MODEL_PATH}")


# ── Main execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load data
    X, y = load_data()

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Scale
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 4. Tune & train
    best_model = tune_and_train(X_train_scaled, y_train)

    # 5. Evaluate
    evaluate(best_model, X_test_scaled, y_test)

    # 6. Save
    save_model(best_model, scaler)
