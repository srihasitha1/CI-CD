# ML CI/CD Pipeline

A complete end-to-end Machine Learning CI/CD pipeline using Python, scikit-learn, XGBoost, Flask, and GitHub Actions.

---

## Project Structure

```
.
├── train.py                        # Trains and saves the model
├── evaluate.py                     # Evaluates model; fails pipeline if accuracy < 0.82
├── app.py                          # Flask REST API
├── requirements.txt
├── model/
│   └── model.pkl                   # Generated after running train.py
└── .github/
    └── workflows/
        └── pipeline.yml            # GitHub Actions CI/CD definition
```

---

## Quick Start (Local)

### 1 — Clone and set up environment

```bash
git clone <your-repo-url>
cd ml-cicd-pipeline

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Train the model

```bash
python train.py
```

Expected output:
```
📦 Loading Wine dataset...
   Samples: 178  |  Features: 13  |  Classes: 3
   Class distribution: {0: 59, 1: 71, 2: 48}

✂️  Train size: 142  |  Test size: 36
📐 Feature scaling applied (StandardScaler).

🔍 Starting GridSearchCV (3-fold CV)...
   Param grid: {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
Fitting 3 folds for each of 8 candidates, totalling 24 fits

🏆 Best parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
🏆 Best CV accuracy: 0.9368

📊 Test accuracy: 0.9722
📂 Previous model backed up → model\model_previous.pkl
✅ Model + scaler saved → model\model.pkl
```

### 3 — Evaluate the model

```bash
python evaluate.py
```

The script exits with code `0` (pass) if accuracy ≥ 0.82, or code `1` (fail) otherwise (and rolls back to previous model).

### 4 — Run the Flask API

```bash
python app.py
```

The API starts at `http://localhost:5000`.

---

## API Reference

### `GET /health`

```bash
curl http://localhost:5000/health
```
```json
{"model_loaded": true, "status": "ok"}
```

### `GET /model/info`

```bash
curl http://localhost:5000/model/info
```
```json
{
  "class_names": ["class_0", "class_1", "class_2"],
  "classes": [0, 1, 2],
  "feature_names": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"],
  "hyperparameters": {
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 100
  },
  "model_type": "XGBClassifier",
  "n_classes": 3,
  "n_features": 13
}
```

### `POST /predict` — Single prediction

Send a JSON body with a `features` array of exactly 13 floats (Wine dataset):

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]}'
```
```json
{
  "label": "class_0",
  "prediction": 0,
  "probabilities": {
    "class_0": 0.98,
    "class_1": 0.01,
    "class_2": 0.01
  }
}
```

### `POST /predict/batch` — Batch predictions

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
      [12.29, 1.61, 2.21, 20.4, 103.0, 1.1, 1.02, 0.37, 1.46, 3.05, 0.906, 1.82, 870.0]
    ]
  }'
```
```json
{
  "count": 2,
  "predictions": [
    {"label": "class_0", "prediction": 0, "probabilities": {"class_0": 0.98, "class_1": 0.01, "class_2": 0.01}},
    {"label": "class_1", "prediction": 1, "probabilities": {"class_0": 0.12, "class_1": 0.85, "class_2": 0.03}}
  ]
}
```

---

## CI/CD Pipeline — How It Works

```
Push to any branch
        │
        ▼
┌─────────────────────────┐
│  1. Checkout code        │
│  2. Set up Python 3.11   │
│  3. pip install -r ...   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. python train.py      │  → writes model/model.pkl
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐    accuracy < 0.82 → ❌ PIPELINE FAILS (Rollback)
│  5. python evaluate.py  │
└────────────┬────────────┘    accuracy ≥ 0.82 → ✅ continue
             │
             ▼
┌─────────────────────────┐
│  6. Upload model.pkl     │  (GitHub Actions artifact, kept 7 days)
│     as artifact          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  7. Start Flask API      │
│  8. Smoke-test /health   │
│  9. Smoke-test /predict  │
└────────────┬────────────┘
             │
             ▼
        ✅ PASSED
```

### Why `sys.exit(1)` in evaluate.py?

GitHub Actions marks a step as failed when the process exits with a non-zero code. `evaluate.py` calls `sys.exit(1)` when accuracy is below threshold, which stops the pipeline at that step — preventing any deployment.

---

## Setting Up GitHub Actions

1. Push this project to a new GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial ML pipeline"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Go to **Actions** tab in your GitHub repository.

3. Every push will automatically trigger the pipeline. You can watch it run in real time.

4. Download the trained `model.pkl` artifact from the Actions run page after a successful run.

---

## Customising the Pipeline

| What to change | Where |
|---|---|
| Accuracy threshold | `evaluate.py` → `ACCURACY_THRESHOLD` |
| Model algorithm | `train.py` → replace `XGBClassifier` |
| Dataset | `train.py` → swap `load_wine` for real data loading |
| API port | `app.py` → `PORT` env var, or default `5000` |
| Python version | `pipeline.yml` → `python-version` |
| Trigger branches | `pipeline.yml` → `on.push.branches` |

---

## Production Notes

- For production, serve with **Gunicorn**: `gunicorn -w 4 app:app`
- Store `model.pkl` in cloud storage (S3, GCS) for multi-instance deployments
- Add authentication middleware before exposing the API publicly
- Consider versioning your models with MLflow or DVC for experiment tracking
- this is the new line
