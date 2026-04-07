# ML CI/CD Pipeline

A complete end-to-end Machine Learning CI/CD pipeline using Python, scikit-learn, Flask, and GitHub Actions.

---

## Project Structure

```
.
├── train.py                        # Trains and saves the model
├── evaluate.py                     # Evaluates model; fails pipeline if accuracy < 0.80
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
📦 Generating synthetic classification dataset...
   Dataset shape: X=(1000, 20), y=(1000,)
✂️  Train size: 800 | Test size: 200
🌲 Training RandomForestClassifier...
📊 Train accuracy : 1.0000
📊 Test  accuracy : 0.9250
✅ Model saved to model/model.pkl
```

### 3 — Evaluate the model

```bash
python evaluate.py
```

The script exits with code `0` (pass) if accuracy ≥ 0.80, or code `1` (fail) otherwise.

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
  "classes": [0, 1],
  "max_depth": 10,
  "model_type": "RandomForestClassifier",
  "n_classes": 2,
  "n_estimators": 100,
  "n_features": 20
}
```

### `POST /predict` — Single prediction

Send a JSON body with a `features` array of exactly 20 floats:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.3, 0.8, -0.5, 1.1, -0.2, 0.6, 0.9, -0.4,
                    0.7,  0.2, -0.8, 1.3, 0.1, -0.6, 0.4, -0.3, 0.5, 0.0]}'
```
```json
{
  "label": "Class 1",
  "prediction": 1,
  "probabilities": {
    "Class 0": 0.17,
    "Class 1": 0.83
  }
}
```

### `POST /predict/batch` — Batch predictions

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      [0.5, -1.2, 0.3, 0.8, -0.5, 1.1, -0.2, 0.6, 0.9, -0.4,
       0.7,  0.2, -0.8, 1.3, 0.1, -0.6, 0.4, -0.3, 0.5, 0.0],
      [0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0,  0.0,
       0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0,  0.0]
    ]
  }'
```
```json
{
  "count": 2,
  "predictions": [
    {"label": "Class 1", "prediction": 1, "probabilities": {"Class 0": 0.17, "Class 1": 0.83}},
    {"label": "Class 0", "prediction": 0, "probabilities": {"Class 0": 0.62, "Class 1": 0.38}}
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
┌─────────────────────────┐    accuracy < 0.80 → ❌ PIPELINE FAILS
│  5. python evaluate.py  │
└────────────┬────────────┘    accuracy ≥ 0.80 → ✅ continue
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
| Model algorithm | `train.py` → replace `RandomForestClassifier` |
| Dataset | `train.py` → swap `make_classification` for real data loading |
| API port | `app.py` → `PORT` env var, or default `5000` |
| Python version | `pipeline.yml` → `python-version` |
| Trigger branches | `pipeline.yml` → `on.push.branches` |

---

## Production Notes

- For production, serve with **Gunicorn**: `gunicorn -w 4 app:app`
- Store `model.pkl` in cloud storage (S3, GCS) for multi-instance deployments
- Add authentication middleware before exposing the API publicly
- Consider versioning your models with MLflow or DVC for experiment tracking
