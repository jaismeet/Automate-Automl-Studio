# Automate AutoML Application

This is a simple AutoML web application.

## Flow

```text
User uploads CSV or connects SQLite database
-> System profiles columns
-> User chooses dashboard or model generation
-> Dashboard suggests KPIs, charts, filters, summaries
-> Model training ranks AutoML models with cross-validation
-> System shows metrics, validation diagnostics, explainability, and prediction API
```

## Run

```bash
cd C:\Users\Lenovo\Documents\jaismeet\automate
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Main API Routes

```text
POST /api/upload-csv
POST /api/connect-sqlite
GET  /api/datasets
GET  /api/datasets/{dataset_id}/profile
GET  /api/datasets/{dataset_id}/quality
POST /api/datasets/{dataset_id}/cleaning-plan
POST /api/datasets/{dataset_id}/clean
GET  /api/datasets/{dataset_id}/preview
GET  /api/sample-datasets
POST /api/sample-datasets/{sample_id}/load
GET  /api/dashboard/{dataset_id}
POST /api/model-suggestions
POST /api/train
POST /api/accuracy-booster
GET  /api/models
GET  /api/experiments
GET  /api/experiments/{model_id}
GET  /api/prediction-audits
GET  /api/drift-reports
GET  /api/models/export
GET  /api/experiments/export
GET  /api/models/{model_id}/trust
GET  /api/models/{model_id}/readiness
GET  /api/models/{model_id}/monitoring
GET  /api/models/{model_id}/monitoring/history
GET  /api/monitoring-alerts
GET  /api/models/{model_id}/challengers
POST /api/models/{model_id}/promote
GET  /api/models/{model_id}/api-docs
GET  /api/models/{model_id}/drift/latest
POST /api/models/{model_id}/drift
POST /api/models/{model_id}/retrain-from-drift
GET  /api/models/{model_id}/download
GET  /api/models/{model_id}/leaderboard/export
GET  /api/models/{model_id}/explainability/export
GET  /api/models/{model_id}/report
GET  /api/models/{model_id}/playground
POST /api/predict/{model_id}
POST /api/predict/{model_id}/batch
```

## Notes

This first version is intentionally learning-friendly:

- CSV upload is fully supported.
- Sample CSVs can be loaded directly from the UI.
- SQLite table loading is supported.
- Dataset Health Score grades ML readiness from 0-100 with risk level, blocker count, warning count, health checks, and fix-before-training actions.
- The Data Cleaning Assistant flags duplicates, missing values, constants, identifier-like columns, high-cardinality text, and numeric outliers.
- One-click cleaning can create a cleaned dataset copy by removing duplicates, dropping noisy columns, filling missing feature values, capping outliers, and grouping rare categories.
- AutoML trains expanded classifier/regressor candidate sets, including ensembles.
- AutoML now trains imbalance-aware classifiers, including balanced-subsample and oversampled candidates.
- Training results include an Accuracy Improver panel with baseline comparison, class-balance checks, weakest-class recall, and next fixes.
- Auto Accuracy Booster can run a cleaned boosted training attempt, compare it against the latest saved run, and show the score delta.
- Hyperparameter Tuning Studio tunes the top trained models, compares base vs tuned score, and saves the tuned winner when it improves the ranking metric.
- Data Leakage Detection warns about target-leaking columns and auto-excludes them from model training.
- AutoML now includes tuned tree and boosting variants for stronger accuracy search.
- AutoML drops identifier-like noise columns and engineers date features before training.
- Model ranking uses cross-validation when the dataset has enough rows.
- The leaderboard also shows baseline metrics, so you can tell whether the model is actually useful.
- The leaderboard shows model quality labels, failed-model reasons, and training insights.
- The training result now includes recommended next steps when accuracy or recall is weak.
- Binary classification training now includes threshold tuning for precision/recall control.
- Model Studio lets you include or exclude feature columns before suggesting/training models.
- Classification models show confusion matrix and per-class precision/recall/F1 diagnostics.
- Prediction Confidence Dashboard shows average/minimum confidence, high/medium/low confidence counts, class probabilities, and low-confidence warnings.
- Prediction What-If Simulator builds editable feature controls from the selected saved model and runs live API playground predictions.
- Prediction Audit Log saves recent single-row and batch prediction requests with model, inputs, outputs, confidence, and timestamp.
- Model Risk & Trust Score grades every saved model from 0-100 with low/medium/high risk, blockers, warnings, checks, and deployment guidance.
- Deployment Readiness Checklist combines trust score, model artifact, API status, audit history, confidence, batch scoring, explainability, report readiness, and drift review into one release decision.
- Dataset Drift Monitor compares a model's training data with a selected scoring dataset, saves the drift report, and updates deployment readiness from the latest drift result.
- Model Monitoring Dashboard combines readiness, trust, drift, prediction confidence, and audit coverage into one production health score with a recommended next action.
- Monitoring History and Alert Center saves health snapshots, renders score trends, and raises active alerts for drift, readiness, confidence, audit, and champion issues.
- Champion/Challenger compares saved models with the same target and metric, marks the current champion, and supports one-click retraining from a drift comparison dataset.
- Manual production promotion lets you choose the live model per target/problem group, and the API Docs Generator creates payloads, response examples, curl, and Python snippets for the selected model.
- Classification predictions include confidence/probability details when the trained model supports it.
- Single-row and batch classification predictions can use the selected decision threshold.
- Batch prediction accepts a CSV and downloads a scored CSV with prediction, confidence band, low-confidence flag, and probability columns.
- Regression models show residual summaries and actual-vs-predicted samples.
- Auto Report Generator exports a six-slide PPTX with dataset readiness, model leaderboard, accuracy notes, explainability, deployment details, and future actions.
- Regression diagnostics now highlight the largest residual errors and MAPE when available.
- Advanced Explainability Studio uses permutation importance to show top drivers, signal strength, weak/noisy features, and explanation actions.
- Saved model registry records now keep quality, trained/failed counts, feature cleanup counts, leaderboard snapshots, and next actions.
- Evaluate includes an experiment history panel with filters, previous-run deltas, best-run highlights, and saved run review details.
- Model registry reports can be exported as CSV.
- Export Center downloads trained model artifacts, experiment history CSVs, leaderboard snapshots, and explainability summaries.
- Dashboard suggestions include bar, pie, histogram, scatter, line, and heatmap chart types.
- Dashboard filters are generated from the active dataset and refresh charts/KPIs interactively.
- Saved models are stored in `models/`.
- Uploaded datasets are stored in `data/`.
- Dataset and model registry metadata are stored in `registry/`.

## Sample Data

Use these files to test the app:

```text
sample_data/customers.csv
sample_data/customer_churn.csv
sample_data/loan_default.csv
sample_data/retail_sales_forecast.csv
```

Recommended model targets:

```text
customers.csv -> churn
customer_churn.csv -> churn
loan_default.csv -> defaulted
retail_sales_forecast.csv -> next_month_sales
```
