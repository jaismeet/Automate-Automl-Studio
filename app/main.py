import json
import io
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api_docs import build_model_api_docs
from app.automl import predict_rows_with_details, suggest_model_candidates, train_models
from app.champion import attach_champion_signals, build_champion_challenger
from app.database import load_sqlite_table
from app.dashboard import suggest_dashboard
from app.drift import build_dataset_drift_report
from app.monitoring import build_model_monitoring, build_monitoring_alerts, build_monitoring_snapshot
from app.profiler import profile_dataframe
from app.quality import analyze_data_quality, clean_dataframe
from app.readiness import build_deployment_readiness
from app.reporting import build_model_report_pptx
from app.samples import list_sample_datasets, load_sample_dataframe
from app.storage import (
    DATA_DIR,
    MODEL_DIR,
    get_model_record,
    get_latest_drift_report,
    list_datasets,
    list_drift_reports,
    list_experiment_history,
    list_model_promotions,
    list_models,
    list_monitoring_alerts,
    list_monitoring_snapshots,
    list_prediction_audits,
    load_dataframe,
    preview_dataframe,
    register_drift_report,
    register_model_promotion,
    register_monitoring_snapshot,
    register_prediction_audit,
    save_dataframe,
    sync_monitoring_alerts,
    utc_now,
    model_group_key,
)
from app.trust import attach_model_trust_score, build_model_trust_score


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Automate AutoML Application",
    description="CSV/database profiling, dashboard suggestions, AutoML model training, explainability, and prediction API.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: str = "auto"
    feature_columns: list[str] | None = None


class ModelSuggestionRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: str = "auto"
    feature_columns: list[str] | None = None


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]
    threshold: float | None = None


class SQLiteRequest(BaseModel):
    db_path: str
    table_name: str


class DriftRequest(BaseModel):
    dataset_id: str


class RetrainFromDriftRequest(BaseModel):
    dataset_id: str | None = None


class CleanDatasetRequest(BaseModel):
    target_column: str | None = None
    options: dict[str, bool] | None = None


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.get("/api/sample-datasets")
def sample_datasets():
    return {"samples": list_sample_datasets()}


@app.post("/api/sample-datasets/{sample_id}/load")
def load_sample_dataset(sample_id: str):
    df, sample = load_sample_dataframe(sample_id)
    dataset_id = save_dataframe(df, sample["filename"])
    profile = profile_dataframe(df)
    quality = analyze_data_quality(df)

    return {
        "dataset_id": dataset_id,
        "filename": sample["filename"],
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "profile": profile,
        "quality": quality,
        "sample": sample,
        "suggested_target": {
            "target_column": sample.get("target_column"),
            "problem_type": sample.get("problem_type"),
        },
    }


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / file.filename
    raw_path.write_bytes(await file.read())

    try:
        df = pd.read_csv(raw_path)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Could not read CSV file.") from error

    dataset_id = save_dataframe(df, file.filename)
    profile = profile_dataframe(df)
    quality = analyze_data_quality(df)

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "profile": profile,
        "quality": quality,
    }


@app.post("/api/connect-sqlite")
def connect_sqlite(request: SQLiteRequest):
    df = load_sqlite_table(request.db_path, request.table_name)
    dataset_id = save_dataframe(df, f"{request.table_name}.csv")

    return {
        "dataset_id": dataset_id,
        "source": "sqlite",
        "table": request.table_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "profile": profile_dataframe(df),
        "quality": analyze_data_quality(df),
    }


def filter_dashboard_dataframe(df: pd.DataFrame, raw_filters: str | None) -> tuple[pd.DataFrame, dict]:
    if not raw_filters:
        return df, {}

    try:
        requested_filters = json.loads(raw_filters)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="Dashboard filters must be valid JSON.") from error

    if not isinstance(requested_filters, dict):
        raise HTTPException(status_code=400, detail="Dashboard filters must be a JSON object.")

    active_filters = {
        column: value
        for column, value in requested_filters.items()
        if column in df.columns and value is not None and str(value) not in {"", "__all__"}
    }

    filtered_df = df.copy()
    for column, value in active_filters.items():
        filter_value = str(value)
        if len(filter_value) == 7 and filter_value[4] == "-":
            parsed_dates = pd.to_datetime(filtered_df[column], errors="coerce", format="%Y-%m-%d")
            filtered_df = filtered_df[parsed_dates.dt.strftime("%Y-%m") == str(value)]
        else:
            filtered_df = filtered_df[filtered_df[column].astype(str) == filter_value]

    return filtered_df, active_filters


@app.get("/api/dashboard/{dataset_id}")
def dashboard(dataset_id: str, filters: str | None = None):
    df = load_dataframe(dataset_id)
    filtered_df, active_filters = filter_dashboard_dataframe(df, filters)
    return suggest_dashboard(filtered_df, source_df=df, active_filters=active_filters)


@app.get("/api/datasets")
def datasets():
    return {"datasets": list_datasets()}


@app.get("/api/datasets/{dataset_id}/preview")
def dataset_preview(dataset_id: str, limit: int = 20):
    return preview_dataframe(dataset_id, limit)


@app.get("/api/datasets/{dataset_id}/profile")
def dataset_profile(dataset_id: str):
    df = load_dataframe(dataset_id)
    return {
        "dataset_id": dataset_id,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "profile": profile_dataframe(df),
        "quality": analyze_data_quality(df),
    }


@app.get("/api/datasets/{dataset_id}/quality")
def dataset_quality(dataset_id: str):
    df = load_dataframe(dataset_id)
    return {
        "dataset_id": dataset_id,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "quality": analyze_data_quality(df),
    }


def serialize_cleaning_result(dataset_id: str, result: dict, target_column: str | None = None) -> dict:
    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "summary": result["summary"],
        "steps": result["steps"],
        "before_quality": result["before_quality"],
        "after_quality": result["after_quality"],
        "apply_available": result["apply_available"],
        "options": result["options"],
    }


@app.post("/api/datasets/{dataset_id}/cleaning-plan")
def dataset_cleaning_plan(dataset_id: str, request: CleanDatasetRequest):
    df = load_dataframe(dataset_id)
    result = clean_dataframe(df, request.target_column, request.options)
    return serialize_cleaning_result(dataset_id, result, request.target_column)


@app.post("/api/datasets/{dataset_id}/clean")
def clean_dataset(dataset_id: str, request: CleanDatasetRequest):
    df = load_dataframe(dataset_id)
    result = clean_dataframe(df, request.target_column, request.options)
    if not result["apply_available"]:
        raise HTTPException(status_code=400, detail="No automatic cleaning actions are available for this dataset.")

    cleaned_df = result["cleaned_df"]
    source_record = next((item for item in list_datasets() if item.get("dataset_id") == dataset_id), None)
    source_name = source_record.get("name", "dataset.csv") if source_record else "dataset.csv"
    cleaned_name = f"cleaned_{Path(source_name).stem}.csv"
    cleaned_dataset_id = save_dataframe(cleaned_df, cleaned_name)
    target = request.target_column if request.target_column in cleaned_df.columns else None

    return {
        "dataset_id": cleaned_dataset_id,
        "source_dataset_id": dataset_id,
        "filename": cleaned_name,
        "source": "cleaned",
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "profile": profile_dataframe(cleaned_df),
        "quality": result["after_quality"],
        "cleaning_summary": result["summary"],
        "cleaning_steps": result["steps"],
        "suggested_target": {
            "target_column": target,
            "problem_type": "auto",
        },
    }


@app.get("/api/models")
def models():
    trusted_models = [attach_model_trust_score(model) for model in list_models()]
    return {"models": attach_champion_signals(trusted_models, list_model_promotions())}


@app.get("/api/experiments")
def experiments(dataset_id: str | None = None, target_column: str | None = None, limit: int = 20):
    return list_experiment_history(dataset_id=dataset_id, target_column=target_column, limit=limit)


@app.get("/api/models/export")
def export_models():
    models = attach_champion_signals([attach_model_trust_score(model) for model in list_models()], list_model_promotions())
    export_rows = []
    for model in models:
        trust = model.get("trust_score") or {}
        export_rows.append(
            {
                "model_id": model.get("model_id"),
                "dataset_id": model.get("dataset_id"),
                "target_column": model.get("target_column"),
                "problem_type": model.get("problem_type"),
                "best_model": model.get("best_model"),
                "quality_label": model.get("quality_label"),
                "primary_metric": model.get("primary_metric"),
                "rank_score": model.get("rank_score"),
                "candidate_models": model.get("candidate_models"),
                "trained_models": model.get("trained_models"),
                "failed_models": model.get("failed_models"),
                "raw_feature_count": model.get("raw_feature_count"),
                "model_feature_count": model.get("model_feature_count"),
                "selected_feature_count": model.get("selected_feature_count"),
                "excluded_feature_count": model.get("excluded_feature_count"),
                "dropped_feature_count": model.get("dropped_feature_count"),
                "engineered_feature_count": model.get("engineered_feature_count"),
                "recommended_threshold": model.get("recommended_threshold"),
                "trust_score": trust.get("score"),
                "risk_level": trust.get("risk_level"),
                "trust_decision": trust.get("decision"),
                "trust_blockers": trust.get("blockers"),
                "trust_warnings": trust.get("warnings"),
                "champion_status": (model.get("champion") or {}).get("status"),
                "production_status": (model.get("production") or {}).get("status"),
                "production_model_id": (model.get("production") or {}).get("production_model_id"),
                "first_recommendation": (model.get("next_actions") or [""])[0],
                "prediction_api": model.get("prediction_api"),
                "created_at": model.get("created_at"),
            }
        )

    output = io.StringIO()
    pd.DataFrame(export_rows).to_csv(output, index=False)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="automate-model-report.csv"'},
    )


@app.get("/api/experiments/export")
def export_experiments(dataset_id: str | None = None, target_column: str | None = None, limit: int = 100):
    history = list_experiment_history(dataset_id=dataset_id, target_column=target_column, limit=limit)
    export_rows = []
    for run in history.get("experiments", []):
        export_rows.append(
            {
                "experiment_id": run.get("experiment_id"),
                "model_id": run.get("model_id"),
                "dataset_id": run.get("dataset_id"),
                "target_column": run.get("target_column"),
                "problem_type": run.get("problem_type"),
                "best_model": run.get("best_model"),
                "quality_label": run.get("quality_label"),
                "primary_metric": run.get("primary_metric"),
                "rank_score": run.get("rank_score"),
                "previous_score": run.get("previous_score"),
                "score_delta": run.get("score_delta"),
                "is_latest": run.get("is_latest"),
                "is_best_for_group": run.get("is_best_for_group"),
                "group_run_number": run.get("group_run_number"),
                "candidate_models": run.get("candidate_models"),
                "trained_models": run.get("trained_models"),
                "failed_models": run.get("failed_models"),
                "model_feature_count": run.get("model_feature_count"),
                "raw_feature_count": run.get("raw_feature_count"),
                "first_recommendation": (run.get("next_actions") or [""])[0],
                "prediction_api": run.get("prediction_api"),
                "created_at": run.get("created_at"),
            }
        )

    output = io.StringIO()
    pd.DataFrame(export_rows).to_csv(output, index=False)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="automate-experiment-history.csv"'},
    )


@app.get("/api/experiments/{model_id}")
def experiment_detail(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {"experiment": record}


@app.get("/api/models/{model_id}/trust")
def model_trust(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    return {
        "model_id": model_id,
        "trust_score": build_model_trust_score(record),
    }


@app.get("/api/models/{model_id}/readiness")
def model_readiness(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    audits = list_prediction_audits(model_id=model_id, limit=50)
    drift = get_latest_drift_report(model_id)
    return {
        "model_id": model_id,
        "readiness": build_deployment_readiness(record, audits, drift),
    }


@app.get("/api/models/{model_id}/monitoring")
def model_monitoring(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    audits = list_prediction_audits(model_id=model_id, limit=50)
    drift = get_latest_drift_report(model_id)
    readiness = build_deployment_readiness(record, audits, drift)
    models = list_models()
    promotions = list_model_promotions()
    comparison = build_champion_challenger(record, models, promotions)
    monitoring = build_model_monitoring(record, audits, drift, readiness)
    snapshot = build_monitoring_snapshot(monitoring, utc_now())
    register_monitoring_snapshot(snapshot)
    alerts = build_monitoring_alerts(monitoring, comparison)
    alert_state = sync_monitoring_alerts(model_id, alerts)
    return {
        "model_id": model_id,
        "monitoring": monitoring,
        "history": list_monitoring_snapshots(model_id=model_id, limit=40),
        "alerts": alert_state,
        "champion_challenger": comparison,
    }


@app.get("/api/models/{model_id}/monitoring/history")
def model_monitoring_history(model_id: str, limit: int = 40):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    return list_monitoring_snapshots(model_id=model_id, limit=limit)


@app.get("/api/monitoring-alerts")
def monitoring_alerts(model_id: str | None = None, limit: int = 30, include_resolved: bool = False):
    return list_monitoring_alerts(model_id=model_id, limit=limit, include_resolved=include_resolved)


def _load_model_feature_columns(record: dict) -> list[str]:
    model_id = record.get("model_id")
    training_summary = record.get("training_summary") or {}
    fallback = training_summary.get("feature_columns") or []
    if not model_id:
        return fallback

    model_path = MODEL_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        return fallback

    try:
        artifact = joblib.load(model_path)
    except Exception:
        return fallback
    return artifact.get("feature_columns") or fallback


@app.get("/api/models/{model_id}/challengers")
def model_challengers(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    return {
        "model_id": model_id,
        "champion_challenger": build_champion_challenger(record, list_models(), list_model_promotions()),
    }


@app.post("/api/models/{model_id}/promote")
def promote_model(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    promotion = register_model_promotion(
        {
            "group_key": model_group_key(record),
            "model_id": model_id,
            "best_model": record.get("best_model"),
            "target_column": record.get("target_column"),
            "problem_type": record.get("problem_type"),
            "primary_metric": record.get("primary_metric"),
            "rank_score": record.get("rank_score"),
            "prediction_api": record.get("prediction_api"),
            "model_created_at": record.get("created_at"),
        }
    )
    return {
        "model_id": model_id,
        "promotion": promotion,
        "champion_challenger": build_champion_challenger(record, list_models(), list_model_promotions()),
    }


@app.post("/api/models/{model_id}/retrain-from-drift")
def retrain_from_drift(model_id: str, request: RetrainFromDriftRequest):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    latest_drift = get_latest_drift_report(model_id)
    dataset_id = request.dataset_id or (latest_drift or {}).get("comparison_dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Run a drift check or choose a comparison dataset before retraining.")

    datasets = list_datasets()
    dataset = next((item for item in datasets if item.get("dataset_id") == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Retraining dataset not found.")

    target_column = record.get("target_column")
    if not target_column:
        raise HTTPException(status_code=400, detail="The source model does not have a saved target column.")

    df = load_dataframe(dataset_id)
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="This comparison dataset does not include the target column, so it cannot be used for retraining. Use a labeled dataset.",
        )

    source_features = [
        column
        for column in _load_model_feature_columns(record)
        if column in df.columns and column != target_column
    ]
    feature_columns = source_features or None
    run_context = {
        "mode": "retrain_from_drift",
        "source_model_id": model_id,
        "source_dataset_id": record.get("dataset_id"),
        "comparison_dataset_id": dataset_id,
        "comparison_dataset_name": dataset.get("name"),
        "drift_id": (latest_drift or {}).get("drift_id"),
        "reused_feature_count": len(source_features),
        "feature_strategy": "source_model_features" if source_features else "auto_features",
    }

    result = train_models(
        df,
        target_column,
        record.get("problem_type") or "auto",
        dataset_id,
        feature_columns,
        run_context=run_context,
    )
    new_record = get_model_record(result["model_id"])
    comparison = build_champion_challenger(new_record or record, list_models(), list_model_promotions())
    return {
        "source_model_id": model_id,
        "new_model_id": result["model_id"],
        "dataset_id": dataset_id,
        "dataset_name": dataset.get("name"),
        "training_result": result,
        "champion_challenger": comparison,
    }


@app.get("/api/drift-reports")
def drift_reports(model_id: str | None = None, limit: int = 20):
    return list_drift_reports(model_id=model_id, limit=limit)


@app.get("/api/models/{model_id}/drift/latest")
def latest_model_drift(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    return {
        "model_id": model_id,
        "drift": get_latest_drift_report(model_id),
    }


@app.post("/api/models/{model_id}/drift")
def model_drift(model_id: str, request: DriftRequest):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    reference_dataset_id = record.get("dataset_id")
    if not reference_dataset_id:
        raise HTTPException(status_code=400, detail="This model is missing its training dataset reference.")

    datasets = list_datasets()
    reference_dataset = next((item for item in datasets if item.get("dataset_id") == reference_dataset_id), None)
    if not reference_dataset:
        raise HTTPException(status_code=404, detail="Training dataset for this model was not found.")

    comparison_dataset = next((item for item in datasets if item.get("dataset_id") == request.dataset_id), None)
    if not comparison_dataset:
        raise HTTPException(status_code=404, detail="Comparison dataset not found.")

    reference_df = load_dataframe(reference_dataset_id)
    comparison_df = load_dataframe(request.dataset_id)
    record_for_drift = {
        **record,
        "dataset_name": reference_dataset.get("name") or reference_dataset_id,
    }
    report = build_dataset_drift_report(
        record_for_drift,
        reference_df,
        comparison_df,
        comparison_dataset,
        feature_columns=_load_model_feature_columns(record_for_drift),
    )
    report["drift_id"] = uuid4().hex
    report["created_at"] = utc_now()
    register_drift_report(report)

    return {
        "model_id": model_id,
        "drift": report,
    }


@app.get("/api/models/{model_id}/download")
def download_model(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    model_path = MODEL_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found.")

    safe_model = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(record.get("best_model") or "model")).strip("_")
    download_name = f"{safe_model or 'model'}-{model_id}.joblib"
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=download_name,
    )


@app.get("/api/models/{model_id}/leaderboard/export")
def export_model_leaderboard(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    rows = []
    for index, item in enumerate(record.get("leaderboard_snapshot") or [], start=1):
        metrics = item.get("metrics") or {}
        cross_validation = item.get("cross_validation") or {}
        rows.append(
            {
                "rank": index,
                "model_id": model_id,
                "model_name": item.get("model_name"),
                "status": item.get("status"),
                "quality_label": item.get("quality_label"),
                "rank_score": item.get("rank_score"),
                "cv_available": cross_validation.get("available"),
                "cv_metric": cross_validation.get("metric"),
                "cv_mean": cross_validation.get("mean"),
                "cv_std": cross_validation.get("std"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "f1_weighted": metrics.get("f1_weighted"),
                "r2": metrics.get("r2"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "tuned_from": item.get("tuned_from"),
                "error": item.get("error"),
            }
        )
    if not rows:
        metrics = record.get("holdout_metrics") or {}
        rows.append(
            {
                "rank": 1,
                "model_id": model_id,
                "model_name": record.get("best_model"),
                "status": "saved",
                "quality_label": record.get("quality_label"),
                "rank_score": record.get("rank_score"),
                "cv_available": None,
                "cv_metric": record.get("primary_metric"),
                "cv_mean": None,
                "cv_std": None,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "f1_weighted": metrics.get("f1_weighted"),
                "r2": metrics.get("r2"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "tuned_from": None,
                "error": None,
            }
        )

    output = io.StringIO()
    pd.DataFrame(rows).to_csv(output, index=False)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="leaderboard-{model_id}.csv"'},
    )


@app.get("/api/models/{model_id}/explainability/export")
def export_model_explainability(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    studio = record.get("explainability_studio") or {}
    features = studio.get("features") or record.get("top_features") or []
    rows = []
    for index, item in enumerate(features, start=1):
        rows.append(
            {
                "rank": index,
                "model_id": model_id,
                "best_model": record.get("best_model"),
                "target_column": record.get("target_column"),
                "feature": item.get("feature"),
                "importance": item.get("importance"),
                "impact": item.get("impact"),
                "strength": item.get("strength"),
                "share": item.get("share"),
                "interpretation": item.get("interpretation"),
                "method": studio.get("method") or "permutation_importance",
                "summary": studio.get("summary"),
            }
        )

    output = io.StringIO()
    pd.DataFrame(rows).to_csv(output, index=False)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="explainability-{model_id}.csv"'},
    )


@app.get("/api/models/{model_id}/report")
def export_model_report(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    dataset = next((item for item in list_datasets() if item.get("dataset_id") == record.get("dataset_id")), None)
    history = list_experiment_history(
        dataset_id=record.get("dataset_id"),
        target_column=record.get("target_column"),
        limit=20,
    )
    report = build_model_report_pptx(record, dataset=dataset, history=history)
    safe_model = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(record.get("best_model") or "model")
    ).strip("_")
    filename = f"automl-report-{safe_model or 'model'}-{model_id}.pptx"
    return StreamingResponse(
        iter([report]),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _json_safe_cell(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _fallback_playground_value(field: dict) -> Any:
    if field.get("type") == "numeric":
        return field.get("mean") if field.get("mean") is not None else 0
    top_values = field.get("top_values") or []
    if top_values:
        return top_values[0]
    return ""


def _audit_preview_value(value: Any) -> Any:
    value = _json_safe_cell(value)
    if isinstance(value, str) and len(value) > 120:
        return value[:117] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _audit_preview_rows(rows: list[dict], limit: int = 5) -> list[dict]:
    preview = []
    for row in rows[:limit]:
        preview.append({str(key): _audit_preview_value(value) for key, value in row.items()})
    return preview


def _audit_prediction_details(details: list[dict], limit: int = 10) -> list[dict]:
    preview = []
    for item in details[:limit]:
        preview.append(
            {
                "row": item.get("row"),
                "prediction": item.get("prediction"),
                "confidence": item.get("confidence"),
                "confidence_band": item.get("confidence_band"),
                "low_confidence": item.get("low_confidence"),
                "probabilities": item.get("probabilities") or {},
            }
        )
    return preview


def build_prediction_audit(
    model_id: str,
    record: dict | None,
    rows: list[dict],
    prediction_result: dict,
    request_type: str = "single",
    source_name: str | None = None,
) -> dict:
    summary = prediction_result.get("confidence_summary") or {}
    details = prediction_result.get("prediction_details") or []
    return {
        "audit_id": uuid4().hex,
        "created_at": utc_now(),
        "request_type": request_type,
        "source_name": source_name,
        "model_id": model_id,
        "dataset_id": (record or {}).get("dataset_id"),
        "best_model": (record or {}).get("best_model"),
        "target_column": (record or {}).get("target_column"),
        "problem_type": (record or {}).get("problem_type"),
        "primary_metric": (record or {}).get("primary_metric"),
        "rank_score": (record or {}).get("rank_score"),
        "row_count": len(rows),
        "threshold": prediction_result.get("threshold"),
        "input_preview": _audit_preview_rows(rows),
        "predictions": prediction_result.get("predictions", [])[:25],
        "prediction_details": _audit_prediction_details(details),
        "confidence_summary": {
            "available": summary.get("available"),
            "average_confidence": summary.get("average_confidence"),
            "minimum_confidence": summary.get("minimum_confidence"),
            "high_confidence_rows": summary.get("high_confidence_rows"),
            "medium_confidence_rows": summary.get("medium_confidence_rows"),
            "low_confidence_rows": summary.get("low_confidence_rows"),
            "prediction_distribution": summary.get("prediction_distribution") or [],
            "recommendation": summary.get("recommendation") or summary.get("message"),
        },
    }


def save_prediction_audit(
    model_id: str,
    rows: list[dict],
    prediction_result: dict,
    request_type: str = "single",
    source_name: str | None = None,
) -> dict:
    record = get_model_record(model_id)
    audit = build_prediction_audit(model_id, record, rows, prediction_result, request_type, source_name)
    register_prediction_audit(audit)
    return audit


@app.get("/api/prediction-audits")
def prediction_audits(model_id: str | None = None, limit: int = 30):
    return list_prediction_audits(model_id=model_id, limit=limit)


@app.get("/api/models/{model_id}/playground")
def model_playground(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")

    model_path = MODEL_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found.")

    try:
        artifact = joblib.load(model_path)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="This saved model artifact cannot be loaded in the current environment. Retrain it to use the playground.",
        ) from error
    feature_columns = artifact.get("feature_columns") or (record.get("training_summary") or {}).get("feature_columns") or []
    model_feature_columns = artifact.get("model_feature_columns") or (record.get("training_summary") or {}).get("model_feature_columns") or []

    dataset = next((item for item in list_datasets() if item.get("dataset_id") == record.get("dataset_id")), None)
    profile_columns = {}
    sample_rows = []
    fields = []

    try:
        df = load_dataframe(record.get("dataset_id"))
        profile = profile_dataframe(df)
        profile_columns = {column["name"]: column for column in profile.get("columns", [])}
        sample_frame = df.reindex(columns=feature_columns).head(5)
        sample_rows = [
            {column: _json_safe_cell(value) for column, value in row.items()}
            for row in sample_frame.astype(object).where(pd.notna(sample_frame), None).to_dict(orient="records")
        ]
    except Exception:
        profile_columns = {}
        sample_rows = []

    for column in feature_columns:
        profile = profile_columns.get(column, {})
        field = {
            "name": column,
            "type": profile.get("type", "unknown"),
            "missing": profile.get("missing"),
            "missing_percent": profile.get("missing_percent"),
            "unique": profile.get("unique"),
            "mean": profile.get("mean"),
            "min": profile.get("min"),
            "max": profile.get("max"),
            "top_values": list((profile.get("top_values") or {}).keys())[:8],
        }

        example = None
        if sample_rows:
            example = next((_json_safe_cell(row.get(column)) for row in sample_rows if row.get(column) is not None), None)
        if example is None:
            example = _fallback_playground_value(field)
        field["example"] = example
        fields.append(field)

    default_row = {
        field["name"]: _json_safe_cell(field.get("example"))
        for field in fields
    }
    request_template = {"rows": [default_row]}
    if record.get("recommended_threshold") is not None:
        request_template["threshold"] = record.get("recommended_threshold")

    return {
        "model_id": model_id,
        "dataset_id": record.get("dataset_id"),
        "dataset_name": dataset.get("name") if dataset else record.get("dataset_id"),
        "target_column": artifact.get("target_column") or record.get("target_column"),
        "problem_type": artifact.get("problem_type") or record.get("problem_type"),
        "best_model": artifact.get("model_name") or record.get("best_model"),
        "primary_metric": record.get("primary_metric"),
        "rank_score": record.get("rank_score"),
        "quality_label": record.get("quality_label"),
        "prediction_api": record.get("prediction_api") or f"/api/predict/{model_id}",
        "recommended_threshold": record.get("recommended_threshold"),
        "feature_columns": feature_columns,
        "model_feature_columns": model_feature_columns,
        "fields": fields,
        "sample_rows": sample_rows,
        "request_template": request_template,
        "next_actions": record.get("next_actions") or [],
    }


@app.get("/api/models/{model_id}/api-docs")
def model_api_docs(model_id: str):
    record = get_model_record(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    playground = model_playground(model_id)
    promotion = next(
        (
            item
            for item in list_model_promotions()
            if item.get("group_key") == model_group_key(record)
        ),
        None,
    )
    return {
        "model_id": model_id,
        "api_docs": build_model_api_docs(record, playground, promotion),
    }


@app.post("/api/train")
def train(request: TrainRequest):
    df = load_dataframe(request.dataset_id)
    result = train_models(df, request.target_column, request.problem_type, request.dataset_id, request.feature_columns)
    return result


def _score_from_run(run: dict | None) -> float | None:
    if not run:
        return None
    score = run.get("rank_score")
    return float(score) if isinstance(score, (int, float)) else None


def _score_from_training_result(result: dict) -> float | None:
    leaderboard = result.get("leaderboard") or []
    if not leaderboard:
        return None
    score = leaderboard[0].get("rank_score")
    return float(score) if isinstance(score, (int, float)) else None


@app.post("/api/accuracy-booster")
def accuracy_booster(request: TrainRequest):
    df = load_dataframe(request.dataset_id)
    if request.target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found.")

    previous_history = list_experiment_history(
        dataset_id=request.dataset_id,
        target_column=request.target_column,
        limit=20,
    )
    previous_run = previous_history.get("summary", {}).get("latest_run")
    previous_score = _score_from_run(previous_run)

    cleaning_result = clean_dataframe(df, request.target_column)
    cleaned_df = cleaning_result["cleaned_df"] if cleaning_result["apply_available"] else df
    if request.target_column not in cleaned_df.columns:
        raise HTTPException(status_code=400, detail="Cleaning removed the selected target column.")

    requested_features = request.feature_columns or [column for column in df.columns if column != request.target_column]
    boosted_features = [
        column
        for column in requested_features
        if column in cleaned_df.columns and column != request.target_column
    ]
    if not boosted_features:
        boosted_features = [column for column in cleaned_df.columns if column != request.target_column]
    if not boosted_features:
        raise HTTPException(status_code=400, detail="No usable feature columns remain after cleaning.")

    changed_steps = [step for step in cleaning_result["steps"] if step.get("changed")]
    run_context = {
        "mode": "accuracy_booster",
        "cleaning_applied": bool(changed_steps),
        "cleaning_steps": [step["id"] for step in changed_steps],
        "feature_columns_after_cleaning": len(boosted_features),
    }
    result = train_models(
        cleaned_df,
        request.target_column,
        request.problem_type,
        request.dataset_id,
        boosted_features,
        run_context=run_context,
    )

    boosted_score = _score_from_training_result(result)
    delta = round(float(boosted_score - previous_score), 4) if boosted_score is not None and previous_score is not None else None
    if delta is None:
        status = "baseline_unavailable"
    elif delta > 0.001:
        status = "improved"
    elif delta < -0.001:
        status = "needs_review"
    else:
        status = "no_change"

    result["accuracy_booster"] = {
        "available": True,
        "status": status,
        "summary": (
            "Boosted run improved the latest saved score."
            if status == "improved"
            else "Boosted run completed. Compare details before promoting this model."
            if status in {"needs_review", "no_change"}
            else "Boosted run completed. No previous comparable run was available."
        ),
        "comparison": {
            "metric": result.get("primary_metric"),
            "previous_model": previous_run.get("best_model") if previous_run else None,
            "previous_score": previous_score,
            "boosted_model": result.get("best_model"),
            "boosted_score": boosted_score,
            "delta": delta,
        },
        "cleaning": {
            "applied": bool(changed_steps),
            "before_score": cleaning_result["summary"].get("before_score"),
            "after_score": cleaning_result["summary"].get("after_score"),
            "rows_removed": cleaning_result["summary"].get("rows_removed"),
            "columns_removed": cleaning_result["summary"].get("columns_removed"),
            "values_imputed": cleaning_result["summary"].get("values_imputed"),
            "values_capped": cleaning_result["summary"].get("values_capped"),
            "rare_values_grouped": cleaning_result["summary"].get("rare_values_grouped"),
            "steps": [
                {
                    "id": step.get("id"),
                    "title": step.get("title"),
                    "impact": step.get("impact"),
                    "changed": step.get("changed"),
                }
                for step in cleaning_result["steps"]
            ],
        },
        "strategy": [
            {
                "label": "Cleaning",
                "status": "applied" if changed_steps else "checked",
                "detail": f"{len(changed_steps)} cleaning action(s) applied before retraining.",
            },
            {
                "label": "Feature cleanup",
                "status": "applied" if result.get("training_summary", {}).get("dropped_features") else "checked",
                "detail": f"{len(result.get('training_summary', {}).get('dropped_features', []))} noisy feature(s) dropped.",
            },
            {
                "label": "Balancing",
                "status": "applied" if result.get("training_summary", {}).get("balanced_candidate_count") else "checked",
                "detail": f"{result.get('training_summary', {}).get('balanced_candidate_count') or 0} balanced candidate(s) evaluated.",
            },
            {
                "label": "Tuning",
                "status": "applied" if result.get("training_summary", {}).get("tuned_models") else "checked",
                "detail": f"{result.get('training_summary', {}).get('tuned_models') or 0} tuned model(s), {result.get('training_summary', {}).get('improved_tuned_models') or 0} gain(s).",
            },
        ],
    }
    return result


@app.post("/api/model-suggestions")
def model_suggestions(request: ModelSuggestionRequest):
    df = load_dataframe(request.dataset_id)
    return suggest_model_candidates(df, request.target_column, request.problem_type, request.feature_columns)


@app.post("/api/predict/{model_id}")
def predict(model_id: str, request: PredictRequest):
    model_path = MODEL_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        artifact = joblib.load(model_path)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="This saved model artifact cannot be loaded in the current environment. Retrain it before prediction.",
        ) from error
    prediction_result = predict_rows_with_details(artifact, request.rows, request.threshold)
    audit = save_prediction_audit(model_id, request.rows, prediction_result)

    return {
        "model_id": model_id,
        "audit_id": audit["audit_id"],
        **prediction_result,
    }


@app.post("/api/predict/{model_id}/batch")
async def batch_predict(model_id: str, file: UploadFile = File(...), threshold: float | None = Form(None)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file for batch prediction.")

    model_path = MODEL_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        batch_df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as error:
        raise HTTPException(status_code=400, detail="Could not read batch prediction CSV.") from error

    rows = batch_df.astype(object).where(pd.notna(batch_df), None).to_dict(orient="records")
    try:
        artifact = joblib.load(model_path)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="This saved model artifact cannot be loaded in the current environment. Retrain it before batch prediction.",
        ) from error
    prediction_result = predict_rows_with_details(artifact, rows, threshold)
    audit = save_prediction_audit(model_id, rows, prediction_result, request_type="batch", source_name=file.filename)

    output_df = batch_df.copy()
    output_df["prediction"] = prediction_result["predictions"]
    if prediction_result.get("threshold") is not None:
        output_df["threshold_used"] = prediction_result["threshold"]
    details = prediction_result.get("prediction_details", [])

    if details and any("confidence" in item for item in details):
        output_df["prediction_confidence"] = [item.get("confidence") for item in details]
        output_df["prediction_confidence_band"] = [item.get("confidence_band") for item in details]
        output_df["low_confidence"] = [item.get("low_confidence") for item in details]

    probability_labels = sorted(
        {
            label
            for item in details
            for label in (item.get("probabilities") or {}).keys()
        }
    )
    for label in probability_labels:
        safe_label = "".join(character if character.isalnum() else "_" for character in str(label)).strip("_")
        output_df[f"probability_{safe_label or 'class'}"] = [
            (item.get("probabilities") or {}).get(label)
            for item in details
        ]

    output = io.StringIO()
    output_df.to_csv(output, index=False)
    download_name = f"{Path(file.filename).stem}_predictions.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "X-Audit-Id": audit["audit_id"],
        },
    )
