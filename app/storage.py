import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import pandas as pd
from fastapi import HTTPException


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REGISTRY_DIR = BASE_DIR / "registry"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
DATASET_REGISTRY_PATH = REGISTRY_DIR / "datasets.json"
MODEL_REGISTRY_PATH = REGISTRY_DIR / "models.json"
PREDICTION_AUDIT_PATH = REGISTRY_DIR / "prediction_audits.json"
DRIFT_REPORT_PATH = REGISTRY_DIR / "drift_reports.json"
MONITORING_SNAPSHOT_PATH = REGISTRY_DIR / "monitoring_snapshots.json"
MONITORING_ALERT_PATH = REGISTRY_DIR / "monitoring_alerts.json"
MODEL_PROMOTION_PATH = REGISTRY_DIR / "model_promotions.json"


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json_list(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def save_dataframe(df: pd.DataFrame, original_name: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid4().hex
    safe_name = Path(original_name).stem.replace(" ", "_")
    path = DATA_DIR / f"{dataset_id}_{safe_name}.parquet"
    df.to_parquet(path, index=False)
    register_dataset(
        {
            "dataset_id": dataset_id,
            "name": original_name,
            "format": "CSV",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "path": str(path),
            "created_at": utc_now(),
        }
    )
    return dataset_id


def load_dataframe(dataset_id: str) -> pd.DataFrame:
    matches = list(DATA_DIR.glob(f"{dataset_id}_*.parquet"))
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return pd.read_parquet(matches[0])


def register_dataset(record: dict) -> None:
    rows = [item for item in read_json_list(DATASET_REGISTRY_PATH) if item["dataset_id"] != record["dataset_id"]]
    rows.insert(0, record)
    write_json_list(DATASET_REGISTRY_PATH, rows)


def list_datasets() -> list[dict]:
    return read_json_list(DATASET_REGISTRY_PATH)


def preview_dataframe(dataset_id: str, limit: int = 20) -> dict:
    df = load_dataframe(dataset_id)
    preview = df.head(limit).where(pd.notna(df), None)
    return {
        "columns": df.columns.tolist(),
        "rows": preview.to_dict(orient="records"),
        "row_count": int(df.shape[0]),
        "preview_count": int(min(limit, df.shape[0])),
    }


def register_model(record: dict) -> None:
    rows = [item for item in read_json_list(MODEL_REGISTRY_PATH) if item["model_id"] != record["model_id"]]
    rows.insert(0, record)
    write_json_list(MODEL_REGISTRY_PATH, rows)


def list_models() -> list[dict]:
    return read_json_list(MODEL_REGISTRY_PATH)


def get_model_record(model_id: str) -> dict | None:
    return next((item for item in list_models() if item.get("model_id") == model_id), None)


def register_prediction_audit(record: dict) -> None:
    rows = read_json_list(PREDICTION_AUDIT_PATH)
    rows.insert(0, record)
    write_json_list(PREDICTION_AUDIT_PATH, rows[:500])


def list_prediction_audits(model_id: str | None = None, limit: int = 30) -> dict:
    rows = read_json_list(PREDICTION_AUDIT_PATH)
    filtered = [
        row
        for row in rows
        if not model_id or row.get("model_id") == model_id
    ]
    capped = filtered[: max(1, min(int(limit or 30), 100))]
    low_confidence = 0
    available_confidence = []
    for row in filtered:
        summary = row.get("confidence_summary") or {}
        low_confidence += int(summary.get("low_confidence_rows") or 0)
        if isinstance(summary.get("average_confidence"), (int, float)):
            available_confidence.append(float(summary["average_confidence"]))

    return {
        "audits": capped,
        "summary": {
            "total_audits": len(rows),
            "filtered_audits": len(filtered),
            "latest_audit": capped[0] if capped else None,
            "low_confidence_rows": low_confidence,
            "average_confidence": round(sum(available_confidence) / len(available_confidence), 4)
            if available_confidence
            else None,
        },
    }


def register_drift_report(record: dict) -> None:
    rows = read_json_list(DRIFT_REPORT_PATH)
    rows.insert(0, record)
    write_json_list(DRIFT_REPORT_PATH, rows[:300])


def list_drift_reports(model_id: str | None = None, limit: int = 20) -> dict:
    rows = read_json_list(DRIFT_REPORT_PATH)
    filtered = [
        row
        for row in rows
        if not model_id or row.get("model_id") == model_id
    ]
    capped = filtered[: max(1, min(int(limit or 20), 100))]
    return {
        "drift_reports": capped,
        "summary": {
            "total_reports": len(rows),
            "filtered_reports": len(filtered),
            "latest_report": capped[0] if capped else None,
        },
    }


def get_latest_drift_report(model_id: str) -> dict | None:
    return next(
        (row for row in read_json_list(DRIFT_REPORT_PATH) if row.get("model_id") == model_id),
        None,
    )


def register_monitoring_snapshot(record: dict) -> None:
    rows = read_json_list(MONITORING_SNAPSHOT_PATH)
    rows.insert(0, record)
    write_json_list(MONITORING_SNAPSHOT_PATH, rows[:700])


def list_monitoring_snapshots(model_id: str | None = None, limit: int = 40) -> dict:
    rows = read_json_list(MONITORING_SNAPSHOT_PATH)
    filtered = [
        row
        for row in rows
        if not model_id or row.get("model_id") == model_id
    ]
    capped = filtered[: max(1, min(int(limit or 40), 120))]
    chronological = list(reversed(capped))
    latest = capped[0] if capped else None
    previous = capped[1] if len(capped) > 1 else None
    score_delta = None
    if latest and previous and isinstance(latest.get("score"), (int, float)) and isinstance(previous.get("score"), (int, float)):
        score_delta = round(float(latest["score"]) - float(previous["score"]), 4)

    return {
        "snapshots": chronological,
        "summary": {
            "total_snapshots": len(rows),
            "filtered_snapshots": len(filtered),
            "latest_snapshot": latest,
            "score_delta": score_delta,
        },
    }


def sync_monitoring_alerts(model_id: str, alerts: list[dict]) -> dict:
    rows = read_json_list(MONITORING_ALERT_PATH)
    now = utc_now()
    active_keys = {alert.get("key") for alert in alerts if alert.get("key")}
    updated_rows = []
    existing_active = {
        row.get("key"): row
        for row in rows
        if row.get("model_id") == model_id and row.get("status") == "active"
    }

    for row in rows:
        if row.get("model_id") == model_id and row.get("status") == "active" and row.get("key") not in active_keys:
            row = {**row, "status": "resolved", "resolved_at": now}
        updated_rows.append(row)

    for alert in alerts:
        key = alert.get("key")
        if not key:
            continue
        existing = existing_active.get(key)
        if existing:
            for row in updated_rows:
                if row.get("alert_id") == existing.get("alert_id"):
                    row.update({**alert, "status": "active", "last_seen": now})
                    break
        else:
            updated_rows.insert(
                0,
                {
                    **alert,
                    "alert_id": uuid4().hex,
                    "model_id": model_id,
                    "status": "active",
                    "created_at": now,
                    "last_seen": now,
                },
            )

    write_json_list(MONITORING_ALERT_PATH, updated_rows[:700])
    return list_monitoring_alerts(model_id=model_id, limit=50)


def list_monitoring_alerts(model_id: str | None = None, limit: int = 30, include_resolved: bool = False) -> dict:
    rows = read_json_list(MONITORING_ALERT_PATH)
    filtered = [
        row
        for row in rows
        if (not model_id or row.get("model_id") == model_id)
        and (include_resolved or row.get("status") == "active")
    ]
    capped = filtered[: max(1, min(int(limit or 30), 100))]
    critical = len([row for row in filtered if row.get("level") == "critical"])
    warning = len([row for row in filtered if row.get("level") == "warning"])
    info = len([row for row in filtered if row.get("level") == "info"])

    return {
        "alerts": capped,
        "summary": {
            "total_alerts": len(rows),
            "filtered_alerts": len(filtered),
            "critical": critical,
            "warning": warning,
            "info": info,
        },
    }


def model_group_key(record: dict) -> str:
    return "::".join(
        [
            str(record.get("target_column") or ""),
            str(record.get("problem_type") or ""),
            str(record.get("primary_metric") or ""),
        ]
    )


def register_model_promotion(record: dict) -> dict:
    rows = read_json_list(MODEL_PROMOTION_PATH)
    group_key = record["group_key"]
    now = utc_now()
    updated = []
    for row in rows:
        if row.get("group_key") == group_key and row.get("status") == "active":
            row = {**row, "status": "replaced", "replaced_at": now}
        updated.append(row)

    promotion = {
        **record,
        "promotion_id": uuid4().hex,
        "status": "active",
        "promoted_at": now,
    }
    updated.insert(0, promotion)
    write_json_list(MODEL_PROMOTION_PATH, updated[:300])
    return promotion


def list_model_promotions(active_only: bool = True, limit: int = 100) -> list[dict]:
    rows = read_json_list(MODEL_PROMOTION_PATH)
    filtered = [
        row
        for row in rows
        if not active_only or row.get("status") == "active"
    ]
    return filtered[: max(1, min(int(limit or 100), 300))]


def get_model_promotion_for_record(record: dict) -> dict | None:
    group_key = model_group_key(record)
    return next(
        (row for row in list_model_promotions(active_only=True) if row.get("group_key") == group_key),
        None,
    )


def _numeric_score(record: dict) -> float | None:
    score = record.get("rank_score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _experiment_group_key(record: dict) -> tuple:
    return (
        record.get("dataset_id"),
        record.get("target_column"),
        record.get("problem_type"),
        record.get("primary_metric"),
    )


def _summarize_experiment(record: dict, group_index: int, group_size: int, previous: dict | None, best: dict | None) -> dict:
    score = _numeric_score(record)
    previous_score = _numeric_score(previous or {})
    score_delta = round(score - previous_score, 4) if score is not None and previous_score is not None else None
    training_summary = record.get("training_summary") or {}
    holdout_metrics = record.get("holdout_metrics") or {}

    return {
        "experiment_id": record.get("experiment_id") or record.get("model_id"),
        "model_id": record.get("model_id"),
        "dataset_id": record.get("dataset_id"),
        "target_column": record.get("target_column"),
        "problem_type": record.get("problem_type"),
        "best_model": record.get("best_model"),
        "quality_label": record.get("quality_label"),
        "primary_metric": record.get("primary_metric"),
        "rank_score": score,
        "previous_score": previous_score,
        "score_delta": score_delta,
        "is_latest": group_index == 0,
        "is_best_for_group": bool(best and best.get("model_id") == record.get("model_id")),
        "group_run_number": group_size - group_index,
        "group_run_count": group_size,
        "candidate_models": record.get("candidate_models"),
        "trained_models": record.get("trained_models"),
        "failed_models": record.get("failed_models"),
        "raw_feature_count": record.get("raw_feature_count") or training_summary.get("raw_feature_count"),
        "model_feature_count": record.get("model_feature_count") or training_summary.get("feature_count"),
        "selected_feature_count": record.get("selected_feature_count") or training_summary.get("selected_feature_count"),
        "excluded_feature_count": record.get("excluded_feature_count") or training_summary.get("excluded_feature_count"),
        "dropped_feature_count": record.get("dropped_feature_count"),
        "leakage_feature_count": record.get("leakage_feature_count"),
        "engineered_feature_count": record.get("engineered_feature_count"),
        "tuned_models": (record.get("tuning_studio") or {}).get("tuned_count"),
        "improved_tuned_models": (record.get("tuning_studio") or {}).get("improved_count"),
        "recommended_threshold": record.get("recommended_threshold"),
        "holdout_metrics": holdout_metrics,
        "baseline_metrics": record.get("baseline_metrics") or {},
        "leaderboard_snapshot": record.get("leaderboard_snapshot") or [],
        "next_actions": record.get("next_actions") or [],
        "prediction_api": record.get("prediction_api"),
        "created_at": record.get("created_at"),
    }


def list_experiment_history(dataset_id: str | None = None, target_column: str | None = None, limit: int = 20) -> dict:
    rows = list_models()
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_experiment_group_key(row), []).append(row)

    best_by_group: dict[tuple, dict | None] = {}
    for key, group_rows in grouped.items():
        scored_rows = [row for row in group_rows if _numeric_score(row) is not None]
        best_by_group[key] = max(scored_rows, key=_numeric_score) if scored_rows else None

    enriched = []
    for group_rows in grouped.values():
        group_size = len(group_rows)
        best = best_by_group[_experiment_group_key(group_rows[0])]
        for index, row in enumerate(group_rows):
            previous = group_rows[index + 1] if index + 1 < group_size else None
            enriched.append(_summarize_experiment(row, index, group_size, previous, best))

    filtered = [
        item
        for item in enriched
        if (not dataset_id or item.get("dataset_id") == dataset_id)
        and (not target_column or item.get("target_column") == target_column)
    ]
    filtered = filtered[: max(1, min(int(limit or 20), 100))]
    scored_filtered = [item for item in filtered if isinstance(item.get("rank_score"), (int, float))]
    best_run = max(scored_filtered, key=lambda item: item["rank_score"]) if scored_filtered else None
    latest_run = filtered[0] if filtered else None
    improved_runs = len([item for item in filtered if isinstance(item.get("score_delta"), (int, float)) and item["score_delta"] > 0])

    return {
        "experiments": filtered,
        "summary": {
            "total_runs": len(rows),
            "filtered_runs": len(filtered),
            "best_run": best_run,
            "latest_run": latest_run,
            "improved_runs": improved_runs,
        },
    }
