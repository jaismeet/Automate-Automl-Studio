from datetime import datetime
from typing import Any


def _number(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return fallback


def _score(value: Any, fallback: int = 0) -> int:
    return max(0, min(100, round(_number(value, fallback))))


def _status_from_score(score: int, blocked: bool = False) -> str:
    if blocked or score < 55:
        return "critical"
    if score < 70:
        return "review"
    if score < 85:
        return "watch"
    return "healthy"


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _event(kind: str, title: str, detail: str, timestamp: str | None, status: str = "info") -> dict:
    return {
        "kind": kind,
        "title": title,
        "detail": detail,
        "created_at": timestamp,
        "status": status,
    }


def _confidence_monitor(record: dict, audits: dict) -> tuple[dict, int]:
    summary = audits.get("summary") or {}
    average = summary.get("average_confidence")
    low_rows = int(summary.get("low_confidence_rows") or 0)

    if record.get("problem_type") != "classification":
        card = {
            "key": "confidence",
            "label": "Prediction confidence",
            "status": "healthy",
            "value": "Regression",
            "detail": "Regression models are monitored with residuals and prediction audits.",
        }
        return card, 82

    if average is None:
        card = {
            "key": "confidence",
            "label": "Prediction confidence",
            "status": "review",
            "value": "Not audited",
            "detail": "Run what-if or batch predictions to capture confidence signals.",
        }
        return card, 45

    confidence_score = _score(average * 100)
    if low_rows:
        confidence_score = max(0, confidence_score - min(30, low_rows * 5))
    status = _status_from_score(confidence_score)
    card = {
        "key": "confidence",
        "label": "Prediction confidence",
        "status": status,
        "value": f"{round(average * 100, 2)}%",
        "detail": f"{low_rows} low-confidence row(s) found in recent audits.",
    }
    return card, confidence_score


def _drift_monitor(drift: dict | None) -> tuple[dict, int]:
    if not drift:
        return (
            {
                "key": "drift",
                "label": "Dataset drift",
                "status": "review",
                "value": "Pending",
                "detail": "No drift check has been run for this model yet.",
            },
            50,
        )

    status_map = {
        "low": "healthy",
        "medium": "watch",
        "high": "critical",
        "blocked": "critical",
    }
    score = _score(drift.get("score"), 50)
    return (
        {
            "key": "drift",
            "label": "Dataset drift",
            "status": status_map.get(drift.get("status"), "review"),
            "value": f"{score}/100",
            "detail": drift.get("summary") or "Latest drift report is available.",
        },
        score,
    )


def _audit_monitor(audits: dict) -> tuple[dict, int]:
    summary = audits.get("summary") or {}
    audit_rows = audits.get("audits") or []
    count = int(summary.get("filtered_audits") or 0)
    has_batch = any(row.get("request_type") == "batch" for row in audit_rows)
    audit_score = min(100, count * 25 + (25 if has_batch else 0))
    status = _status_from_score(audit_score if count else 45)
    return (
        {
            "key": "audit",
            "label": "Audit coverage",
            "status": status,
            "value": f"{count} event(s)",
            "detail": "Batch scoring tested." if has_batch else "Run a batch prediction before production handoff.",
        },
        audit_score,
    )


def _readiness_monitor(readiness: dict) -> tuple[dict, int]:
    score = _score(readiness.get("score"), 0)
    blocked = readiness.get("status") == "blocked"
    return (
        {
            "key": "readiness",
            "label": "Deployment readiness",
            "status": _status_from_score(score, blocked),
            "value": f"{score}/100",
            "detail": readiness.get("decision") or "Readiness checklist is available.",
        },
        score,
    )


def _trust_monitor(readiness: dict) -> tuple[dict, int]:
    trust = readiness.get("trust_score") or {}
    score = _score(trust.get("score"), 0)
    return (
        {
            "key": "trust",
            "label": "Model trust",
            "status": _status_from_score(score),
            "value": f"{score}/100",
            "detail": trust.get("summary") or "Trust score is calculated from model risk signals.",
        },
        score,
    )


def _timeline(record: dict, audits: dict, drift: dict | None, readiness: dict) -> list[dict]:
    events = [
        _event(
            "model",
            "Model saved",
            f"{record.get('best_model') or 'Model'} trained for {record.get('target_column') or 'target'}.",
            record.get("created_at"),
            "healthy",
        )
    ]

    if drift:
        drift_status = "healthy" if drift.get("status") == "low" else "warning"
        if drift.get("status") in {"high", "blocked"}:
            drift_status = "critical"
        events.append(
            _event(
                "drift",
                "Drift check completed",
                f"{drift.get('comparison_dataset_name') or 'Dataset'} scored {drift.get('score')}/100.",
                drift.get("created_at"),
                drift_status,
            )
        )

    for audit in (audits.get("audits") or [])[:8]:
        summary = audit.get("confidence_summary") or {}
        low_rows = int(summary.get("low_confidence_rows") or 0)
        events.append(
            _event(
                "audit",
                f"{audit.get('request_type') or 'single'} prediction audit",
                f"{audit.get('row_count') or 0} row(s), {low_rows} low-confidence row(s).",
                audit.get("created_at"),
                "warning" if low_rows else "healthy",
            )
        )

    events.sort(key=lambda item: _parse_time(item.get("created_at")), reverse=True)
    if readiness.get("status") == "blocked":
        events.insert(
            0,
            _event(
                "readiness",
                "Deployment blocker detected",
                readiness.get("summary") or "Fix readiness blockers before production.",
                None,
                "critical",
            ),
        )
    return events[:10]


def _recommendation(
    record: dict,
    readiness: dict,
    drift: dict | None,
    audits: dict,
    confidence_score: int,
    health_score: int,
) -> dict:
    if readiness.get("status") == "blocked":
        return {
            "level": "critical",
            "title": "Fix deployment blockers first",
            "detail": readiness.get("summary") or "The readiness checklist has blockers.",
            "action": (readiness.get("actions") or ["Review readiness blockers."])[0],
            "retrain_recommended": False,
        }

    if drift and drift.get("status") in {"high", "blocked"}:
        return {
            "level": "critical",
            "title": "Retrain or validate before production",
            "detail": drift.get("summary") or "High drift was detected.",
            "action": (drift.get("actions") or ["Retrain with recent data."])[0],
            "retrain_recommended": True,
        }

    low_confidence = int((audits.get("summary") or {}).get("low_confidence_rows") or 0)
    if record.get("problem_type") == "classification" and (confidence_score < 65 or low_confidence >= 3):
        return {
            "level": "review",
            "title": "Review confidence before handoff",
            "detail": f"{low_confidence} low-confidence row(s) are present in prediction audits.",
            "action": "Tune the threshold, review weak classes, or retrain with more examples.",
            "retrain_recommended": confidence_score < 55,
        }

    if not drift:
        return {
            "level": "review",
            "title": "Run drift check next",
            "detail": "Monitoring is incomplete until a current scoring dataset is compared.",
            "action": "Select a recent dataset and run the drift check.",
            "retrain_recommended": False,
        }

    if health_score >= 85:
        return {
            "level": "healthy",
            "title": "Model is stable for controlled testing",
            "detail": "Monitoring signals are healthy. Continue checking audits and drift after each data refresh.",
            "action": "Proceed with controlled API testing and keep audit logging enabled.",
            "retrain_recommended": False,
        }

    return {
        "level": "watch",
        "title": "Monitor before full rollout",
        "detail": "The model can be tested, but some monitoring signals still need attention.",
        "action": (readiness.get("actions") or ["Review monitoring cards before handoff."])[0],
        "retrain_recommended": False,
    }


def build_model_monitoring(
    record: dict,
    audits: dict | None,
    drift: dict | None,
    readiness: dict,
) -> dict:
    audits = audits or {"audits": [], "summary": {}}
    readiness_card, readiness_score = _readiness_monitor(readiness)
    trust_card, trust_score = _trust_monitor(readiness)
    drift_card, drift_score = _drift_monitor(drift)
    confidence_card, confidence_score = _confidence_monitor(record, audits)
    audit_card, audit_score = _audit_monitor(audits)

    weighted_score = round(
        readiness_score * 0.34
        + trust_score * 0.22
        + drift_score * 0.2
        + confidence_score * 0.12
        + audit_score * 0.12
    )
    blocked = readiness.get("status") == "blocked" or (drift and drift.get("status") in {"high", "blocked"})
    status = _status_from_score(weighted_score, blocked)
    recommendation = _recommendation(record, readiness, drift, audits, confidence_score, weighted_score)

    checks = readiness.get("checks") or []
    blockers = [check for check in checks if check.get("status") == "blocked"]
    warnings = [check for check in checks if check.get("status") in {"warning", "pending"}]

    return {
        "available": True,
        "model_id": record.get("model_id"),
        "model_name": record.get("best_model"),
        "target_column": record.get("target_column"),
        "problem_type": record.get("problem_type"),
        "status": status,
        "score": weighted_score,
        "summary": recommendation["detail"],
        "recommendation": recommendation,
        "cards": [readiness_card, trust_card, drift_card, confidence_card, audit_card],
        "signals": {
            "readiness_score": readiness_score,
            "trust_score": trust_score,
            "drift_score": drift_score,
            "confidence_score": confidence_score,
            "audit_score": audit_score,
            "blockers": len(blockers),
            "warnings": len(warnings),
        },
        "latest_drift": drift,
        "audit_summary": audits.get("summary") or {},
        "timeline": _timeline(record, audits, drift, readiness),
        "actions": [
            recommendation["action"],
            *[action for action in readiness.get("actions", []) if action != recommendation["action"]],
        ][:6],
    }


def build_monitoring_snapshot(monitoring: dict, created_at: str) -> dict:
    signals = monitoring.get("signals") or {}
    recommendation = monitoring.get("recommendation") or {}
    return {
        "snapshot_id": f"{monitoring.get('model_id')}-{created_at}",
        "model_id": monitoring.get("model_id"),
        "model_name": monitoring.get("model_name"),
        "target_column": monitoring.get("target_column"),
        "problem_type": monitoring.get("problem_type"),
        "status": monitoring.get("status"),
        "score": monitoring.get("score"),
        "readiness_score": signals.get("readiness_score"),
        "trust_score": signals.get("trust_score"),
        "drift_score": signals.get("drift_score"),
        "confidence_score": signals.get("confidence_score"),
        "audit_score": signals.get("audit_score"),
        "blockers": signals.get("blockers"),
        "warnings": signals.get("warnings"),
        "recommendation_level": recommendation.get("level"),
        "recommendation_title": recommendation.get("title"),
        "created_at": created_at,
    }


def _alert(
    key: str,
    level: str,
    title: str,
    detail: str,
    action: str,
    source: str,
) -> dict:
    return {
        "key": key,
        "level": level,
        "title": title,
        "detail": detail,
        "action": action,
        "source": source,
    }


def _card_by_key(monitoring: dict, key: str) -> dict:
    return next((card for card in monitoring.get("cards", []) if card.get("key") == key), {})


def build_monitoring_alerts(monitoring: dict, champion_comparison: dict | None = None) -> list[dict]:
    alerts = []
    signals = monitoring.get("signals") or {}
    recommendation = monitoring.get("recommendation") or {}

    if monitoring.get("status") == "critical":
        alerts.append(
            _alert(
                "monitoring_critical",
                "critical",
                "Production health is critical",
                monitoring.get("summary") or "The model has critical monitoring signals.",
                recommendation.get("action") or "Review monitoring signals before deployment.",
                "monitoring",
            )
        )
    elif monitoring.get("status") in {"review", "watch"}:
        alerts.append(
            _alert(
                "monitoring_watch",
                "warning",
                "Model needs monitoring review",
                monitoring.get("summary") or "The model should be reviewed before full rollout.",
                recommendation.get("action") or "Review monitoring signals.",
                "monitoring",
            )
        )

    drift_card = _card_by_key(monitoring, "drift")
    if drift_card.get("status") == "critical":
        alerts.append(
            _alert(
                "drift_critical",
                "critical",
                "High dataset drift detected",
                drift_card.get("detail") or "The latest drift report requires action.",
                "Retrain or validate with recent labeled data before production use.",
                "drift",
            )
        )
    elif drift_card.get("status") in {"review", "watch"}:
        alerts.append(
            _alert(
                "drift_review",
                "warning",
                "Dataset drift needs review",
                drift_card.get("detail") or "Run or review the drift comparison.",
                "Run a drift check with a recent scoring dataset.",
                "drift",
            )
        )

    readiness_card = _card_by_key(monitoring, "readiness")
    if readiness_card.get("status") == "critical" or int(signals.get("blockers") or 0) > 0:
        alerts.append(
            _alert(
                "readiness_blocked",
                "critical",
                "Deployment readiness is blocked",
                readiness_card.get("detail") or "Readiness checks include blockers.",
                "Fix readiness blockers before production handoff.",
                "readiness",
            )
        )

    confidence_card = _card_by_key(monitoring, "confidence")
    if confidence_card.get("status") in {"critical", "review"}:
        alerts.append(
            _alert(
                "confidence_low",
                "warning",
                "Prediction confidence is weak",
                confidence_card.get("detail") or "Confidence signals need review.",
                "Run what-if tests, tune threshold, or retrain with more labeled examples.",
                "confidence",
            )
        )

    audit_card = _card_by_key(monitoring, "audit")
    if audit_card.get("status") in {"critical", "review"}:
        alerts.append(
            _alert(
                "audit_coverage_low",
                "warning",
                "Audit coverage is low",
                audit_card.get("detail") or "Prediction audit coverage is incomplete.",
                "Run single-row and batch predictions to build audit coverage.",
                "audit",
            )
        )

    if champion_comparison and champion_comparison.get("status") == "challenger":
        champion = champion_comparison.get("champion") or {}
        selected = champion_comparison.get("selected_model") or {}
        alerts.append(
            _alert(
                "champion_available",
                "info",
                "A stronger champion model exists",
                f"{champion.get('best_model') or 'Champion'} is ranked above this selected model.",
                "Use the champion endpoint for production testing or validate this challenger before promotion.",
                "champion",
            )
        )
        if isinstance(selected.get("score_delta_vs_champion"), (int, float)) and selected["score_delta_vs_champion"] >= -0.02:
            alerts.append(
                _alert(
                    "challenger_close",
                    "info",
                    "Selected challenger is close to champion",
                    "This challenger is within 0.02 score points of the champion.",
                    "Compare drift and confidence before choosing which API to use.",
                    "champion",
                )
            )

    return alerts
