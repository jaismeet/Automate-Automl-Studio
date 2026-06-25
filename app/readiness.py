from typing import Any

from app.storage import MODEL_DIR
from app.trust import build_model_trust_score


def _add_check(
    checks: list[dict],
    key: str,
    label: str,
    status: str,
    detail: str,
    action: str,
    weight: int,
) -> None:
    checks.append(
        {
            "key": key,
            "label": label,
            "status": status,
            "detail": detail,
            "action": action,
            "weight": int(weight),
        }
    )


def _feature_count(record: dict) -> int:
    studio = record.get("explainability_studio") or {}
    features = studio.get("features") or record.get("top_features") or []
    return len(features)


def _artifact_exists(model_id: str) -> bool:
    return (MODEL_DIR / f"{model_id}.joblib").exists()


def _confidence_check(record: dict, audits: dict, checks: list[dict]) -> None:
    summary = audits.get("summary") or {}
    average_confidence = summary.get("average_confidence")
    low_confidence_rows = int(summary.get("low_confidence_rows") or 0)

    if record.get("problem_type") != "classification":
        _add_check(
            checks,
            "confidence",
            "Prediction confidence",
            "pass",
            "Regression predictions do not require class-confidence bands.",
            "Review residual diagnostics before deployment.",
            10,
        )
        return

    if average_confidence is None:
        _add_check(
            checks,
            "confidence",
            "Prediction confidence",
            "warning",
            "No confidence audit is available yet.",
            "Run the what-if simulator or batch scoring before deployment.",
            10,
        )
        return

    if average_confidence >= 0.75 and low_confidence_rows == 0:
        _add_check(
            checks,
            "confidence",
            "Prediction confidence",
            "pass",
            f"Average audited confidence is {round(average_confidence * 100, 2)}% with no low-confidence rows.",
            "Keep monitoring confidence after deployment.",
            10,
        )
        return

    _add_check(
        checks,
        "confidence",
        "Prediction confidence",
        "warning",
        f"Average audited confidence is {round(average_confidence * 100, 2)}% with {low_confidence_rows} low-confidence row(s).",
        "Review low-confidence rows before production use.",
        10,
    )


def _audit_check(audits: dict, checks: list[dict]) -> None:
    summary = audits.get("summary") or {}
    count = int(summary.get("filtered_audits") or 0)
    if count:
        _add_check(
            checks,
            "audit",
            "Prediction audit",
            "pass",
            f"{count} prediction audit event(s) are saved for this model.",
            "Keep audit logging enabled for production testing.",
            10,
        )
        return

    _add_check(
        checks,
        "audit",
        "Prediction audit",
        "warning",
        "No prediction audit event is saved for this model.",
        "Run at least one what-if or batch prediction before deployment.",
        10,
    )


def _trust_check(record: dict, checks: list[dict]) -> dict:
    trust = build_model_trust_score(record)
    score = trust.get("score") or 0
    if score >= 80:
        status = "pass"
    elif score >= 60:
        status = "warning"
    else:
        status = "blocked"

    _add_check(
        checks,
        "trust",
        "Model trust score",
        status,
        f"{trust.get('trust_label')} at {score}/100: {trust.get('summary')}",
        trust.get("actions", ["Review model risk signals before deployment."])[0],
        20,
    )
    return trust


def _explainability_check(record: dict, checks: list[dict]) -> None:
    count = _feature_count(record)
    if count:
        _add_check(
            checks,
            "explainability",
            "Explainability",
            "pass",
            f"{count} feature driver(s) are available.",
            "Review top drivers with stakeholders.",
            10,
        )
        return

    _add_check(
        checks,
        "explainability",
        "Explainability",
        "warning",
        "No feature driver summary is saved for this model.",
        "Retrain or export explainability before stakeholder review.",
        10,
    )


def _artifact_check(record: dict, checks: list[dict]) -> None:
    model_id = record.get("model_id")
    if model_id and _artifact_exists(model_id):
        _add_check(
            checks,
            "artifact",
            "Model artifact",
            "pass",
            "The trained model artifact is available for download and prediction.",
            "Version the model artifact before production testing.",
            15,
        )
        return

    _add_check(
        checks,
        "artifact",
        "Model artifact",
        "blocked",
        "The trained model artifact is missing.",
        "Retrain the model before deployment.",
        15,
    )


def _api_check(record: dict, checks: list[dict]) -> None:
    if record.get("prediction_api"):
        _add_check(
            checks,
            "api",
            "Prediction API",
            "pass",
            f"Prediction endpoint is saved at {record.get('prediction_api')}.",
            "Test the endpoint with representative rows.",
            10,
        )
        return

    _add_check(
        checks,
        "api",
        "Prediction API",
        "blocked",
        "No prediction endpoint is saved for this model.",
        "Retrain or restore the model API before deployment.",
        10,
    )


def _report_check(record: dict, checks: list[dict]) -> None:
    if record.get("model_id"):
        _add_check(
            checks,
            "report",
            "Model report",
            "pass",
            "The PPTX model report can be generated from this saved model.",
            "Download the report before deployment review.",
            8,
        )
        return

    _add_check(
        checks,
        "report",
        "Model report",
        "warning",
        "No saved model id is available for report generation.",
        "Train and save the model before report export.",
        8,
    )


def _drift_check(record: dict, checks: list[dict], drift: dict | None = None) -> None:
    if drift:
        drift_status = drift.get("status")
        score = drift.get("score")
        comparison_name = drift.get("comparison_dataset_name") or "comparison dataset"
        action = (drift.get("actions") or ["Review drift details before deployment."])[0]

        if drift_status == "low":
            status = "pass"
            detail = f"Latest drift check against {comparison_name} is low risk at {score}/100."
        elif drift_status == "medium":
            status = "warning"
            detail = f"Latest drift check against {comparison_name} shows medium drift at {score}/100."
        else:
            status = "blocked"
            detail = f"Latest drift check against {comparison_name} needs action: {drift.get('summary')}"

        _add_check(
            checks,
            "drift",
            "Dataset drift",
            status,
            detail,
            action,
            7,
        )
        return

    _add_check(
        checks,
        "drift",
        "Dataset drift",
        "pending",
        "No drift comparison has been run yet for this model.",
        "Compare a fresh scoring dataset against the training dataset before production.",
        7,
    )


def _batch_check(audits: dict, checks: list[dict]) -> None:
    audits_list = audits.get("audits") or []
    has_batch = any(item.get("request_type") == "batch" for item in audits_list)
    if has_batch:
        _add_check(
            checks,
            "batch",
            "Batch scoring",
            "pass",
            "At least one batch prediction has been audited.",
            "Review the downloaded scored CSV before handoff.",
            10,
        )
        return

    _add_check(
        checks,
        "batch",
        "Batch scoring",
        "warning",
        "No batch scoring audit is saved for this model.",
        "Run a CSV batch prediction before production handoff.",
        10,
    )


def _score_checks(checks: list[dict]) -> int:
    earned = 0.0
    total = 0.0
    for check in checks:
        weight = float(check.get("weight") or 0)
        total += weight
        if check.get("status") == "pass":
            earned += weight
        elif check.get("status") in {"warning", "pending"}:
            earned += weight * 0.5
    if not total:
        return 0
    return round((earned / total) * 100)


def _decision(score: int, checks: list[dict]) -> tuple[str, str, str]:
    blocked = [check for check in checks if check.get("status") == "blocked"]
    warnings = [check for check in checks if check.get("status") == "warning"]
    pending = [check for check in checks if check.get("status") == "pending"]

    if blocked:
        return (
            "blocked",
            "Not ready for deployment",
            "Critical deployment requirements are missing. Fix blockers before handoff.",
        )
    if score >= 85 and len(warnings) == 0 and len(pending) <= 1:
        return (
            "ready",
            "Ready for controlled deployment",
            "Core deployment checks pass. Continue with monitoring and drift review.",
        )
    return (
        "review",
        "Needs review before deployment",
        "The model can be tested, but warnings or pending checks should be reviewed first.",
    )


def _actions(checks: list[dict]) -> list[str]:
    statuses = {"blocked", "warning", "pending"}
    actions = [check.get("action") for check in checks if check.get("status") in statuses]
    deduped = []
    for action in actions:
        if action and action not in deduped:
            deduped.append(action)
    return deduped[:6]


def build_deployment_readiness(record: dict, audits: dict | None = None, drift: dict | None = None) -> dict:
    checks: list[dict] = []
    audits = audits or {"audits": [], "summary": {}}
    trust = _trust_check(record, checks)
    _artifact_check(record, checks)
    _api_check(record, checks)
    _audit_check(audits, checks)
    _confidence_check(record, audits, checks)
    _batch_check(audits, checks)
    _explainability_check(record, checks)
    _report_check(record, checks)
    _drift_check(record, checks, drift)

    score = _score_checks(checks)
    status, decision, summary = _decision(score, checks)
    return {
        "available": True,
        "model_id": record.get("model_id"),
        "status": status,
        "decision": decision,
        "summary": summary,
        "score": score,
        "passed": len([check for check in checks if check["status"] == "pass"]),
        "warnings": len([check for check in checks if check["status"] == "warning"]),
        "pending": len([check for check in checks if check["status"] == "pending"]),
        "blockers": len([check for check in checks if check["status"] == "blocked"]),
        "trust_score": trust,
        "audit_summary": audits.get("summary") or {},
        "drift_report": drift,
        "checks": checks,
        "actions": _actions(checks),
    }
