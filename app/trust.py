from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int):
        return value
    return 0


def _metric_value(metrics: dict, *names: str) -> float | None:
    for name in names:
        value = _number(metrics.get(name))
        if value is not None:
            return value
    return None


def _first_metric_name(metrics: dict, *names: str) -> str | None:
    for name in names:
        if _number(metrics.get(name)) is not None:
            return name
    return None


def _add_check(checks: list[dict], label: str, status: str, detail: str, points: int = 0) -> int:
    checks.append(
        {
            "label": label,
            "status": status,
            "detail": detail,
            "points": int(points),
        }
    )
    return int(points)


def _performance_penalty(record: dict, checks: list[dict]) -> int:
    problem_type = record.get("problem_type")
    score = _number(record.get("rank_score"))
    metric = record.get("primary_metric") or ("f1_weighted" if problem_type == "classification" else "r2")

    if score is None:
        return _add_check(
            checks,
            "Validation performance",
            "risk",
            "No ranking score is saved for this model.",
            18,
        )

    if problem_type == "regression":
        if score >= 0.65:
            return _add_check(checks, "Validation performance", "good", f"{metric} is strong at {score:.4f}.")
        if score >= 0.35:
            return _add_check(checks, "Validation performance", "warning", f"{metric} is usable but needs review at {score:.4f}.", 8)
        return _add_check(checks, "Validation performance", "risk", f"{metric} is weak at {score:.4f}.", 22)

    if score >= 0.85:
        return _add_check(checks, "Validation performance", "good", f"{metric} is strong at {score:.4f}.")
    if score >= 0.7:
        return _add_check(checks, "Validation performance", "warning", f"{metric} is usable but needs review at {score:.4f}.", 8)
    return _add_check(checks, "Validation performance", "risk", f"{metric} is weak at {score:.4f}.", 22)


def _baseline_penalty(record: dict, checks: list[dict]) -> int:
    metrics = record.get("holdout_metrics") or {}
    baseline = record.get("baseline_metrics") or {}
    problem_type = record.get("problem_type")

    if problem_type == "regression":
        metric_name = _first_metric_name(metrics, "r2")
    else:
        metric_name = _first_metric_name(metrics, "f1_weighted", "balanced_accuracy", "accuracy")

    if not metric_name:
        return _add_check(
            checks,
            "Baseline lift",
            "info",
            "No holdout metric is saved to compare against the baseline.",
            4,
        )

    model_value = _number(metrics.get(metric_name))
    baseline_value = _number(baseline.get(metric_name))
    if model_value is None or baseline_value is None:
        return _add_check(
            checks,
            "Baseline lift",
            "info",
            f"{metric_name.replace('_', ' ')} is saved, but baseline comparison is incomplete.",
            4,
        )

    lift = model_value - baseline_value
    label = metric_name.replace("_", " ")
    if problem_type == "regression":
        if model_value >= 0.35 and lift >= 0.15:
            return _add_check(checks, "Baseline lift", "good", f"Model beats baseline {label} by {lift:.4f}.")
        if model_value >= 0.15 and lift > 0:
            return _add_check(checks, "Baseline lift", "warning", f"Model beats baseline {label}, but lift is only {lift:.4f}.", 8)
        return _add_check(checks, "Baseline lift", "risk", f"Model barely beats or trails baseline {label} by {lift:.4f}.", 18)

    if lift >= 0.12:
        return _add_check(checks, "Baseline lift", "good", f"Model beats baseline {label} by {lift:.4f}.")
    if lift >= 0.04:
        return _add_check(checks, "Baseline lift", "warning", f"Model beats baseline {label}, but lift is modest at {lift:.4f}.", 8)
    return _add_check(checks, "Baseline lift", "risk", f"Model is too close to baseline {label}; lift is {lift:.4f}.", 18)


def _leakage_penalty(record: dict, checks: list[dict]) -> int:
    leakage_count = _count(record.get("leakage_features")) or _count(record.get("leakage_feature_count"))
    if leakage_count == 0:
        return _add_check(checks, "Leakage guard", "good", "No target leakage columns were detected in the saved run.")
    if leakage_count <= 2:
        return _add_check(checks, "Leakage guard", "warning", f"{leakage_count} possible leakage feature(s) were auto-excluded.", 8)
    return _add_check(checks, "Leakage guard", "risk", f"{leakage_count} possible leakage feature(s) were detected.", 16)


def _imbalance_penalty(record: dict, checks: list[dict]) -> int:
    if record.get("problem_type") != "classification":
        return _add_check(checks, "Class balance", "info", "Class balance is not required for regression models.")

    balance = record.get("class_balance") or {}
    severity = balance.get("severity")
    if not severity:
        return _add_check(checks, "Class balance", "info", "Class balance details are not saved for this run.", 4)
    if severity == "balanced":
        return _add_check(checks, "Class balance", "good", "Target classes are balanced enough for standard validation.")
    if severity == "mild":
        return _add_check(checks, "Class balance", "warning", "Target classes have mild imbalance.", 4)
    if severity == "moderate":
        return _add_check(checks, "Class balance", "warning", "Target classes have moderate imbalance.", 9)
    return _add_check(checks, "Class balance", "risk", "Target classes have severe imbalance.", 15)


def _confidence_penalty(record: dict, checks: list[dict]) -> int:
    if record.get("problem_type") != "classification":
        return _add_check(checks, "Prediction confidence", "info", "Confidence bands are mainly used for classification predictions.")

    if record.get("recommended_threshold") is not None:
        return _add_check(checks, "Prediction confidence", "good", "Threshold tuning is available for confidence-aware prediction.")

    metrics = record.get("holdout_metrics") or {}
    confidence_ready = _metric_value(metrics, "precision_weighted", "recall_weighted", "balanced_accuracy") is not None
    if confidence_ready:
        return _add_check(checks, "Prediction confidence", "warning", "Classifier metrics are saved, but no tuned threshold is available.", 6)
    return _add_check(checks, "Prediction confidence", "warning", "No probability or threshold readiness signal is saved.", 8)


def _feature_penalty(record: dict, checks: list[dict]) -> int:
    model_features = _number(record.get("model_feature_count"))
    raw_features = _number(record.get("raw_feature_count"))
    dropped = _count(record.get("dropped_feature_count")) or _count((record.get("training_summary") or {}).get("dropped_features"))

    if model_features is not None and model_features < 3:
        return _add_check(checks, "Feature health", "risk", f"Only {int(model_features)} model feature(s) remain after preprocessing.", 12)
    if dropped >= 5:
        return _add_check(checks, "Feature health", "warning", f"{dropped} noisy or ID-like feature(s) were removed.", 6)
    if raw_features and model_features and model_features < raw_features * 0.5:
        return _add_check(checks, "Feature health", "warning", "More than half of raw features were removed or transformed.", 5)
    return _add_check(checks, "Feature health", "good", "Feature cleanup did not leave a major health concern.")


def _validation_penalty(record: dict, checks: list[dict]) -> int:
    trained = _number(record.get("trained_models"))
    candidates = _number(record.get("candidate_models"))
    failed = _number(record.get("failed_models")) or 0
    leaderboard = record.get("leaderboard_snapshot") or []
    has_cv = any((item.get("cross_validation") or {}).get("available") for item in leaderboard)

    if candidates and failed / max(candidates, 1) >= 0.4:
        return _add_check(checks, "Validation coverage", "risk", f"{int(failed)} of {int(candidates)} candidate model(s) failed.", 12)
    if trained and trained >= 4 and has_cv:
        return _add_check(checks, "Validation coverage", "good", f"{int(trained)} model(s) trained with cross-validation evidence.")
    if trained and trained >= 4:
        return _add_check(checks, "Validation coverage", "warning", f"{int(trained)} model(s) trained, but cross-validation detail is limited.", 5)
    return _add_check(checks, "Validation coverage", "warning", "Limited candidate coverage is saved for this model.", 8)


def _explainability_penalty(record: dict, checks: list[dict]) -> int:
    studio = record.get("explainability_studio") or {}
    features = studio.get("features") or record.get("top_features") or []
    if features:
        return _add_check(checks, "Explainability", "good", f"{len(features)} feature driver(s) are available for review.")
    return _add_check(checks, "Explainability", "warning", "No feature importance explanation is saved.", 6)


def _trust_actions(record: dict, checks: list[dict], risk_level: str) -> list[str]:
    actions = []
    if risk_level == "high":
        actions.append("Do not promote this model until high-risk checks are fixed or accepted.")
    elif risk_level == "medium":
        actions.append("Review warnings before using this model for business-critical predictions.")
    else:
        actions.append("Model is ready for controlled testing with monitoring.")

    for check in checks:
        if check["status"] not in {"warning", "risk"}:
            continue
        label = check["label"]
        if label == "Leakage guard":
            actions.append("Keep leakage columns excluded unless they are available before prediction time.")
        elif label == "Class balance":
            actions.append("Review minority-class recall and threshold settings before deployment.")
        elif label == "Baseline lift":
            actions.append("Add stronger predictor columns if baseline lift remains small.")
        elif label == "Validation performance":
            actions.append("Retrain after improving data quality or adding more labeled rows.")
        elif label == "Prediction confidence":
            actions.append("Use the what-if simulator and batch scoring to review low-confidence rows.")
        elif label == "Explainability":
            actions.append("Retrain or export explainability before stakeholder review.")

    actions.extend(record.get("next_actions") or [])
    deduped = []
    for action in actions:
        if action and action not in deduped:
            deduped.append(action)
    return deduped[:6]


def build_model_trust_score(record: dict) -> dict:
    checks: list[dict] = []
    penalty = 0
    penalty += _performance_penalty(record, checks)
    penalty += _baseline_penalty(record, checks)
    penalty += _leakage_penalty(record, checks)
    penalty += _imbalance_penalty(record, checks)
    penalty += _confidence_penalty(record, checks)
    penalty += _feature_penalty(record, checks)
    penalty += _validation_penalty(record, checks)
    penalty += _explainability_penalty(record, checks)

    score = max(0, min(100, 100 - penalty))
    if score >= 80:
        risk_level = "low"
        trust_label = "Low risk"
        decision = "Ready for controlled testing"
        summary = "This model has strong trust signals. Keep monitoring confidence and data drift."
    elif score >= 60:
        risk_level = "medium"
        trust_label = "Medium risk"
        decision = "Review before deployment"
        summary = "This model is usable, but warnings should be reviewed before production use."
    else:
        risk_level = "high"
        trust_label = "High risk"
        decision = "Do not deploy yet"
        summary = "This model has high-risk signals. Improve data, validation, or explainability first."

    return {
        "available": True,
        "score": score,
        "risk_level": risk_level,
        "trust_label": trust_label,
        "decision": decision,
        "summary": summary,
        "blockers": len([check for check in checks if check["status"] == "risk"]),
        "warnings": len([check for check in checks if check["status"] == "warning"]),
        "checks": checks,
        "actions": _trust_actions(record, checks, risk_level),
    }


def attach_model_trust_score(record: dict) -> dict:
    enriched = dict(record)
    enriched["trust_score"] = build_model_trust_score(record)
    return enriched
