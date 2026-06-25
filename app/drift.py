from typing import Any

import pandas as pd


def _safe_number(value: Any, digits: int = 4) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    rounded = round(number, digits)
    return int(rounded) if float(rounded).is_integer() else rounded


def _share(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {}
    values = clean.astype(str).value_counts(normalize=True)
    return {str(key): float(value) for key, value in values.items()}


def _severity(level: float, medium: float, high: float) -> str:
    if level >= high:
        return "high"
    if level >= medium:
        return "medium"
    return "low"


def _check(status: str, label: str, detail: str, action: str) -> dict:
    return {
        "status": status,
        "label": label,
        "detail": detail,
        "action": action,
    }


def _feature_columns(record: dict, reference_df: pd.DataFrame, feature_columns: list[str] | None) -> list[str]:
    if feature_columns:
        return [column for column in feature_columns if column in reference_df.columns]

    training_summary = record.get("training_summary") or {}
    summary_features = training_summary.get("feature_columns") or []
    if summary_features:
        return [column for column in summary_features if column in reference_df.columns]

    target = record.get("target_column")
    return [column for column in reference_df.columns if column != target]


def _missing_drift(reference_df: pd.DataFrame, comparison_df: pd.DataFrame, columns: list[str]) -> tuple[list[dict], int, int]:
    rows = []
    high = 0
    medium = 0
    for column in columns:
        reference_missing = float(reference_df[column].isna().mean())
        comparison_missing = float(comparison_df[column].isna().mean())
        shift = abs(comparison_missing - reference_missing)
        severity = _severity(shift, 0.1, 0.25)
        if severity == "high":
            high += 1
        elif severity == "medium":
            medium += 1
        rows.append(
            {
                "column": column,
                "reference_missing_percent": _safe_number(reference_missing * 100, 2),
                "comparison_missing_percent": _safe_number(comparison_missing * 100, 2),
                "shift_points": _safe_number(shift * 100, 2),
                "severity": severity,
            }
        )

    rows.sort(key=lambda item: item["shift_points"] or 0, reverse=True)
    return rows[:8], high, medium


def _numeric_drift(reference_df: pd.DataFrame, comparison_df: pd.DataFrame, columns: list[str]) -> tuple[list[dict], int, int]:
    rows = []
    high = 0
    medium = 0
    for column in columns:
        reference_numeric = pd.to_numeric(reference_df[column], errors="coerce").dropna()
        comparison_numeric = pd.to_numeric(comparison_df[column], errors="coerce").dropna()
        if reference_numeric.empty or comparison_numeric.empty:
            continue

        reference_mean = float(reference_numeric.mean())
        comparison_mean = float(comparison_numeric.mean())
        reference_std = float(reference_numeric.std(ddof=0) or 0)
        denominator = max(reference_std, abs(reference_mean) * 0.05, 1e-9)
        normalized_shift = abs(comparison_mean - reference_mean) / denominator
        severity = _severity(normalized_shift, 0.75, 1.5)
        if severity == "high":
            high += 1
        elif severity == "medium":
            medium += 1

        rows.append(
            {
                "column": column,
                "reference_mean": _safe_number(reference_mean),
                "comparison_mean": _safe_number(comparison_mean),
                "reference_std": _safe_number(reference_std),
                "mean_shift": _safe_number(normalized_shift, 2),
                "severity": severity,
            }
        )

    rows.sort(key=lambda item: item["mean_shift"] or 0, reverse=True)
    return rows[:8], high, medium


def _categorical_drift(reference_df: pd.DataFrame, comparison_df: pd.DataFrame, columns: list[str]) -> tuple[list[dict], int, int, int]:
    rows = []
    high = 0
    medium = 0
    unseen_warnings = 0
    for column in columns:
        reference_distribution = _share(reference_df[column])
        comparison_distribution = _share(comparison_df[column])
        if not reference_distribution or not comparison_distribution:
            continue

        reference_values = set(reference_distribution)
        comparison_values = set(comparison_distribution)
        all_values = reference_values | comparison_values
        distribution_shift = 0.5 * sum(
            abs(reference_distribution.get(value, 0) - comparison_distribution.get(value, 0))
            for value in all_values
        )
        unseen_share = sum(comparison_distribution.get(value, 0) for value in comparison_values - reference_values)
        severity = _severity(max(distribution_shift, unseen_share), 0.18, 0.35)
        if severity == "high":
            high += 1
        elif severity == "medium":
            medium += 1
        if unseen_share >= 0.08:
            unseen_warnings += 1

        rows.append(
            {
                "column": column,
                "distribution_shift": _safe_number(distribution_shift, 3),
                "unseen_category_share": _safe_number(unseen_share * 100, 2),
                "reference_top": next(iter(reference_distribution), None),
                "comparison_top": next(iter(comparison_distribution), None),
                "severity": severity,
            }
        )

    rows.sort(
        key=lambda item: max(item["distribution_shift"] or 0, (item["unseen_category_share"] or 0) / 100),
        reverse=True,
    )
    return rows[:8], high, medium, unseen_warnings


def build_dataset_drift_report(
    record: dict,
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    comparison_dataset: dict | None,
    feature_columns: list[str] | None = None,
) -> dict:
    required_columns = _feature_columns(record, reference_df, feature_columns)
    comparison_columns = set(comparison_df.columns)
    reference_columns = set(reference_df.columns)
    common_columns = [
        column
        for column in required_columns
        if column in reference_columns and column in comparison_columns
    ]
    missing_required = [column for column in required_columns if column not in comparison_columns]
    extra_columns = [
        column
        for column in comparison_df.columns
        if column not in required_columns and column != record.get("target_column")
    ]

    numeric_columns = [
        column
        for column in common_columns
        if pd.api.types.is_numeric_dtype(reference_df[column])
        or pd.api.types.is_numeric_dtype(comparison_df[column])
    ]
    categorical_columns = [column for column in common_columns if column not in numeric_columns]

    missing_rows, high_missing, medium_missing = _missing_drift(reference_df, comparison_df, common_columns)
    numeric_rows, high_numeric, medium_numeric = _numeric_drift(reference_df, comparison_df, numeric_columns)
    categorical_rows, high_categorical, medium_categorical, unseen_warnings = _categorical_drift(
        reference_df,
        comparison_df,
        categorical_columns,
    )

    penalties = 0
    if missing_required:
        penalties += min(65, 25 + len(missing_required) * 8)
    if extra_columns:
        penalties += min(10, len(extra_columns) * 2)
    penalties += min(20, high_missing * 6 + medium_missing * 3)
    penalties += min(25, high_numeric * 7 + medium_numeric * 3)
    penalties += min(25, high_categorical * 7 + medium_categorical * 3)

    reference_rows = int(reference_df.shape[0])
    comparison_rows = int(comparison_df.shape[0])
    row_ratio = comparison_rows / reference_rows if reference_rows else 0
    row_volume_warning = comparison_rows < 20 or row_ratio < 0.2
    if row_volume_warning:
        penalties += 5

    score = max(0, min(100, 100 - penalties))
    if missing_required or not common_columns:
        status = "blocked"
    elif score >= 80:
        status = "low"
    elif score >= 60:
        status = "medium"
    else:
        status = "high"

    actions = []
    if missing_required:
        actions.append("Add the missing model feature columns before scoring or retrain with the new schema.")
    if high_numeric or high_categorical:
        actions.append("Review the top drifted features and validate accuracy on recent data.")
    if medium_numeric or medium_categorical or high_missing or medium_missing:
        actions.append("Check whether the comparison dataset represents a new customer or time segment.")
    if row_volume_warning:
        actions.append("Run the drift check again with a larger comparison sample.")
    if not actions:
        actions.append("Keep this comparison as the baseline for recurring drift monitoring.")

    checks = []
    if missing_required or not common_columns:
        checks.append(
            _check(
                "blocked",
                "Schema compatibility",
                f"{len(missing_required)} required feature column(s) are missing from the comparison dataset.",
                "Align the scoring dataset schema with the trained model features.",
            )
        )
    elif extra_columns:
        checks.append(
            _check(
                "warning",
                "Schema compatibility",
                f"All model features are present; {len(extra_columns)} extra column(s) will be ignored.",
                "Confirm extra columns are not newly required predictors.",
            )
        )
    else:
        checks.append(
            _check(
                "pass",
                "Schema compatibility",
                "All required model feature columns are present.",
                "Continue with distribution checks.",
            )
        )

    checks.append(
        _check(
            "warning" if row_volume_warning else "pass",
            "Row volume",
            f"Compared {comparison_rows} row(s) against {reference_rows} training row(s).",
            "Use at least 20 recent rows for a more reliable drift signal." if row_volume_warning else "Sample size is sufficient for a quick drift screen.",
        )
    )
    checks.append(
        _check(
            "warning" if high_missing or medium_missing else "pass",
            "Missing values",
            f"{high_missing} high and {medium_missing} medium missing-rate shift(s) detected.",
            "Inspect features with missing-rate changes before deployment." if high_missing or medium_missing else "Missing-value patterns are stable.",
        )
    )
    checks.append(
        _check(
            "warning" if high_numeric or medium_numeric else "pass",
            "Numeric distribution",
            f"{high_numeric} high and {medium_numeric} medium numeric mean shift(s) detected.",
            "Retrain or recalibrate if numeric feature drift affects key predictors." if high_numeric or medium_numeric else "Numeric feature averages are stable.",
        )
    )
    checks.append(
        _check(
            "warning" if high_categorical or medium_categorical else "pass",
            "Category distribution",
            f"{high_categorical} high and {medium_categorical} medium category shift(s) detected.",
            "Validate category-heavy features against recent outcomes." if high_categorical or medium_categorical else "Categorical feature mix is stable.",
        )
    )
    checks.append(
        _check(
            "warning" if unseen_warnings else "pass",
            "Unseen categories",
            f"{unseen_warnings} feature(s) contain notable unseen category share.",
            "Review unseen values and update preprocessing if needed." if unseen_warnings else "No notable unseen-category pressure detected.",
        )
    )

    comparison_name = comparison_dataset.get("name") if comparison_dataset else "comparison dataset"
    reference_name = record.get("dataset_name") or record.get("dataset_id")
    summary = {
        "low": f"Low drift: {comparison_name} is close to the training data.",
        "medium": f"Medium drift: review feature changes before using {comparison_name} in production.",
        "high": f"High drift: retraining or fresh validation is recommended before using {comparison_name}.",
        "blocked": "Drift check is blocked because required model feature columns are missing.",
    }[status]

    return {
        "available": True,
        "model_id": record.get("model_id"),
        "reference_dataset_id": record.get("dataset_id"),
        "reference_dataset_name": reference_name,
        "comparison_dataset_id": comparison_dataset.get("dataset_id") if comparison_dataset else None,
        "comparison_dataset_name": comparison_name,
        "target_column": record.get("target_column"),
        "status": status,
        "score": score,
        "summary": summary,
        "feature_count": len(required_columns),
        "compared_feature_count": len(common_columns),
        "missing_required_columns": missing_required,
        "extra_columns": extra_columns[:20],
        "row_counts": {
            "reference": reference_rows,
            "comparison": comparison_rows,
        },
        "column_counts": {
            "reference": int(reference_df.shape[1]),
            "comparison": int(comparison_df.shape[1]),
            "required_features": len(required_columns),
            "compared_features": len(common_columns),
        },
        "checks": checks,
        "missing_drift": missing_rows,
        "numeric_drift": numeric_rows,
        "categorical_drift": categorical_rows,
        "actions": actions[:5],
    }
