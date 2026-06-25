import warnings

import pandas as pd


DEFAULT_CLEANING_OPTIONS = {
    "remove_duplicates": True,
    "drop_missing_target": True,
    "drop_constant_columns": True,
    "drop_id_columns": True,
    "impute_missing": True,
    "cap_outliers": True,
    "group_rare_categories": True,
}


def normalized_column_name(column: str) -> str:
    return str(column).lower().replace("-", "_").replace(" ", "_")


def is_id_like(column: str, unique_ratio: float, row_count: int) -> bool:
    normalized = normalized_column_name(column)
    name_matches = normalized == "id" or normalized.endswith("_id") or normalized.endswith("id")
    return bool(name_matches and row_count >= 20 and unique_ratio >= 0.9)


def severity_from_percent(percent: float) -> str:
    if percent >= 30:
        return "high"
    if percent >= 10:
        return "medium"
    return "low"


def is_probably_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    non_null = series.dropna()
    if len(non_null) < 8:
        return False

    sample = non_null.astype(str).head(60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def issue(type_name: str, severity: str, title: str, detail: str, action: str, columns=None, metric=None) -> dict:
    return {
        "type": type_name,
        "severity": severity,
        "title": title,
        "detail": detail,
        "action": action,
        "columns": columns or [],
        "metric": metric,
    }


def detect_outliers(df: pd.DataFrame) -> list[dict]:
    outliers = []
    numeric_columns = df.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.nunique() < 8:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count:
            outliers.append(
                {
                    "column": column,
                    "count": count,
                    "percent": round(float(count / max(len(series), 1) * 100), 2),
                    "lower_bound": round(float(lower), 4),
                    "upper_bound": round(float(upper), 4),
                }
            )

    return sorted(outliers, key=lambda item: item["count"], reverse=True)


def score_quality(row_count: int, column_count: int, issues: list[dict]) -> int:
    if row_count == 0 or column_count == 0:
        return 0

    penalty = 0
    for item in issues:
        if item["severity"] == "high":
            penalty += 16
        elif item["severity"] == "medium":
            penalty += 9
        else:
            penalty += 4

    return max(0, min(100, 100 - penalty))


def grade_quality(score: int) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 55:
        return "needs review"
    return "high risk"


def health_risk_level(score: int) -> str:
    if score >= 85:
        return "low"
    if score >= 65:
        return "moderate"
    return "high"


def health_status(score: int, blocker_count: int) -> str:
    if blocker_count:
        return "Fix blockers before training"
    if score >= 85:
        return "Ready for training"
    if score >= 65:
        return "Train with caution"
    return "Clean before training"


def bounded_score(value: float) -> int:
    return int(max(0, min(100, round(float(value)))))


def build_dataset_health(
    row_count: int,
    column_count: int,
    issues: list[dict],
    missing_percent: float,
    duplicate_percent: float,
    constant_columns: list[str],
    id_columns: list[str],
    high_cardinality_columns: list[str],
    outliers: list[dict],
) -> dict:
    severity_counts = {
        "high": len([item for item in issues if item["severity"] == "high"]),
        "medium": len([item for item in issues if item["severity"] == "medium"]),
        "low": len([item for item in issues if item["severity"] == "low"]),
    }
    issue_score = score_quality(row_count, column_count, issues)
    completeness_score = bounded_score(100 - min(70, missing_percent * 2.0))
    duplication_score = bounded_score(100 - min(70, duplicate_percent * 2.5))

    structure_penalty = 0
    if row_count < 30:
        structure_penalty += 35
    elif row_count < 100:
        structure_penalty += 18
    if column_count < 2:
        structure_penalty += 50
    structure_penalty += min(30, len(constant_columns) * 8)
    structure_penalty += min(24, len(id_columns) * 6)
    structure_score = bounded_score(100 - structure_penalty)

    signal_penalty = min(35, len(outliers) * 7) + min(32, len(high_cardinality_columns) * 8)
    if column_count < 3:
        signal_penalty += 18
    signal_score = bounded_score(100 - signal_penalty)

    score = bounded_score(
        issue_score * 0.35
        + completeness_score * 0.20
        + duplication_score * 0.15
        + structure_score * 0.15
        + signal_score * 0.15
    )

    blockers = []
    if row_count < 30:
        blockers.append("Dataset has fewer than 30 rows; validation and model ranking may be unreliable.")
    if column_count < 2:
        blockers.append("Dataset needs at least one feature column and one target column.")
    blockers.extend(item["title"] for item in issues if item["severity"] == "high")

    warnings = [item["title"] for item in issues if item["severity"] == "medium"][:6]
    checks = [
        {
            "label": "Completeness",
            "score": completeness_score,
            "status": "good" if completeness_score >= 85 else "warning" if completeness_score >= 65 else "risk",
            "detail": f"{round(float(missing_percent), 2)}% missing cells.",
        },
        {
            "label": "Duplicates",
            "score": duplication_score,
            "status": "good" if duplication_score >= 90 else "warning" if duplication_score >= 70 else "risk",
            "detail": f"{round(float(duplicate_percent), 2)}% duplicate rows.",
        },
        {
            "label": "Structure",
            "score": structure_score,
            "status": "good" if structure_score >= 85 else "warning" if structure_score >= 65 else "risk",
            "detail": f"{row_count} rows and {column_count} columns.",
        },
        {
            "label": "Model signal",
            "score": signal_score,
            "status": "good" if signal_score >= 85 else "warning" if signal_score >= 65 else "risk",
            "detail": f"{len(high_cardinality_columns)} high-cardinality and {len(outliers)} outlier column group(s).",
        },
    ]

    actions = []
    if blockers:
        actions.extend(blockers[:3])
    actions.extend(item["action"] for item in issues[:4])
    if not actions:
        actions.append("Dataset looks ready. Select a target and train models.")

    return {
        "score": score,
        "grade": grade_quality(score),
        "risk_level": health_risk_level(score),
        "status": health_status(score, len(blockers)),
        "can_train": len(blockers) == 0,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "blockers": blockers[:6],
        "warnings": warnings,
        "checks": checks,
        "issue_breakdown": severity_counts,
        "recommended_actions": actions[:6],
    }


def analyze_data_quality(df: pd.DataFrame) -> dict:
    row_count = int(df.shape[0])
    column_count = int(df.shape[1])
    total_cells = max(row_count * column_count, 1)
    issues = []

    duplicate_rows = int(df.duplicated().sum())
    duplicate_percent = duplicate_rows / max(row_count, 1) * 100
    if duplicate_rows:
        issues.append(
            issue(
                "duplicates",
                severity_from_percent(duplicate_percent),
                "Duplicate rows detected",
                f"{duplicate_rows} row(s) are exact duplicates.",
                "Review and remove duplicates before model training if they are accidental.",
                metric={"count": duplicate_rows, "percent": round(float(duplicate_percent), 2)},
            )
        )

    missing_cells = int(df.isna().sum().sum())
    missing_percent = missing_cells / total_cells * 100
    missing_columns = [
        {
            "column": column,
            "missing": int(df[column].isna().sum()),
            "percent": round(float(df[column].isna().mean() * 100), 2),
        }
        for column in df.columns
        if df[column].isna().any()
    ]
    if missing_cells:
        top_missing = sorted(missing_columns, key=lambda item: item["missing"], reverse=True)[:5]
        issues.append(
            issue(
                "missing",
                severity_from_percent(missing_percent),
                "Missing values found",
                f"{missing_cells} missing cell(s) across {len(missing_columns)} column(s).",
                "Impute important columns or remove rows/columns with heavy missingness.",
                columns=[item["column"] for item in top_missing],
                metric={"count": missing_cells, "percent": round(float(missing_percent), 2)},
            )
        )

    constant_columns = [column for column in df.columns if int(df[column].nunique(dropna=True)) <= 1]
    if constant_columns:
        issues.append(
            issue(
                "constant",
                "medium",
                "Constant columns",
                f"{len(constant_columns)} column(s) have only one usable value.",
                "Exclude constant columns from training because they do not add model signal.",
                columns=constant_columns[:8],
                metric={"count": len(constant_columns)},
            )
        )

    id_columns = []
    high_cardinality_columns = []
    for column in df.columns:
        non_null_count = int(df[column].notna().sum())
        unique_count = int(df[column].nunique(dropna=True))
        unique_ratio = unique_count / max(non_null_count, 1)

        if is_id_like(column, unique_ratio, row_count):
            id_columns.append(column)
            continue

        if (
            not pd.api.types.is_numeric_dtype(df[column])
            and not is_probably_datetime(df[column])
            and unique_count > 50
            and unique_ratio >= 0.6
        ):
            high_cardinality_columns.append(column)

    if id_columns:
        issues.append(
            issue(
                "id_like",
                "medium",
                "ID-like columns",
                f"{len(id_columns)} column(s) look like identifiers.",
                "Exclude identifier columns unless they represent real business behavior.",
                columns=id_columns[:8],
                metric={"count": len(id_columns)},
            )
        )

    if high_cardinality_columns:
        issues.append(
            issue(
                "high_cardinality",
                "medium",
                "High-cardinality categorical columns",
                f"{len(high_cardinality_columns)} text/categorical column(s) have many unique values.",
                "Group rare values or exclude noisy high-cardinality columns before training.",
                columns=high_cardinality_columns[:8],
                metric={"count": len(high_cardinality_columns)},
            )
        )

    outliers = detect_outliers(df)
    if outliers:
        total_outlier_values = sum(item["count"] for item in outliers)
        max_percent = max(item["percent"] for item in outliers)
        issues.append(
            issue(
                "outliers",
                severity_from_percent(max_percent),
                "Numeric outliers",
                f"{total_outlier_values} outlier value(s) found across {len(outliers)} numeric column(s).",
                "Inspect outliers; cap, transform, or keep them if they are valid business events.",
                columns=[item["column"] for item in outliers[:5]],
                metric={"count": total_outlier_values, "columns": len(outliers)},
            )
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    issues = sorted(issues, key=lambda item: (severity_rank.get(item["severity"], 3), item["type"]))
    score = score_quality(row_count, column_count, issues)
    health = build_dataset_health(
        row_count,
        column_count,
        issues,
        missing_percent,
        duplicate_percent,
        constant_columns,
        id_columns,
        high_cardinality_columns,
        outliers,
    )

    return {
        "summary": {
            "score": score,
            "grade": grade_quality(score),
            "issue_count": len(issues),
            "duplicate_rows": duplicate_rows,
            "missing_cells": missing_cells,
            "outlier_columns": len(outliers),
        },
        "health": health,
        "issues": issues,
        "outliers": outliers[:10],
        "missing_columns": sorted(missing_columns, key=lambda item: item["missing"], reverse=True)[:10],
        "recommended_actions": health["recommended_actions"],
    }


def normalized_options(options: dict | None = None) -> dict:
    merged = DEFAULT_CLEANING_OPTIONS.copy()
    if options:
        for key in merged:
            if key in options:
                merged[key] = bool(options[key])
    return merged


def step_result(step_id: str, title: str, detail: str, impact: str, changed: bool) -> dict:
    return {
        "id": step_id,
        "title": title,
        "detail": detail,
        "impact": impact,
        "changed": changed,
    }


def fill_value_for_series(series: pd.Series):
    if pd.api.types.is_bool_dtype(series):
        mode = series.dropna().mode()
        return bool(mode.iloc[0]) if not mode.empty else False

    if pd.api.types.is_numeric_dtype(series):
        median = pd.to_numeric(series, errors="coerce").median()
        return 0 if pd.isna(median) else median

    mode = series.dropna().mode()
    if not mode.empty:
        return mode.iloc[0]

    return "Unknown"


def find_constant_columns(df: pd.DataFrame, target_column: str | None = None) -> list[str]:
    return [
        column
        for column in df.columns
        if column != target_column and int(df[column].nunique(dropna=True)) <= 1
    ]


def find_id_columns(df: pd.DataFrame, target_column: str | None = None) -> list[str]:
    row_count = int(df.shape[0])
    id_columns = []
    for column in df.columns:
        if column == target_column:
            continue

        non_null_count = int(df[column].notna().sum())
        unique_count = int(df[column].nunique(dropna=True))
        unique_ratio = unique_count / max(non_null_count, 1)
        if is_id_like(column, unique_ratio, row_count):
            id_columns.append(column)

    return id_columns


def find_high_cardinality_columns(df: pd.DataFrame, target_column: str | None = None) -> list[str]:
    high_cardinality_columns = []
    for column in df.columns:
        if column == target_column or pd.api.types.is_numeric_dtype(df[column]) or is_probably_datetime(df[column]):
            continue

        non_null_count = int(df[column].notna().sum())
        unique_count = int(df[column].nunique(dropna=True))
        unique_ratio = unique_count / max(non_null_count, 1)
        if unique_count > 50 and unique_ratio >= 0.6:
            high_cardinality_columns.append(column)

    return high_cardinality_columns


def clean_dataframe(df: pd.DataFrame, target_column: str | None = None, options: dict | None = None) -> dict:
    active_options = normalized_options(options)
    target = target_column if target_column in df.columns else None
    cleaned = df.copy()
    before_quality = analyze_data_quality(df)
    steps = []
    stats = {
        "rows_removed": 0,
        "columns_removed": 0,
        "values_imputed": 0,
        "values_capped": 0,
        "rare_values_grouped": 0,
    }

    duplicate_rows = int(cleaned.duplicated().sum())
    if active_options["remove_duplicates"] and duplicate_rows:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        stats["rows_removed"] += duplicate_rows
    steps.append(
        step_result(
            "remove_duplicates",
            "Remove duplicate rows",
            f"{duplicate_rows} exact duplicate row(s) detected.",
            f"{duplicate_rows} row(s) removed" if duplicate_rows else "No duplicate rows",
            bool(active_options["remove_duplicates"] and duplicate_rows),
        )
    )

    target_missing = int(cleaned[target].isna().sum()) if target else 0
    if active_options["drop_missing_target"] and target_missing:
        cleaned = cleaned[cleaned[target].notna()].reset_index(drop=True)
        stats["rows_removed"] += target_missing
    steps.append(
        step_result(
            "drop_missing_target",
            "Drop rows missing target",
            f"{target_missing} row(s) have a missing target value." if target else "No target selected for this check.",
            f"{target_missing} row(s) removed" if target_missing else "Target values are ready",
            bool(active_options["drop_missing_target"] and target_missing),
        )
    )

    constant_columns = find_constant_columns(cleaned, target)
    if active_options["drop_constant_columns"] and constant_columns:
        cleaned = cleaned.drop(columns=constant_columns)
        stats["columns_removed"] += len(constant_columns)
    steps.append(
        step_result(
            "drop_constant_columns",
            "Drop constant columns",
            f"{len(constant_columns)} column(s) have one usable value.",
            ", ".join(constant_columns[:4]) + ("..." if len(constant_columns) > 4 else "") if constant_columns else "No constant columns",
            bool(active_options["drop_constant_columns"] and constant_columns),
        )
    )

    id_columns = find_id_columns(cleaned, target)
    if active_options["drop_id_columns"] and id_columns:
        cleaned = cleaned.drop(columns=id_columns)
        stats["columns_removed"] += len(id_columns)
    steps.append(
        step_result(
            "drop_id_columns",
            "Drop identifier columns",
            f"{len(id_columns)} column(s) look like IDs.",
            ", ".join(id_columns[:4]) + ("..." if len(id_columns) > 4 else "") if id_columns else "No ID-like columns",
            bool(active_options["drop_id_columns"] and id_columns),
        )
    )

    imputed_columns = []
    for column in list(cleaned.columns):
        if column == target:
            continue

        missing = int(cleaned[column].isna().sum())
        if not missing:
            continue

        if active_options["impute_missing"]:
            cleaned[column] = cleaned[column].fillna(fill_value_for_series(cleaned[column]))
            stats["values_imputed"] += missing
        imputed_columns.append({"column": column, "missing": missing})
    steps.append(
        step_result(
            "impute_missing",
            "Fill missing feature values",
            f"{sum(item['missing'] for item in imputed_columns)} missing feature value(s) found.",
            f"{len(imputed_columns)} column(s) imputed" if imputed_columns else "No missing feature values",
            bool(active_options["impute_missing"] and imputed_columns),
        )
    )

    capped_columns = []
    for outlier in detect_outliers(cleaned):
        column = outlier["column"]
        if column == target or column not in cleaned.columns:
            continue

        if active_options["cap_outliers"]:
            numeric = pd.to_numeric(cleaned[column], errors="coerce")
            changed = int(((numeric < outlier["lower_bound"]) | (numeric > outlier["upper_bound"])).sum())
            capped = numeric.clip(lower=outlier["lower_bound"], upper=outlier["upper_bound"])
            cleaned[column] = capped.where(numeric.notna(), cleaned[column])
            stats["values_capped"] += changed
        capped_columns.append(outlier)
    steps.append(
        step_result(
            "cap_outliers",
            "Cap numeric outliers",
            f"{sum(item['count'] for item in capped_columns)} outlier value(s) found in feature columns.",
            f"{len(capped_columns)} column(s) capped" if capped_columns else "No numeric feature outliers",
            bool(active_options["cap_outliers"] and capped_columns),
        )
    )

    grouped_columns = []
    for column in find_high_cardinality_columns(cleaned, target):
        series = cleaned[column].astype("string")
        value_counts = series.value_counts(dropna=True)
        keep_values = set(value_counts.head(25).index)
        rare_mask = series.notna() & ~series.isin(keep_values)
        rare_count = int(rare_mask.sum())
        if not rare_count:
            continue

        if active_options["group_rare_categories"]:
            cleaned[column] = series.mask(rare_mask, "__rare__").astype(object)
            stats["rare_values_grouped"] += rare_count
        grouped_columns.append({"column": column, "count": rare_count})
    steps.append(
        step_result(
            "group_rare_categories",
            "Group rare categories",
            f"{sum(item['count'] for item in grouped_columns)} rare/high-cardinality value(s) found.",
            f"{len(grouped_columns)} column(s) grouped" if grouped_columns else "No rare category grouping needed",
            bool(active_options["group_rare_categories"] and grouped_columns),
        )
    )

    after_quality = analyze_data_quality(cleaned)
    summary = {
        "before_rows": int(df.shape[0]),
        "after_rows": int(cleaned.shape[0]),
        "before_columns": int(df.shape[1]),
        "after_columns": int(cleaned.shape[1]),
        "before_score": before_quality["summary"]["score"],
        "after_score": after_quality["summary"]["score"],
        "before_issue_count": before_quality["summary"]["issue_count"],
        "after_issue_count": after_quality["summary"]["issue_count"],
        **stats,
    }

    return {
        "cleaned_df": cleaned,
        "summary": summary,
        "steps": steps,
        "before_quality": before_quality,
        "after_quality": after_quality,
        "apply_available": any(step["changed"] for step in steps),
        "options": active_options,
    }
