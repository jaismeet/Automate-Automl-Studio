import warnings

import pandas as pd


TARGET_NAME_HINTS = {
    "churn": ("classification", "Customer churn outcome"),
    "default": ("classification", "Loan or payment default outcome"),
    "defaulted": ("classification", "Loan or payment default outcome"),
    "fraud": ("classification", "Fraud detection outcome"),
    "converted": ("classification", "Conversion outcome"),
    "conversion": ("classification", "Conversion outcome"),
    "risk": ("classification", "Risk class or flag"),
    "approved": ("classification", "Approval outcome"),
    "sales": ("regression", "Sales or demand forecast"),
    "revenue": ("regression", "Revenue forecast"),
    "price": ("regression", "Price prediction"),
    "spend": ("regression", "Spend prediction"),
    "amount": ("regression", "Amount prediction"),
    "score": ("regression", "Score prediction"),
    "demand": ("regression", "Demand forecast"),
}


def detect_column_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dates = pd.to_datetime(non_null, errors="coerce")
    if parsed_dates.notna().mean() > 0.8:
        return "date"

    unique_ratio = non_null.nunique() / len(non_null)
    avg_text_length = non_null.astype(str).str.len().mean()

    if unique_ratio < 0.3 and non_null.nunique() <= 50:
        return "categorical"

    if avg_text_length > 35:
        return "text"

    return "categorical"


def infer_target_problem_type(series: pd.Series, column_type: str) -> str:
    if column_type == "numeric" and series.nunique(dropna=True) > 15:
        return "regression"
    return "classification"


def suggest_target_columns(df: pd.DataFrame, column_profiles: list[dict]) -> list[dict]:
    suggestions = []
    profile_by_name = {profile["name"]: profile for profile in column_profiles}

    for column in df.columns:
        profile = profile_by_name[column]
        column_type = profile["type"]
        if column_type in {"empty", "date", "text"}:
            continue

        non_null_count = int(df[column].notna().sum())
        if non_null_count < 10 or profile["unique"] < 2:
            continue

        lower_name = column.lower()
        matched_hint = next((hint for hint in TARGET_NAME_HINTS if hint in lower_name), None)
        inferred_problem = infer_target_problem_type(df[column], column_type)
        problem_type = TARGET_NAME_HINTS[matched_hint][0] if matched_hint else inferred_problem
        reason = TARGET_NAME_HINTS[matched_hint][1] if matched_hint else "Modelable column with enough labeled rows"

        score = 40
        if matched_hint:
            score += 45
        if lower_name in {"id", "customer_id", "applicant_id", "store_id"} or lower_name.endswith("_id"):
            score -= 35
        if column_type == "numeric" and profile["unique"] == non_null_count:
            score -= 8
        if problem_type == "classification" and 2 <= profile["unique"] <= 12:
            score += 10
        if problem_type == "regression" and profile["unique"] > 15:
            score += 10

        suggestions.append(
            {
                "name": column,
                "problem_type": problem_type,
                "reason": reason,
                "score": score,
            }
        )

    return sorted(suggestions, key=lambda item: item["score"], reverse=True)[:5]


def profile_dataframe(df: pd.DataFrame) -> dict:
    columns = []

    for column in df.columns:
        series = df[column]
        column_type = detect_column_type(series)
        details = {
            "name": column,
            "type": column_type,
            "missing": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
        }

        if column_type == "numeric":
            details.update(
                {
                    "mean": round(float(series.mean()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                }
            )
        else:
            top_values = series.dropna().astype(str).value_counts().head(5)
            details["top_values"] = top_values.to_dict()

        columns.append(details)

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": columns,
        "target_suggestions": suggest_target_columns(df, columns),
    }
