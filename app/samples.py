from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from app.profiler import profile_dataframe
from app.storage import SAMPLE_DATA_DIR


SAMPLE_DETAILS = {
    "customers": {
        "title": "Customer Churn",
        "description": "Subscription behavior, support load, satisfaction, and churn labels.",
        "target_column": "churn",
        "problem_type": "classification",
        "domain": "Retention",
    },
    "customer_churn": {
        "title": "Customer Churn Pro",
        "description": "Expanded churn dataset with contract, payment, engagement, and support signals.",
        "target_column": "churn",
        "problem_type": "classification",
        "domain": "Retention",
    },
    "loan_default": {
        "title": "Loan Default Risk",
        "description": "Credit, income, debt, and payment history for default classification.",
        "target_column": "defaulted",
        "problem_type": "classification",
        "domain": "Risk",
    },
    "retail_sales_forecast": {
        "title": "Retail Sales Forecast",
        "description": "Store, campaign, traffic, inventory, and seasonal drivers for sales regression.",
        "target_column": "next_month_sales",
        "problem_type": "regression",
        "domain": "Forecasting",
    },
}


def safe_sample_path(sample_id: str) -> Path:
    safe_id = Path(sample_id).name.replace(".csv", "")
    path = SAMPLE_DATA_DIR / f"{safe_id}.csv"
    if not path.exists() or path.parent != SAMPLE_DATA_DIR:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    return path


def sample_record(path: Path) -> dict:
    df = pd.read_csv(path)
    key = path.stem
    profile = profile_dataframe(df)
    details = SAMPLE_DETAILS.get(key, {})
    target_column = details.get("target_column")

    if not target_column and profile["target_suggestions"]:
        target_column = profile["target_suggestions"][0]["name"]

    return {
        "sample_id": key,
        "filename": path.name,
        "title": details.get("title", path.stem.replace("_", " ").title()),
        "description": details.get("description", "Sample CSV dataset for interactive AutoML testing."),
        "domain": details.get("domain", "General"),
        "target_column": target_column,
        "problem_type": details.get("problem_type"),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": len(df.select_dtypes(exclude=["number"]).columns),
        "target_suggestions": profile["target_suggestions"],
    }


def list_sample_datasets() -> list[dict]:
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return [sample_record(path) for path in sorted(SAMPLE_DATA_DIR.glob("*.csv"))]


def load_sample_dataframe(sample_id: str) -> tuple[pd.DataFrame, dict]:
    path = safe_sample_path(sample_id)
    df = pd.read_csv(path)
    return df, sample_record(path)
