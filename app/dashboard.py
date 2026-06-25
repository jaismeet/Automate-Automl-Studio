import pandas as pd

from app.profiler import detect_column_type


def numeric_bins(series: pd.Series, bins: int = 8) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"labels": [], "values": []}

    counts = pd.cut(clean, bins=min(bins, clean.nunique()), duplicates="drop").value_counts().sort_index()
    return {
        "labels": [str(label) for label in counts.index],
        "values": counts.values.tolist(),
    }


def build_available_filters(df: pd.DataFrame, column_types: dict[str, str]) -> list[dict]:
    filters = []
    for column, kind in column_types.items():
        if kind not in {"categorical", "date"}:
            continue

        values = df[column].dropna()
        if values.empty:
            continue

        if kind == "date":
            parsed = pd.to_datetime(values, errors="coerce").dropna()
            counts = parsed.dt.strftime("%Y-%m").value_counts().head(12)
        else:
            counts = values.astype(str).value_counts().head(12)

        filters.append(
            {
                "column": column,
                "type": kind,
                "options": [
                    {"value": str(value), "count": int(count)}
                    for value, count in counts.items()
                ],
            }
        )

    return filters[:8]


def suggest_dashboard(
    df: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
    active_filters: dict | None = None,
) -> dict:
    source_df = source_df if source_df is not None else df
    active_filters = active_filters or {}
    column_types = {column: detect_column_type(df[column]) for column in df.columns}
    source_column_types = {column: detect_column_type(source_df[column]) for column in source_df.columns}
    numeric = [column for column, kind in column_types.items() if kind == "numeric"]
    categorical = [column for column, kind in column_types.items() if kind == "categorical"]
    dates = [column for column, kind in column_types.items() if kind == "date"]
    text = [column for column, kind in column_types.items() if kind == "text"]

    kpis = [
        {"label": "Active rows", "value": int(df.shape[0])},
        {"label": "Total rows", "value": int(source_df.shape[0])},
        {"label": "Total columns", "value": int(df.shape[1])},
        {"label": "Missing cells", "value": int(df.isna().sum().sum())},
    ]

    for column in numeric[:2]:
        kpis.append(
            {
                "label": f"Average {column}",
                "value": round(float(df[column].mean()), 2),
            }
        )

    charts = []
    for column in categorical[:3]:
        values = df[column].dropna().astype(str).value_counts().head(8)
        charts.append(
            {
                "title": f"Top {column}",
                "type": "bar",
                "x": values.index.tolist(),
                "y": values.values.tolist(),
                "reason": "Categorical columns work well as bar charts.",
            }
        )

    for column in categorical[:2]:
        values = df[column].dropna().astype(str).value_counts().head(6)
        charts.append(
            {
                "title": f"{column} share",
                "type": "pie",
                "x": values.index.tolist(),
                "y": values.values.tolist(),
                "reason": "Pie charts show category contribution to the whole.",
            }
        )

    for column in numeric[:3]:
        bins = numeric_bins(df[column])
        charts.append(
            {
                "title": f"Distribution of {column}",
                "type": "histogram",
                "column": column,
                "x": bins["labels"],
                "y": bins["values"],
                "reason": "Numeric columns are useful for distribution analysis.",
            }
        )

    if len(numeric) >= 2:
        x_column = numeric[0]
        y_column = numeric[1]
        temp = df[[x_column, y_column]].dropna().head(120)
        charts.append(
            {
                "title": f"{x_column} vs {y_column}",
                "type": "scatter",
                "x_label": x_column,
                "y_label": y_column,
                "points": temp.to_dict(orient="records"),
                "reason": "Scatter charts help reveal relationships between numeric columns.",
            }
        )

    if dates and numeric:
        date_column = dates[0]
        value_column = numeric[0]
        temp = df[[date_column, value_column]].dropna().copy()
        temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
        temp = temp.dropna().sort_values(date_column).head(200)
        charts.append(
            {
                "title": f"{value_column} over time",
                "type": "line",
                "x": temp[date_column].dt.strftime("%Y-%m-%d").tolist(),
                "y": temp[value_column].tolist(),
                "reason": "Date plus numeric columns can create trend charts.",
            }
        )

    if len(numeric) >= 2:
        correlation = df[numeric[:5]].corr(numeric_only=True).round(3)
        charts.append(
            {
                "title": "Numeric correlation heatmap",
                "type": "heatmap",
                "columns": correlation.columns.tolist(),
                "matrix": correlation.fillna(0).values.tolist(),
                "reason": "Correlation helps compare numeric feature relationships.",
            }
        )

    filters = categorical[:5] + dates[:2]

    summaries = [
        f"The current slice has {df.shape[0]} rows from {source_df.shape[0]} total rows.",
        f"Detected {len(numeric)} numeric, {len(categorical)} categorical, {len(dates)} date, and {len(text)} text columns.",
        "Suggested dashboard includes bars, pies, histograms, scatter plots, heatmaps, and time trends when the data supports them.",
    ]
    if active_filters:
        summaries.insert(
            1,
            "Active filters: "
            + ", ".join(f"{column} = {value}" for column, value in active_filters.items()),
        )

    return {
        "kpis": kpis,
        "charts": charts,
        "filters": filters,
        "available_filters": build_available_filters(source_df, source_column_types),
        "active_filters": active_filters,
        "active_row_count": int(df.shape[0]),
        "total_row_count": int(source_df.shape[0]),
        "summaries": summaries,
    }
