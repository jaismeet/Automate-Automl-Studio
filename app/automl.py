import warnings
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.storage import MODEL_DIR, register_model, utc_now


def infer_problem_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 15:
        return "regression"
    return "classification"


def validate_target_for_problem(target: pd.Series, problem_type: str, target_column: str) -> None:
    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(target):
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{target_column}' is not numeric, so it cannot be used for regression. "
                "Choose Classification, Auto detect, or select a numeric target column."
            ),
        )

    if problem_type == "classification" and target.nunique(dropna=True) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"'{target_column}' needs at least two classes for classification.",
        )


class BalancedOversampleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, random_state: int = 42):
        self.estimator = estimator
        self.random_state = random_state

    def _resample(self, X, y):
        y_series = pd.Series(y).reset_index(drop=True)
        counts = y_series.value_counts()
        if len(counts) < 2:
            return X, y

        rng = np.random.default_rng(self.random_state)
        target_count = int(counts.max())
        indices = []
        y_values = y_series.to_numpy()
        for label, count in counts.items():
            label_indices = np.flatnonzero(y_values == label)
            indices.extend(label_indices.tolist())
            needed = target_count - int(count)
            if needed > 0 and len(label_indices):
                indices.extend(rng.choice(label_indices, size=needed, replace=True).tolist())

        rng.shuffle(indices)
        if hasattr(X, "iloc"):
            X_resampled = X.iloc[indices]
        else:
            X_resampled = X[indices]
        y_resampled = y_series.iloc[indices].to_numpy()
        return X_resampled, y_resampled

    def fit(self, X, y):
        X_resampled, y_resampled = self._resample(X, y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_resampled, y_resampled)
        self.classes_ = getattr(self.estimator_, "classes_", np.unique(y_resampled))
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("Wrapped estimator does not expose predict_proba.")
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError("Wrapped estimator does not expose decision_function.")
        return self.estimator_.decision_function(X)


def analyze_class_balance(target: pd.Series) -> dict:
    counts = target.value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0 or counts.empty:
        return {"available": False, "reason": "No class labels found."}

    majority_label = counts.index[0]
    minority_label = counts.index[-1]
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])
    majority_share = majority_count / max(total, 1)
    minority_share = minority_count / max(total, 1)
    imbalance_ratio = majority_count / max(minority_count, 1)

    if majority_share >= 0.85 or imbalance_ratio >= 8:
        severity = "severe"
    elif majority_share >= 0.7 or imbalance_ratio >= 4:
        severity = "moderate"
    elif majority_share >= 0.6 or imbalance_ratio >= 2:
        severity = "mild"
    else:
        severity = "balanced"

    return {
        "available": True,
        "severity": severity,
        "class_count": int(len(counts)),
        "majority_class": str(majority_label),
        "minority_class": str(minority_label),
        "majority_count": majority_count,
        "minority_count": minority_count,
        "majority_share": round(float(majority_share), 4),
        "minority_share": round(float(minority_share), 4),
        "imbalance_ratio": round(float(imbalance_ratio), 2),
        "classes": [
            {"label": str(label), "count": int(count), "share": round(float(count / max(total, 1)), 4)}
            for label, count in counts.items()
        ],
    }


def is_date_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    non_empty = series.dropna()
    if non_empty.empty:
        return False
    return float(parsed.notna().mean()) >= 0.7 and int(parsed.nunique(dropna=True)) >= 3


def normalized_feature_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def classification_category_purity(feature: pd.Series, target: pd.Series) -> float:
    frame = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if frame.empty or frame["feature"].nunique(dropna=True) < 2:
        return 0.0

    grouped = frame.groupby("feature")["target"].agg(lambda values: values.value_counts(normalize=True).iloc[0])
    weights = frame["feature"].value_counts(normalize=True)
    aligned = grouped.reindex(weights.index).fillna(0)
    return float((aligned * weights).sum())


def detect_leakage_columns(X: pd.DataFrame, y: pd.Series, target_column: str, problem_type: str) -> list[dict]:
    target_name = normalized_feature_name(target_column)
    leakage_columns = []

    y_string = y.astype("string")
    y_numeric = pd.to_numeric(y, errors="coerce")
    encoded_target = pd.Series(pd.factorize(y.astype(str))[0], index=y.index)

    for column in X.columns:
        series = X[column]
        normalized = normalized_feature_name(column)
        non_null_mask = series.notna() & y.notna()
        if int(non_null_mask.sum()) < 10:
            continue

        reasons = []
        severity = "medium"
        score = 0.0

        exact_match = (series[non_null_mask].astype("string") == y_string[non_null_mask]).mean()
        if exact_match >= 0.98:
            reasons.append("feature value almost exactly matches the target")
            severity = "high"
            score = max(score, float(exact_match))

        target_name_hit = target_name and target_name in normalized and normalized != target_name
        if target_name_hit:
            reasons.append("column name contains the target name")
            score = max(score, 0.9)

        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_frame = pd.DataFrame({"feature": numeric_series, "target": y_numeric}).dropna()
        if problem_type == "regression" and len(numeric_frame) >= 10 and numeric_frame["feature"].nunique() > 1:
            corr = numeric_frame["feature"].corr(numeric_frame["target"])
            if pd.notna(corr) and abs(float(corr)) >= 0.98:
                reasons.append(f"numeric correlation with target is {round(float(corr), 4)}")
                severity = "high"
                score = max(score, abs(float(corr)))

        if problem_type == "classification":
            if numeric_series.notna().sum() >= 10 and numeric_series.nunique(dropna=True) > 1 and encoded_target.nunique() > 1:
                corr_frame = pd.DataFrame({"feature": numeric_series, "target": encoded_target}).dropna()
                corr = corr_frame["feature"].corr(corr_frame["target"])
                if pd.notna(corr) and abs(float(corr)) >= 0.98:
                    reasons.append(f"encoded target correlation is {round(float(corr), 4)}")
                    severity = "high"
                    score = max(score, abs(float(corr)))

            unique_ratio = int(series.nunique(dropna=True)) / max(int(series.notna().sum()), 1)
            if unique_ratio < 0.9:
                purity = classification_category_purity(series, y)
                if purity >= 0.98:
                    reasons.append(f"feature categories predict the target with {round(purity * 100, 1)}% purity")
                    severity = "high"
                    score = max(score, float(purity))

        if reasons:
            leakage_columns.append(
                {
                    "name": column,
                    "severity": severity,
                    "score": round(float(score), 4),
                    "reason": "; ".join(dict.fromkeys(reasons)),
                    "action": "Auto-excluded from model training to prevent misleading accuracy.",
                }
            )

    return sorted(leakage_columns, key=lambda item: (item["severity"] != "high", -item["score"], item["name"]))


def build_feature_plan(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    target_column: str | None = None,
    problem_type: str | None = None,
) -> dict:
    row_count = max(int(X.shape[0]), 1)
    date_columns = [column for column in X.columns if is_date_like(X[column])]
    date_column_set = set(date_columns)
    dropped_columns = []
    leakage_columns = detect_leakage_columns(X, y, target_column or "", problem_type or "classification") if y is not None and target_column else []
    leakage_column_set = {item["name"] for item in leakage_columns}

    for column in X.columns:
        if column in date_column_set or column in leakage_column_set:
            continue

        unique_count = int(X[column].nunique(dropna=True))
        non_null_count = int(X[column].notna().sum())
        unique_ratio = unique_count / max(non_null_count, 1)
        normalized_name = normalized_feature_name(column)

        if unique_count <= 1:
            dropped_columns.append(
                {
                    "name": column,
                    "reason": "constant or empty feature",
                    "kind": "constant",
                }
            )
            continue

        id_like_name = normalized_name == "id" or normalized_name.endswith("_id") or normalized_name.endswith("id")
        if id_like_name and row_count >= 20 and unique_ratio >= 0.9:
            dropped_columns.append(
                {
                    "name": column,
                    "reason": "identifier-like high-cardinality feature",
                    "kind": "id_like",
                }
            )

    generated_features = [
        generated
        for column in date_columns
        for generated in [
            f"{column}_year",
            f"{column}_month",
            f"{column}_day",
            f"{column}_dayofweek",
            f"{column}_elapsed_days",
        ]
    ]

    return {
        "original_columns": X.columns.tolist(),
        "date_columns": date_columns,
        "dropped_columns": dropped_columns,
        "leakage_columns": leakage_columns,
        "generated_features": generated_features,
    }


def apply_feature_plan(X: pd.DataFrame, feature_plan: dict) -> pd.DataFrame:
    transformed = X.copy()

    for column in feature_plan.get("date_columns", []):
        if column not in transformed.columns:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(transformed[column], errors="coerce")
        min_date = parsed.min()
        if pd.isna(min_date):
            elapsed_days = np.nan
        else:
            elapsed_days = (parsed - min_date).dt.days
        transformed[f"{column}_year"] = parsed.dt.year
        transformed[f"{column}_month"] = parsed.dt.month
        transformed[f"{column}_day"] = parsed.dt.day
        transformed[f"{column}_dayofweek"] = parsed.dt.dayofweek
        transformed[f"{column}_elapsed_days"] = elapsed_days

    drop_names = [
        item["name"]
        for item in feature_plan.get("dropped_columns", [])
        if item.get("name") in transformed.columns
    ]
    drop_names.extend(
        item["name"]
        for item in feature_plan.get("leakage_columns", [])
        if item.get("name") in transformed.columns
    )
    drop_names.extend(
        column
        for column in feature_plan.get("date_columns", [])
        if column in transformed.columns
    )

    if drop_names:
        transformed = transformed.drop(columns=list(dict.fromkeys(drop_names)))

    return transformed


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def get_model_specs(problem_type: str) -> list[dict]:
    if problem_type == "classification":
        return [
            {
                "name": "Logistic Regression",
                "fit": "Fast linear baseline",
                "why": "Good first classifier and easy to compare against complex models.",
                "estimator": LogisticRegression(max_iter=2000, class_weight="balanced"),
            },
            {
                "name": "Ridge Classifier",
                "fit": "Stable linear model",
                "why": "Useful when there are many encoded categorical features.",
                "estimator": RidgeClassifier(class_weight="balanced"),
            },
            {
                "name": "Linear SVM",
                "fit": "Margin classifier",
                "why": "Strong for high-dimensional tabular data after one-hot encoding.",
                "estimator": LinearSVC(class_weight="balanced", max_iter=5000, random_state=42),
            },
            {
                "name": "KNN Classifier",
                "fit": "Similarity model",
                "why": "Checks whether nearby rows have similar target outcomes.",
                "estimator": KNeighborsClassifier(n_neighbors=7),
            },
            {
                "name": "Decision Tree Classifier",
                "fit": "Interpretable tree",
                "why": "Simple non-linear baseline that shows whether split rules are useful.",
                "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=42),
            },
            {
                "name": "Random Forest Classifier",
                "fit": "Strong general model",
                "why": "Handles mixed feature types well after preprocessing and captures non-linear patterns.",
                "estimator": RandomForestClassifier(n_estimators=180, class_weight="balanced", random_state=42),
            },
            {
                "name": "Random Forest Tuned Classifier",
                "fit": "Tuned tree ensemble",
                "why": "Uses deeper search settings with leaf control to improve generalization on noisy rows.",
                "estimator": RandomForestClassifier(
                    n_estimators=260,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            },
            {
                "name": "Extra Trees Classifier",
                "fit": "Fast tree ensemble",
                "why": "Adds randomized tree splits and often performs well on tabular data.",
                "estimator": ExtraTreesClassifier(n_estimators=180, class_weight="balanced", random_state=42),
            },
            {
                "name": "Extra Trees Tuned Classifier",
                "fit": "Tuned randomized trees",
                "why": "Tries a larger randomized forest with leaf control for stronger tabular accuracy.",
                "estimator": ExtraTreesClassifier(
                    n_estimators=260,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            },
            {
                "name": "Gradient Boosting Classifier",
                "fit": "High accuracy candidate",
                "why": "Often performs well when relationships are not purely linear.",
                "estimator": GradientBoostingClassifier(random_state=42),
            },
            {
                "name": "Gradient Boosting Tuned Classifier",
                "fit": "Tuned boosting",
                "why": "Uses slower learning with more trees to improve fit while reducing over-correction.",
                "estimator": GradientBoostingClassifier(
                    n_estimators=180,
                    learning_rate=0.045,
                    max_depth=2,
                    random_state=42,
                ),
            },
            {
                "name": "Histogram Gradient Boosting Classifier",
                "fit": "Fast boosted trees",
                "why": "Adds a modern boosting candidate that can handle non-linear tabular signal efficiently.",
                "estimator": HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08, random_state=42),
            },
            {
                "name": "Balanced Subsample Forest Classifier",
                "fit": "Imbalance-aware forest",
                "why": "Reweights each tree bootstrap sample so minority classes get a fairer vote.",
                "estimator": RandomForestClassifier(
                    n_estimators=260,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            },
            {
                "name": "Oversampled Logistic Regression",
                "fit": "Balanced linear classifier",
                "why": "Duplicates minority-class rows inside training folds to improve recall on rare classes.",
                "estimator": BalancedOversampleClassifier(LogisticRegression(max_iter=2000)),
            },
            {
                "name": "Oversampled Random Forest",
                "fit": "Balanced tree ensemble",
                "why": "Trains a forest on a resampled class-balanced training set.",
                "estimator": BalancedOversampleClassifier(
                    RandomForestClassifier(n_estimators=220, min_samples_leaf=2, random_state=42)
                ),
            },
            {
                "name": "Oversampled Gradient Boosting",
                "fit": "Balanced boosting",
                "why": "Gives boosting a balanced view of minority labels without requiring external packages.",
                "estimator": BalancedOversampleClassifier(
                    GradientBoostingClassifier(n_estimators=170, learning_rate=0.05, max_depth=2, random_state=42)
                ),
            },
            {
                "name": "AdaBoost Classifier",
                "fit": "Boosted weak learners",
                "why": "Tests whether sequentially corrected shallow trees improve classification.",
                "estimator": AdaBoostClassifier(random_state=42),
            },
            {
                "name": "Voting Ensemble Classifier",
                "fit": "Combined classifier",
                "why": "Blends logistic, random forest, and boosting predictions for a stronger final vote.",
                "estimator": VotingClassifier(
                    estimators=[
                        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced")),
                        (
                            "forest",
                            RandomForestClassifier(n_estimators=180, class_weight="balanced", random_state=42),
                        ),
                        ("boosting", GradientBoostingClassifier(random_state=42)),
                    ],
                    voting="soft",
                ),
            },
        ]

    return [
        {
            "name": "Ridge Regression",
            "fit": "Fast linear baseline",
            "why": "Good first regression model when numeric relationships are mostly linear.",
            "estimator": Ridge(),
        },
        {
            "name": "Lasso Regression",
            "fit": "Sparse linear model",
            "why": "Can reduce weak features by pushing some coefficients toward zero.",
            "estimator": Lasso(alpha=0.01, max_iter=5000, random_state=42),
        },
        {
            "name": "Elastic Net Regression",
            "fit": "Regularized linear model",
            "why": "Balances Ridge and Lasso behavior for noisy tabular features.",
            "estimator": ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=5000, random_state=42),
        },
        {
            "name": "KNN Regressor",
            "fit": "Similarity model",
            "why": "Predicts from nearby rows and helps test local patterns.",
            "estimator": KNeighborsRegressor(n_neighbors=7),
        },
        {
            "name": "Decision Tree Regressor",
            "fit": "Interpretable tree",
            "why": "Simple non-linear baseline for rule-like regression patterns.",
            "estimator": DecisionTreeRegressor(random_state=42),
        },
        {
            "name": "Random Forest Regressor",
            "fit": "Strong general model",
            "why": "Captures non-linear patterns and works well on tabular datasets.",
            "estimator": RandomForestRegressor(n_estimators=180, random_state=42),
        },
        {
            "name": "Random Forest Tuned Regressor",
            "fit": "Tuned tree ensemble",
            "why": "Uses more trees with controlled leaves to reduce overfitting on noisy numeric targets.",
            "estimator": RandomForestRegressor(
                n_estimators=260,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
            ),
        },
        {
            "name": "Extra Trees Regressor",
            "fit": "Fast tree ensemble",
            "why": "Adds randomized splits and often improves robust tabular regression.",
            "estimator": ExtraTreesRegressor(n_estimators=180, random_state=42),
        },
        {
            "name": "Extra Trees Tuned Regressor",
            "fit": "Tuned randomized trees",
            "why": "Adds a larger randomized ensemble with leaf control for robust numeric prediction.",
            "estimator": ExtraTreesRegressor(
                n_estimators=260,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
            ),
        },
        {
            "name": "Gradient Boosting Regressor",
            "fit": "High accuracy candidate",
            "why": "Useful when feature interactions matter.",
            "estimator": GradientBoostingRegressor(random_state=42),
        },
        {
            "name": "Gradient Boosting Tuned Regressor",
            "fit": "Tuned boosting",
            "why": "Uses slower learning with more trees to improve non-linear regression fit.",
            "estimator": GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.045,
                max_depth=2,
                random_state=42,
            ),
        },
        {
            "name": "Histogram Gradient Boosting Regressor",
            "fit": "Fast boosted trees",
            "why": "Adds a modern tree boosting candidate for stronger non-linear numeric prediction.",
            "estimator": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.08, random_state=42),
        },
        {
            "name": "AdaBoost Regressor",
            "fit": "Boosted weak learners",
            "why": "Tests whether sequentially corrected trees improve numeric prediction.",
            "estimator": AdaBoostRegressor(random_state=42),
        },
        {
            "name": "Support Vector Regressor",
            "fit": "Kernel model",
            "why": "Adds a non-linear margin-based regression candidate for smaller datasets.",
            "estimator": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        },
        {
            "name": "Voting Ensemble Regressor",
            "fit": "Combined regressor",
            "why": "Averages linear, forest, and boosting predictions to reduce single-model risk.",
            "estimator": VotingRegressor(
                estimators=[
                    ("ridge", Ridge()),
                    ("forest", RandomForestRegressor(n_estimators=180, random_state=42)),
                    ("boosting", GradientBoostingRegressor(random_state=42)),
                ]
            ),
        },
    ]


def get_models(problem_type: str) -> list[tuple[str, object]]:
    return [(item["name"], item["estimator"]) for item in get_model_specs(problem_type)]


def resolve_feature_columns(df: pd.DataFrame, target_column: str, feature_columns: list[str] | None = None) -> list[str]:
    available_features = [column for column in df.columns if column != target_column]
    if feature_columns is None:
        return available_features

    selected = []
    seen = set()
    for column in feature_columns:
        if column == target_column or column in seen:
            continue
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Feature column not found: {column}")
        selected.append(column)
        seen.add(column)

    if not selected:
        raise HTTPException(status_code=400, detail="Select at least one feature column for training.")

    return selected


def suggest_model_candidates(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = "auto",
    feature_columns: list[str] | None = None,
) -> dict:
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found.")

    y = df[target_column].dropna()
    selected_problem_type = infer_problem_type(y) if problem_type == "auto" else problem_type
    if selected_problem_type not in {"classification", "regression"}:
        raise HTTPException(status_code=400, detail="Problem type must be auto, classification, or regression.")
    validate_target_for_problem(y, selected_problem_type, target_column)

    selected_features = resolve_feature_columns(df, target_column, feature_columns)
    feature_count = len(selected_features)
    row_count = int(df.dropna(subset=[target_column]).shape[0])
    feature_frame = df.dropna(subset=[target_column])
    aligned_y = feature_frame[target_column]
    X = feature_frame[selected_features]
    feature_plan = build_feature_plan(X, aligned_y, target_column, selected_problem_type)
    model_X = apply_feature_plan(X, feature_plan)
    numeric_count = len(model_X.select_dtypes(include=["number"]).columns)
    categorical_count = len(model_X.select_dtypes(exclude=["number"]).columns)

    suggestions = [
        {
            "name": spec["name"],
            "fit": spec["fit"],
            "why": spec["why"],
            "trainable": True,
        }
        for spec in get_model_specs(selected_problem_type)
    ]
    primary_metric = "f1_weighted" if selected_problem_type == "classification" else "r2"

    return {
        "problem_type": selected_problem_type,
        "target_column": target_column,
        "primary_metric": primary_metric,
        "candidate_count": len(suggestions),
        "dataset_summary": {
            "rows_with_target": row_count,
            "feature_count": feature_count,
            "model_feature_count": int(model_X.shape[1]),
            "numeric_features": numeric_count,
            "categorical_features": categorical_count,
            "engineered_features": len(feature_plan["generated_features"]),
            "dropped_features": len(feature_plan["dropped_columns"]),
            "leakage_features": len(feature_plan.get("leakage_columns", [])),
            "selected_features": len(selected_features),
            "excluded_features": max((len(df.columns) - 1) - len(selected_features), 0),
        },
        "selected_features": selected_features,
        "feature_plan": feature_plan,
        "suggestions": suggestions,
    }


def score_model(problem_type: str, y_true, y_pred) -> dict:
    if problem_type == "classification":
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        }

    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def ranking_score(problem_type: str, metrics: dict) -> float:
    if problem_type == "classification":
        return metrics["f1_weighted"]
    return metrics["r2"]


def quality_label(problem_type: str, score: float | None) -> str:
    if score is None:
        return "failed"
    if problem_type == "classification":
        if score >= 0.9:
            return "excellent"
        if score >= 0.75:
            return "good"
        if score >= 0.6:
            return "fair"
        return "needs work"

    if score >= 0.8:
        return "excellent"
    if score >= 0.6:
        return "good"
    if score >= 0.3:
        return "fair"
    return "needs work"


def baseline_metrics(problem_type: str, y_train: pd.Series, y_test: pd.Series) -> dict:
    if problem_type == "classification":
        majority_class = y_train.mode().iloc[0]
        baseline_predictions = [majority_class] * len(y_test)
        return {
            "strategy": f"always predict {majority_class}",
            "accuracy": round(float(accuracy_score(y_test, baseline_predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, baseline_predictions)), 4),
            "f1_weighted": round(
                float(f1_score(y_test, baseline_predictions, average="weighted", zero_division=0)),
                4,
            ),
        }

    baseline_value = float(y_train.mean())
    baseline_predictions = [baseline_value] * len(y_test)
    return {
        "strategy": "predict training mean",
        "r2": round(float(r2_score(y_test, baseline_predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, baseline_predictions)), 4),
    }


def cross_validation_score(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, problem_type: str) -> dict:
    try:
        if problem_type == "classification":
            min_class_count = int(y.value_counts().min())
            folds = min(5, min_class_count)
            if folds < 2:
                return {"available": False, "reason": "Not enough examples in each class for cross-validation."}
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            scoring = "f1_weighted"
        else:
            folds = min(5, len(y))
            if folds < 2:
                return {"available": False, "reason": "Not enough rows for cross-validation."}
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)
            scoring = "r2"

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        return {
            "available": True,
            "metric": scoring,
            "folds": folds,
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }
    except Exception as error:
        return {"available": False, "reason": str(error)}


def explain_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> list[dict]:
    if X_test.empty:
        return []

    try:
        scoring = "f1_weighted" if problem_type == "classification" else "r2"
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring=scoring,
        )
    except Exception:
        return []

    importances = []
    for column, value in zip(X_test.columns, result.importances_mean):
        importances.append({"feature": column, "importance": round(float(value), 4)})

    top_importances = sorted(importances, key=lambda item: abs(item["importance"]), reverse=True)[:12]
    total_strength = sum(abs(item["importance"]) for item in top_importances) or 1.0
    enriched_importances = []
    for item in top_importances:
        score = float(item["importance"])
        absolute_score = abs(score)
        if absolute_score >= 0.03:
            strength = "High"
        elif absolute_score >= 0.01:
            strength = "Medium"
        else:
            strength = "Low"

        enriched_importances.append(
            {
                **item,
                "impact": "Helpful" if score >= 0 else "Weak signal",
                "impact_key": "helpful" if score >= 0 else "weak_signal",
                "strength": strength,
                "share": round(float(absolute_score / total_strength), 4),
                "interpretation": (
                    "Validation score dropped when this feature was shuffled."
                    if score >= 0
                    else "Shuffling this feature did not hurt validation, so it may be unstable."
                ),
            }
        )

    return enriched_importances


def build_explainability_studio(
    importances: list[dict],
    diagnostics: dict,
    problem_type: str,
    best_model: str,
    primary_metric: str,
    feature_plan: dict,
) -> dict:
    if not importances:
        return {
            "available": False,
            "method": "permutation_importance",
            "summary": "Feature impact could not be calculated for this run.",
            "cards": [],
            "features": [],
            "actions": [
                "Train again after adding more rows or selecting clearer feature columns.",
                "Check that the target has enough variation for validation scoring.",
            ],
            "review_focus": [],
        }

    helpful_features = [item for item in importances if item.get("impact_key") == "helpful"]
    weak_features = [item for item in importances if item.get("impact_key") == "weak_signal"]
    top_feature = importances[0]
    top_feature_name = top_feature.get("feature", "top feature")
    top_strength = top_feature.get("strength", "Low")
    top_score = abs(float(top_feature.get("importance") or 0))
    dropped_count = len(feature_plan.get("dropped_columns", []))
    leakage_count = len(feature_plan.get("leakage_columns", []))

    if top_strength == "High":
        summary = f"{best_model} is strongly influenced by {top_feature_name} for {primary_metric}."
    elif top_strength == "Medium":
        summary = f"{best_model} uses {top_feature_name}, but signal is shared across multiple features."
    else:
        summary = f"{best_model} has light feature signal; review data quality and add stronger predictors."

    if problem_type == "classification":
        weakest = (diagnostics.get("summary") or {}).get("weakest_class") or {}
        focus_value = weakest.get("label") or "All classes"
        focus_note = "weakest recall focus" if weakest else "class review"
    else:
        residual_summary = diagnostics.get("residual_summary") or {}
        focus_value = "Residual errors"
        focus_note = f"MAE {residual_summary.get('mae', 'n/a')}"

    actions = [
        f"Use {top_feature_name} first when explaining predictions to users.",
        "Review the top three drivers before accepting low-confidence predictions.",
    ]
    if weak_features:
        actions.append(f"Validate {weak_features[0]['feature']} because it behaved like a weak or noisy signal.")
    if dropped_count:
        actions.append(f"Keep the {dropped_count} dropped noisy feature(s) out of future training runs.")
    if leakage_count:
        actions.append(f"Do not re-add the {leakage_count} leakage-risk feature(s) unless they are fixed upstream.")

    return {
        "available": True,
        "method": "permutation_importance",
        "summary": summary,
        "cards": [
            {
                "label": "Top driver",
                "value": top_feature_name,
                "note": f"{top_strength} impact",
            },
            {
                "label": "Driver score",
                "value": round(float(top_score), 4),
                "note": primary_metric,
            },
            {
                "label": "Helpful features",
                "value": len(helpful_features),
                "note": "improved validation score",
            },
            {
                "label": "Weak signals",
                "value": len(weak_features),
                "note": "possible noise",
            },
            {
                "label": "Review focus",
                "value": focus_value,
                "note": focus_note,
            },
        ],
        "features": importances,
        "actions": actions,
        "review_focus": [item.get("feature") for item in helpful_features[:3]],
    }


def summarize_target(target: pd.Series, problem_type: str) -> dict:
    if problem_type == "classification":
        counts = target.astype(str).value_counts()
        total = max(int(counts.sum()), 1)
        return {
            "kind": "classification",
            "classes": [
                {
                    "label": str(label),
                    "count": int(count),
                    "share": round(float(count / total), 4),
                }
                for label, count in counts.items()
            ],
        }

    clean = pd.to_numeric(target, errors="coerce").dropna()
    return {
        "kind": "regression",
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "std": round(float(clean.std(ddof=0)), 4),
    }


def safe_metric(value, digits: int = 4):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return round(numeric, digits)


def get_pipeline_classes(pipeline: Pipeline):
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        model = getattr(pipeline, "named_steps", {}).get("model")
        classes = getattr(model, "classes_", None)
    return classes


def build_threshold_predictions(probabilities, classes, threshold: float):
    negative_class = classes[0]
    positive_class = classes[1]
    return np.where(probabilities[:, 1] >= threshold, positive_class, negative_class)


def score_binary_threshold(y_true, predictions, positive_class) -> dict:
    predicted_positive = pd.Series(predictions) == positive_class
    return {
        "accuracy": safe_metric(accuracy_score(y_true, predictions)),
        "precision": safe_metric(precision_score(y_true, predictions, pos_label=positive_class, zero_division=0)),
        "recall": safe_metric(recall_score(y_true, predictions, pos_label=positive_class, zero_division=0)),
        "f1": safe_metric(f1_score(y_true, predictions, pos_label=positive_class, zero_division=0)),
        "predicted_positive": int(predicted_positive.sum()),
        "predicted_positive_rate": safe_metric(predicted_positive.mean()),
    }


def build_threshold_analysis(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> dict:
    if problem_type != "classification":
        return {"available": False, "reason": "Threshold tuning is only available for classification."}
    if not hasattr(pipeline, "predict_proba"):
        return {"available": False, "reason": "The best model does not expose prediction probabilities."}

    classes = get_pipeline_classes(pipeline)
    if classes is None or len(classes) != 2:
        return {"available": False, "reason": "Threshold tuning is only available for binary classifiers."}

    try:
        probabilities = pipeline.predict_proba(X_test)
    except Exception as error:
        return {"available": False, "reason": f"Could not calculate threshold probabilities: {error}"}

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    positive_class = classes[1]
    curve = []
    for threshold in thresholds:
        predictions = build_threshold_predictions(probabilities, classes, threshold)
        metrics = score_binary_threshold(y_test, predictions, positive_class)
        curve.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    recommended = max(curve, key=lambda item: (item.get("f1") or -1, item.get("recall") or -1))
    default = next((item for item in curve if item["threshold"] == 0.5), curve[0])

    return {
        "available": True,
        "positive_class": serialize_prediction_value(positive_class),
        "negative_class": serialize_prediction_value(classes[0]),
        "recommended_threshold": recommended["threshold"],
        "default_threshold": 0.5,
        "recommended_metrics": recommended,
        "default_metrics": default,
        "curve": curve,
    }


def build_diagnostics(problem_type: str, y_test: pd.Series, predictions) -> dict:
    if problem_type == "classification":
        actual_series = pd.Series(y_test).reset_index(drop=True)
        predicted_series = pd.Series(predictions).reset_index(drop=True)
        correct_mask = actual_series == predicted_series
        error_count = int((~correct_mask).sum())
        total_count = int(len(actual_series))
        raw_labels = pd.Series(list(actual_series) + list(predicted_series)).dropna().unique().tolist()
        labels = sorted(raw_labels, key=lambda value: str(value))
        matrix_array = confusion_matrix(actual_series, predicted_series, labels=labels)
        matrix = matrix_array.tolist()
        report = classification_report(
            actual_series,
            predicted_series,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        class_report = [
            {
                "label": str(label),
                "precision": round(float(report.get(str(label), {}).get("precision", 0)), 4),
                "recall": round(float(report.get(str(label), {}).get("recall", 0)), 4),
                "f1": round(float(report.get(str(label), {}).get("f1-score", 0)), 4),
                "support": int(report.get(str(label), {}).get("support", 0)),
            }
            for label in labels
        ]
        off_diagonal = matrix_array.copy()
        np.fill_diagonal(off_diagonal, 0)
        top_confusion = None
        if off_diagonal.size and int(off_diagonal.max()) > 0:
            row_index, column_index = np.unravel_index(int(off_diagonal.argmax()), off_diagonal.shape)
            top_confusion = {
                "actual": str(labels[row_index]),
                "predicted": str(labels[column_index]),
                "count": int(off_diagonal[row_index, column_index]),
            }

        weakest_class = None
        supported_report = [item for item in class_report if item["support"] > 0]
        if supported_report:
            weakest_class = min(supported_report, key=lambda item: (item["recall"], item["f1"]))

        mistake_samples = [
            {
                "row": index + 1,
                "actual": str(actual),
                "predicted": str(predicted),
            }
            for index, (actual, predicted, correct) in enumerate(zip(actual_series, predicted_series, correct_mask))
            if not bool(correct)
        ][:20]
        prediction_counts = predicted_series.astype(str).value_counts()

        return {
            "kind": "classification",
            "labels": [str(label) for label in labels],
            "confusion_matrix": matrix,
            "class_report": class_report,
            "summary": {
                "holdout_rows": total_count,
                "correct": int(correct_mask.sum()),
                "errors": error_count,
                "error_rate": safe_metric(error_count / max(total_count, 1)),
                "accuracy": safe_metric(accuracy_score(actual_series, predicted_series)),
                "top_confusion": top_confusion,
                "weakest_class": weakest_class,
            },
            "prediction_distribution": [
                {"label": str(label), "count": int(count)}
                for label, count in prediction_counts.items()
            ],
            "mistake_samples": mistake_samples,
        }

    actual = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors="coerce")
    predicted = pd.to_numeric(pd.Series(predictions).reset_index(drop=True), errors="coerce")
    residuals = actual - predicted
    abs_errors = residuals.abs()
    percent_errors = abs_errors / actual.abs().replace(0, np.nan)
    error_frame = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "residual": residuals,
            "abs_error": abs_errors,
            "percent_error": percent_errors,
        }
    ).dropna(subset=["actual", "predicted", "residual"])
    outliers = error_frame.sort_values("abs_error", ascending=False).head(20)
    samples = [
        {
            "row": int(index + 1),
            "actual": round(float(actual_value), 4),
            "predicted": round(float(predicted_value), 4),
            "residual": round(float(residual), 4),
            "abs_error": round(float(abs_error), 4),
        }
        for index, actual_value, predicted_value, residual, abs_error in [
            (idx, row.actual, row.predicted, row.residual, row.abs_error)
            for idx, row in outliers.iterrows()
        ]
    ]
    valid_percent_errors = percent_errors.dropna()

    return {
        "kind": "regression",
        "residual_summary": {
            "holdout_rows": int(len(error_frame)),
            "r2": safe_metric(r2_score(actual, predicted)),
            "mae": safe_metric(mean_absolute_error(actual, predicted)),
            "rmse": safe_metric(np.sqrt(mean_squared_error(actual, predicted))),
            "mape": safe_metric(valid_percent_errors.mean()) if not valid_percent_errors.empty else None,
            "mean_error": safe_metric(residuals.mean()),
            "median_error": safe_metric(residuals.median()),
            "largest_over_prediction": safe_metric(residuals.min()),
            "largest_under_prediction": safe_metric(residuals.max()),
        },
        "samples": samples,
    }


def build_next_actions(
    problem_type: str,
    diagnostics: dict,
    target_summary: dict,
    feature_plan: dict,
    leaderboard: list[dict],
) -> list[str]:
    actions = []

    if problem_type == "classification":
        summary = diagnostics.get("summary", {})
        error_rate = summary.get("error_rate")
        if isinstance(error_rate, (int, float)) and error_rate > 0.2:
            actions.append("Review the holdout mistakes first; more labeled rows or stronger predictors are needed.")

        weakest = summary.get("weakest_class")
        if weakest and weakest.get("recall", 1) < 0.75:
            actions.append(f"Collect or balance more examples for class '{weakest['label']}' to improve recall.")

        confusion = summary.get("top_confusion")
        if confusion:
            actions.append(
                f"Add features that separate actual '{confusion['actual']}' from predicted '{confusion['predicted']}'."
            )

        classes = target_summary.get("classes", []) if target_summary else []
        if classes and classes[0].get("share", 0) > 0.7:
            actions.append("Try class balancing or threshold tuning because one class dominates the target.")

    else:
        residuals = diagnostics.get("residual_summary", {})
        r2 = residuals.get("r2")
        mape = residuals.get("mape")
        if isinstance(r2, (int, float)) and r2 < 0.5:
            actions.append("Add more business driver columns; the current features explain limited target variance.")
        if isinstance(mape, (int, float)) and mape > 0.2:
            actions.append("Inspect high-error rows and outliers before trusting numeric forecasts.")
        if diagnostics.get("samples"):
            actions.append("Use the residual outlier table to find rows where the model is over or under predicting.")

    leakage_columns = feature_plan.get("leakage_columns", [])
    if leakage_columns:
        actions.append("Review auto-excluded leakage columns before trusting unusually high accuracy.")

    dropped_columns = feature_plan.get("dropped_columns", [])
    if dropped_columns:
        actions.append("Keep identifier-like columns out of training unless they carry real behavioral signal.")

    engineered_features = feature_plan.get("generated_features", [])
    if engineered_features:
        actions.append("Date columns are now split into reusable year/month/day/elapsed features.")

    successful = [item for item in leaderboard if isinstance(item.get("rank_score"), (int, float))]
    if len(successful) >= 2 and abs(successful[0]["rank_score"] - successful[1]["rank_score"]) <= 0.02:
        actions.append("Top models are very close; retrain after adding rows to confirm the winner.")

    return actions[:5]


def build_quality_insights(
    problem_type: str,
    y: pd.Series,
    leaderboard: list[dict],
    baseline: dict,
    trained_count: int,
    failed_count: int,
    feature_plan: dict,
    class_balance: dict | None = None,
) -> list[str]:
    insights = []
    successful = [item for item in leaderboard if isinstance(item.get("rank_score"), (int, float))]
    primary_metric = "f1_weighted" if problem_type == "classification" else "r2"
    best = successful[0] if successful else None

    if best:
        baseline_score = baseline.get(primary_metric)
        if isinstance(baseline_score, (int, float)):
            delta = round(float(best["rank_score"] - baseline_score), 4)
            if delta > 0:
                insights.append(
                    f"Best model improves {primary_metric} by {delta} over baseline."
                )
            else:
                insights.append(
                    "Best model is not beating the baseline yet; the target may need better predictors."
                )

        if problem_type == "classification" and best["rank_score"] < 0.7:
            insights.append(
                "Classification quality is still low; add stronger features or more labeled rows."
            )
        if problem_type == "regression" and best["rank_score"] < 0.5:
            insights.append(
                "Regression signal is weak; check target noise, outliers, and missing drivers."
            )

    if problem_type == "classification":
        class_share = y.astype(str).value_counts(normalize=True)
        if not class_share.empty and float(class_share.iloc[0]) > 0.7:
            insights.append(
                f"Class balance warning: top class is {round(float(class_share.iloc[0]) * 100, 1)}% of rows."
            )
        if class_balance and class_balance.get("severity") not in {None, "balanced"}:
            insights.append(
                f"Auto accuracy improver trained balanced and oversampled candidates for {class_balance['severity']} class imbalance."
            )

    if len(y) < 200:
        insights.append("Dataset is small; cross-validation can move a lot with new rows.")

    leakage_columns = feature_plan.get("leakage_columns", [])
    if leakage_columns:
        names = ", ".join(item["name"] for item in leakage_columns[:3])
        suffix = "..." if len(leakage_columns) > 3 else ""
        insights.append(f"Auto-excluded possible data leakage feature(s): {names}{suffix}.")

    dropped_columns = feature_plan.get("dropped_columns", [])
    if dropped_columns:
        names = ", ".join(item["name"] for item in dropped_columns[:3])
        suffix = "..." if len(dropped_columns) > 3 else ""
        insights.append(f"Dropped non-useful feature(s): {names}{suffix}.")

    date_columns = feature_plan.get("date_columns", [])
    if date_columns:
        names = ", ".join(date_columns[:3])
        suffix = "..." if len(date_columns) > 3 else ""
        insights.append(f"Engineered date features from: {names}{suffix}.")

    if failed_count:
        insights.append(f"{failed_count} candidate model(s) failed and were kept visible in the leaderboard.")

    if trained_count and best:
        insights.append(f"{trained_count} model(s) trained successfully; top model is {best['model_name']}.")

    return insights[:5]


def build_model_pipeline(model_X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(model_X)),
            ("model", estimator),
        ]
    )


def tuning_grid_for_model(model_name: str, problem_type: str) -> list[dict]:
    normalized = model_name.lower()
    nested_prefix = "model__estimator__" if "oversampled" in normalized else "model__"

    if problem_type == "classification":
        if "logistic" in normalized:
            return [{f"{nested_prefix}C": [0.3, 1.0, 3.0]}]
        if "random forest" in normalized or "subsample forest" in normalized:
            return [
                {
                    f"{nested_prefix}n_estimators": [180, 260],
                    f"{nested_prefix}max_depth": [None, 10],
                    f"{nested_prefix}min_samples_leaf": [1, 3],
                }
            ]
        if "extra trees" in normalized:
            return [
                {
                    f"{nested_prefix}n_estimators": [180, 260],
                    f"{nested_prefix}max_depth": [None, 12],
                    f"{nested_prefix}min_samples_leaf": [1, 3],
                }
            ]
        if "gradient boosting" in normalized and "histogram" not in normalized:
            return [
                {
                    f"{nested_prefix}n_estimators": [120, 180],
                    f"{nested_prefix}learning_rate": [0.035, 0.06],
                    f"{nested_prefix}max_depth": [2, 3],
                }
            ]
        if "histogram gradient" in normalized:
            return [{"model__max_iter": [120, 200], "model__learning_rate": [0.045, 0.08]}]
        if "adaboost" in normalized:
            return [{"model__n_estimators": [70, 120, 170], "model__learning_rate": [0.05, 0.15, 0.35]}]
        if "decision tree" in normalized:
            return [{"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3, 5]}]
        return []

    if "ridge" in normalized:
        return [{"model__alpha": [0.3, 1.0, 3.0, 10.0]}]
    if "lasso" in normalized:
        return [{"model__alpha": [0.001, 0.01, 0.05, 0.1]}]
    if "elastic" in normalized:
        return [{"model__alpha": [0.001, 0.01, 0.05], "model__l1_ratio": [0.2, 0.5, 0.8]}]
    if "random forest" in normalized:
        return [{"model__n_estimators": [180, 260], "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]}]
    if "extra trees" in normalized:
        return [{"model__n_estimators": [180, 260], "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]}]
    if "gradient boosting" in normalized:
        return [{"model__n_estimators": [120, 180], "model__learning_rate": [0.035, 0.06], "model__max_depth": [2, 3]}]
    if "histogram gradient" in normalized:
        return [{"model__max_iter": [120, 200], "model__learning_rate": [0.045, 0.08]}]
    if "adaboost" in normalized:
        return [{"model__n_estimators": [70, 120, 170], "model__learning_rate": [0.05, 0.15, 0.35]}]
    if "support vector" in normalized:
        return [{"model__C": [3.0, 10.0, 30.0], "model__epsilon": [0.05, 0.1, 0.2]}]
    if "decision tree" in normalized:
        return [{"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3, 5]}]
    return []


def serialize_tuning_params(params: dict) -> dict:
    cleaned = {}
    for key, value in params.items():
        short_key = key.replace("model__estimator__", "").replace("model__", "")
        cleaned[short_key] = value
    return cleaned


def tune_top_models(
    problem_type: str,
    model_X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    leaderboard: list[dict],
    estimator_by_name: dict,
    max_models: int = 3,
    max_trials: int = 8,
) -> dict:
    successful = [item for item in leaderboard if isinstance(item.get("rank_score"), (int, float))]
    selected = successful[:max_models]
    results = []
    best_internal = None

    for base in selected:
        model_name = base["model_name"]
        estimator = estimator_by_name.get(model_name)
        grid = tuning_grid_for_model(model_name, problem_type)
        if estimator is None or not grid:
            results.append(
                {
                    "model_name": model_name,
                    "status": "skipped",
                    "reason": "No safe tuning grid is configured for this model.",
                    "base_score": base.get("rank_score"),
                    "tuned_score": None,
                    "improvement": None,
                    "trials": 0,
                }
            )
            continue

        best_trial = None
        trial_count = 0
        for params in list(ParameterGrid(grid))[:max_trials]:
            trial_count += 1
            pipeline = build_model_pipeline(model_X, clone(estimator))
            try:
                pipeline.set_params(**params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_test)
                    metrics = score_model(problem_type, y_test, predictions)
                holdout_score = ranking_score(problem_type, metrics)
            except Exception as error:
                if best_trial is None:
                    best_trial = {"error": str(error)}
                continue

            if best_trial is None or holdout_score > best_trial["holdout_score"]:
                best_trial = {
                    "pipeline": pipeline,
                    "params": params,
                    "metrics": metrics,
                    "holdout_score": float(holdout_score),
                }

        if not best_trial or "pipeline" not in best_trial:
            results.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "reason": best_trial.get("error", "No tuning trial completed.") if best_trial else "No tuning trial completed.",
                    "base_score": base.get("rank_score"),
                    "tuned_score": None,
                    "improvement": None,
                    "trials": trial_count,
                }
            )
            continue

        cv_score = cross_validation_score(best_trial["pipeline"], model_X, y, problem_type)
        tuned_score = cv_score["mean"] if cv_score.get("available") else best_trial["holdout_score"]
        improvement = round(float(tuned_score - base.get("rank_score", 0)), 4)
        status = "improved" if improvement > 0.001 else "no_gain"
        public_result = {
            "model_name": model_name,
            "status": status,
            "base_score": base.get("rank_score"),
            "tuned_score": round(float(tuned_score), 4),
            "holdout_score": round(float(best_trial["holdout_score"]), 4),
            "improvement": improvement,
            "trials": trial_count,
            "best_params": serialize_tuning_params(best_trial["params"]),
            "metrics": best_trial["metrics"],
            "cross_validation": cv_score,
        }
        results.append(public_result)

        if best_internal is None or tuned_score > best_internal["rank_score"]:
            best_internal = {
                "model_name": f"{model_name} Tuned",
                "base_model_name": model_name,
                "pipeline": best_trial["pipeline"],
                "metrics": best_trial["metrics"],
                "rank_score": float(tuned_score),
                "cross_validation": cv_score,
                "params": best_trial["params"],
                "public_result": public_result,
            }

    improved = [item for item in results if item.get("status") == "improved"]
    return {
        "available": bool(results),
        "selected_models": [item["model_name"] for item in selected],
        "results": results,
        "best_internal": best_internal,
        "tuned_count": len([item for item in results if item.get("status") in {"improved", "no_gain"}]),
        "improved_count": len(improved),
    }


def build_accuracy_improver(
    problem_type: str,
    leaderboard: list[dict],
    baseline: dict,
    diagnostics: dict,
    target_summary: dict,
    feature_plan: dict,
    class_balance: dict | None = None,
) -> dict:
    successful = [item for item in leaderboard if isinstance(item.get("rank_score"), (int, float))]
    best = successful[0] if successful else None
    if not best:
        return {
            "available": False,
            "status": "blocked",
            "summary": "No trained model is available for improvement analysis.",
            "checks": [],
            "actions": ["Fix failed candidate errors, then train again."],
        }

    checks = []
    actions = []

    if problem_type == "classification":
        metrics = best.get("metrics", {})
        f1 = metrics.get("f1_weighted")
        accuracy = metrics.get("accuracy")
        balanced_accuracy = metrics.get("balanced_accuracy")
        baseline_f1 = baseline.get("f1_weighted")
        baseline_delta = None
        if isinstance(f1, (int, float)) and isinstance(baseline_f1, (int, float)):
            baseline_delta = round(float(f1 - baseline_f1), 4)

        if isinstance(baseline_delta, (int, float)) and baseline_delta <= 0.02:
            checks.append(
                {
                    "label": "Model signal",
                    "status": "warning",
                    "detail": f"Best weighted F1 is only {baseline_delta} above baseline.",
                }
            )
            actions.append("Add stronger predictor columns or clean target noise; the model is close to majority-class guessing.")
        else:
            checks.append(
                {
                    "label": "Model signal",
                    "status": "good",
                    "detail": f"Best model beats baseline by {baseline_delta if baseline_delta is not None else 'n/a'} weighted F1.",
                }
            )

        if class_balance and class_balance.get("available"):
            severity = class_balance.get("severity", "balanced")
            detail = (
                f"Top class {class_balance.get('majority_class')} is "
                f"{round(float(class_balance.get('majority_share', 0)) * 100, 1)}%; "
                f"minority class {class_balance.get('minority_class')} is "
                f"{round(float(class_balance.get('minority_share', 0)) * 100, 1)}%."
            )
            checks.append(
                {
                    "label": "Class balance",
                    "status": "good" if severity == "balanced" else "warning",
                    "detail": detail,
                }
            )
            if severity != "balanced":
                actions.append("Use the balanced/oversampled models near the top of the leaderboard when minority recall matters.")

        if isinstance(accuracy, (int, float)) and isinstance(balanced_accuracy, (int, float)):
            gap = round(float(accuracy - balanced_accuracy), 4)
            if gap > 0.08:
                checks.append(
                    {
                        "label": "Accuracy gap",
                        "status": "warning",
                        "detail": f"Accuracy is {accuracy}, but balanced accuracy is {balanced_accuracy}.",
                    }
                )
                actions.append("Prefer balanced accuracy/F1 over raw accuracy because raw accuracy is inflated by common classes.")
            else:
                checks.append(
                    {
                        "label": "Accuracy gap",
                        "status": "good",
                        "detail": "Raw accuracy and balanced accuracy are close.",
                    }
                )

        weakest = diagnostics.get("summary", {}).get("weakest_class")
        if weakest and weakest.get("recall", 1) < 0.7:
            checks.append(
                {
                    "label": "Weakest class recall",
                    "status": "warning",
                    "detail": f"Class {weakest['label']} recall is {weakest['recall']}.",
                }
            )
            actions.append(f"Collect more rows or add separation features for class '{weakest['label']}'.")

        leakage_columns = feature_plan.get("leakage_columns", [])
        if leakage_columns:
            checks.append(
                {
                    "label": "Leakage guard",
                    "status": "warning",
                    "detail": f"Auto-excluded {len(leakage_columns)} possible target-leaking feature(s).",
                }
            )
            actions.append("Keep leakage columns excluded unless they are genuinely available before prediction time.")

        dropped_columns = feature_plan.get("dropped_columns", [])
        if dropped_columns:
            checks.append(
                {
                    "label": "Feature cleanup",
                    "status": "good",
                    "detail": f"Removed {len(dropped_columns)} noisy/ID-like feature(s) before training.",
                }
            )

        status = "good" if isinstance(f1, (int, float)) and f1 >= 0.75 and not any(item["status"] == "warning" for item in checks if item["label"] != "Leakage guard") else "needs_attention"
        summary = (
            f"Best classifier is {best['model_name']} with weighted F1 {f1} and accuracy {accuracy}."
        )
        return {
            "available": True,
            "kind": "classification",
            "status": status,
            "summary": summary,
            "best_model": best["model_name"],
            "checks": checks[:6],
            "actions": list(dict.fromkeys(actions))[:5],
        }

    metrics = best.get("metrics", {})
    r2 = metrics.get("r2")
    rmse = metrics.get("rmse")
    baseline_r2 = baseline.get("r2")
    if isinstance(r2, (int, float)) and isinstance(baseline_r2, (int, float)) and r2 <= baseline_r2 + 0.05:
        checks.append(
            {
                "label": "Model signal",
                "status": "warning",
                "detail": "Best regression model is close to the simple baseline.",
            }
        )
        actions.append("Add stronger numeric drivers, reduce target noise, or segment the dataset before retraining.")
    else:
        checks.append(
            {
                "label": "Model signal",
                "status": "good",
                "detail": "Best model improves over the mean baseline.",
            }
        )

    return {
        "available": True,
        "kind": "regression",
        "status": "good" if isinstance(r2, (int, float)) and r2 >= 0.6 else "needs_attention",
        "summary": f"Best regressor is {best['model_name']} with R2 {r2} and RMSE {rmse}.",
        "best_model": best["model_name"],
        "checks": checks,
        "actions": actions[:5],
    }


def train_models(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = "auto",
    dataset_id: str | None = None,
    feature_columns: list[str] | None = None,
    run_context: dict | None = None,
) -> dict:
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found.")

    clean_df = df.dropna(subset=[target_column]).copy()
    if clean_df.shape[0] < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 rows with target values.")

    y = clean_df[target_column]
    selected_problem_type = infer_problem_type(y) if problem_type == "auto" else problem_type
    if selected_problem_type not in {"classification", "regression"}:
        raise HTTPException(status_code=400, detail="Problem type must be auto, classification, or regression.")
    validate_target_for_problem(y, selected_problem_type, target_column)

    selected_features = resolve_feature_columns(clean_df, target_column, feature_columns)
    X = clean_df[selected_features]

    if X.empty:
        raise HTTPException(status_code=400, detail="Need at least one feature column.")

    raw_feature_columns = X.columns.tolist()
    feature_plan = build_feature_plan(X, y, target_column, selected_problem_type)
    model_X = apply_feature_plan(X, feature_plan)

    if model_X.empty:
        raise HTTPException(
            status_code=400,
            detail="No usable feature columns remain after AutoML cleanup.",
        )

    class_balance = analyze_class_balance(y) if selected_problem_type == "classification" else None

    stratify = y if selected_problem_type == "classification" and y.nunique() > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            model_X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            model_X,
            y,
            test_size=0.2,
            random_state=42,
        )

    baseline = baseline_metrics(selected_problem_type, y_train, y_test)
    model_specs = get_model_specs(selected_problem_type)
    estimator_by_name = {item["name"]: item["estimator"] for item in model_specs}
    leaderboard = []
    best_pipeline = None
    best_score = -np.inf
    best_name = ""
    best_metrics = {}
    trained_count = 0
    failed_count = 0

    for spec in model_specs:
        model_name = spec["name"]
        estimator = spec["estimator"]
        pipeline = build_model_pipeline(model_X, estimator)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                metrics = score_model(selected_problem_type, y_test, predictions)
                cv_score = cross_validation_score(pipeline, model_X, y, selected_problem_type)
            score = cv_score["mean"] if cv_score.get("available") else ranking_score(selected_problem_type, metrics)
        except Exception as error:
            failed_count += 1
            leaderboard.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "quality_label": "failed",
                    "metrics": {},
                    "cross_validation": {"available": False, "reason": "Training failed."},
                    "rank_score": None,
                    "error": str(error),
                }
            )
            continue

        trained_count += 1

        leaderboard.append(
            {
                "model_name": model_name,
                "status": "trained",
                "quality_label": quality_label(selected_problem_type, float(score)),
                "metrics": metrics,
                "cross_validation": cv_score,
                "rank_score": round(float(score), 4),
            }
        )

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_name = model_name
            best_metrics = metrics

    if best_pipeline is None:
        raise HTTPException(status_code=400, detail="No candidate models could be trained for this dataset.")

    leaderboard = sorted(
        leaderboard,
        key=lambda item: item["rank_score"] if isinstance(item.get("rank_score"), (int, float)) else -np.inf,
        reverse=True,
    )
    tuning_studio = tune_top_models(
        selected_problem_type,
        model_X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        leaderboard,
        estimator_by_name,
    )
    best_tuned = tuning_studio.get("best_internal")
    saved_tuned_model = False
    if best_tuned and best_tuned["rank_score"] > float(leaderboard[0].get("rank_score") or -np.inf) + 0.001:
        saved_tuned_model = True
        best_pipeline = best_tuned["pipeline"]
        best_name = best_tuned["model_name"]
        best_metrics = best_tuned["metrics"]
        best_score = best_tuned["rank_score"]
        leaderboard.insert(
            0,
            {
                "model_name": best_name,
                "status": "tuned",
                "quality_label": quality_label(selected_problem_type, float(best_score)),
                "metrics": best_metrics,
                "cross_validation": best_tuned["cross_validation"],
                "rank_score": round(float(best_score), 4),
                "tuned_from": best_tuned["base_model_name"],
                "best_params": serialize_tuning_params(best_tuned["params"]),
            },
        )
        leaderboard = sorted(
            leaderboard,
            key=lambda item: item["rank_score"] if isinstance(item.get("rank_score"), (int, float)) else -np.inf,
            reverse=True,
        )

    tuning_studio_public = {
        "available": tuning_studio.get("available", False),
        "selected_models": tuning_studio.get("selected_models", []),
        "results": tuning_studio.get("results", []),
        "tuned_count": tuning_studio.get("tuned_count", 0),
        "improved_count": tuning_studio.get("improved_count", 0),
        "saved_tuned_model": saved_tuned_model,
        "best_tuned_model": best_name if saved_tuned_model else None,
    }
    model_id = uuid4().hex
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_id}.joblib"

    artifact = {
        "model_id": model_id,
        "pipeline": best_pipeline,
        "target_column": target_column,
        "problem_type": selected_problem_type,
        "feature_columns": raw_feature_columns,
        "model_feature_columns": model_X.columns.tolist(),
        "feature_plan": feature_plan,
        "model_name": best_name,
    }
    joblib.dump(artifact, model_path)
    best_predictions = best_pipeline.predict(X_test)
    diagnostics = build_diagnostics(selected_problem_type, y_test, best_predictions)
    threshold_analysis = build_threshold_analysis(best_pipeline, X_test, y_test, selected_problem_type)
    target_summary = summarize_target(y, selected_problem_type)
    quality_insights = build_quality_insights(
        selected_problem_type,
        y,
        leaderboard,
        baseline,
        trained_count,
        failed_count,
        feature_plan,
        class_balance,
    )
    next_actions = build_next_actions(
        selected_problem_type,
        diagnostics,
        target_summary,
        feature_plan,
        leaderboard,
    )
    accuracy_improver = build_accuracy_improver(
        selected_problem_type,
        leaderboard,
        baseline,
        diagnostics,
        target_summary,
        feature_plan,
        class_balance,
    )
    primary_metric = "f1_weighted" if selected_problem_type == "classification" else "r2"
    explainability = explain_model(best_pipeline, X_test, y_test, selected_problem_type)
    explainability_studio = build_explainability_studio(
        explainability,
        diagnostics,
        selected_problem_type,
        best_name,
        primary_metric,
        feature_plan,
    )
    metric_note = (
        "Accuracy is shown for every classifier, but ranking uses weighted F1 so class imbalance is handled more fairly."
        if selected_problem_type == "classification"
        else "Regression does not have an accuracy metric; ranking uses R2 and every model also shows MAE/RMSE."
    )

    result = {
        "model_id": model_id,
        "problem_type": selected_problem_type,
        "target_column": target_column,
        "best_model": best_name,
        "best_metrics": best_metrics,
        "baseline_metrics": baseline,
        "primary_metric": primary_metric,
        "metric_note": metric_note,
        "tuning_studio": tuning_studio_public,
        "quality_insights": quality_insights,
        "next_actions": next_actions,
        "accuracy_improver": accuracy_improver,
        "class_balance": class_balance,
        "run_context": run_context or {},
        "leaderboard": leaderboard,
        "training_summary": {
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "raw_feature_count": int(X.shape[1]),
            "feature_count": int(model_X.shape[1]),
            "candidate_models": len(leaderboard),
            "trained_models": trained_count,
            "failed_models": failed_count,
            "feature_columns": raw_feature_columns,
            "model_feature_columns": model_X.columns.tolist(),
            "dropped_features": feature_plan["dropped_columns"],
            "leakage_features": feature_plan.get("leakage_columns", []),
            "engineered_features": feature_plan["generated_features"],
            "selected_feature_count": len(selected_features),
            "excluded_feature_count": max((len(clean_df.columns) - 1) - len(selected_features), 0),
            "class_balance_severity": class_balance.get("severity") if class_balance else None,
            "balanced_candidate_count": len([item for item in leaderboard if "Balanced" in item.get("model_name", "") or "Oversampled" in item.get("model_name", "")]),
            "tuned_models": tuning_studio_public["tuned_count"],
            "improved_tuned_models": tuning_studio_public["improved_count"],
            "saved_tuned_model": saved_tuned_model,
        },
        "target_summary": target_summary,
        "diagnostics": diagnostics,
        "threshold_analysis": threshold_analysis,
        "explainability": explainability,
        "explainability_studio": explainability_studio,
        "prediction_api": f"/api/predict/{model_id}",
    }

    register_model(
        {
            "experiment_id": model_id,
            "experiment_label": f"{target_column} {selected_problem_type} run",
            "model_id": model_id,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "problem_type": selected_problem_type,
            "best_model": best_name,
            "primary_metric": result["primary_metric"],
            "rank_score": leaderboard[0]["rank_score"] if leaderboard else None,
            "quality_label": leaderboard[0].get("quality_label") if leaderboard else None,
            "holdout_metrics": best_metrics,
            "baseline_metrics": baseline,
            "candidate_models": len(leaderboard),
            "trained_models": trained_count,
            "failed_models": failed_count,
            "raw_feature_count": int(X.shape[1]),
            "model_feature_count": int(model_X.shape[1]),
            "selected_feature_count": len(selected_features),
            "excluded_feature_count": max((len(clean_df.columns) - 1) - len(selected_features), 0),
            "dropped_feature_count": len(feature_plan["dropped_columns"]),
            "leakage_feature_count": len(feature_plan.get("leakage_columns", [])),
            "leakage_features": feature_plan.get("leakage_columns", []),
            "engineered_feature_count": len(feature_plan["generated_features"]),
            "recommended_threshold": threshold_analysis.get("recommended_threshold") if threshold_analysis.get("available") else None,
            "tuning_studio": tuning_studio_public,
            "class_balance": class_balance,
            "accuracy_improver": accuracy_improver,
            "explainability_studio": explainability_studio,
            "top_features": explainability[:8],
            "run_context": run_context or {},
            "training_summary": result["training_summary"],
            "leaderboard_snapshot": [
                {
                    "model_name": item.get("model_name"),
                    "status": item.get("status"),
                    "quality_label": item.get("quality_label"),
                    "rank_score": item.get("rank_score"),
                    "metrics": item.get("metrics"),
                    "cross_validation": item.get("cross_validation"),
                    "error": item.get("error"),
                    "tuned_from": item.get("tuned_from"),
                }
                for item in leaderboard[:12]
            ],
            "next_actions": next_actions,
            "prediction_api": result["prediction_api"],
            "created_at": utc_now(),
        }
    )

    return result


def prepare_prediction_frame(artifact: dict, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise HTTPException(status_code=400, detail="Prediction rows cannot be empty.")

    input_df = pd.DataFrame(rows)
    missing_columns = [
        column for column in artifact["feature_columns"] if column not in input_df.columns
    ]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_columns}",
        )

    input_df = input_df[artifact["feature_columns"]]
    if artifact.get("feature_plan"):
        input_df = apply_feature_plan(input_df, artifact["feature_plan"])
    return input_df


def serialize_prediction_value(value):
    return value.item() if hasattr(value, "item") else value


def predict_rows(artifact: dict, rows: list[dict]) -> list:
    input_df = prepare_prediction_frame(artifact, rows)
    predictions = artifact["pipeline"].predict(input_df)

    return [serialize_prediction_value(value) for value in predictions]


def normalize_threshold(threshold: float | None) -> float | None:
    if threshold is None:
        return None
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Threshold must be a number between 0 and 1.")
    if value <= 0 or value >= 1:
        raise HTTPException(status_code=400, detail="Threshold must be greater than 0 and less than 1.")
    return value


def confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "unavailable"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def build_prediction_confidence_summary(details: list[dict], problem_type: str) -> dict:
    distribution = pd.Series([str(item.get("prediction")) for item in details]).value_counts()
    confidence_values = [
        float(item["confidence"])
        for item in details
        if isinstance(item.get("confidence"), (int, float))
    ]
    low_confidence_rows = [
        item["row"]
        for item in details
        if item.get("confidence_band") == "low"
    ]

    if not confidence_values:
        return {
            "available": False,
            "problem_type": problem_type,
            "total_rows": len(details),
            "prediction_distribution": [
                {"label": label, "count": int(count)}
                for label, count in distribution.items()
            ],
            "message": "Confidence is available for classification models with probability output.",
        }

    average_confidence = round(float(np.mean(confidence_values)), 4)
    minimum_confidence = round(float(np.min(confidence_values)), 4)
    high_count = len([value for value in confidence_values if value >= 0.75])
    medium_count = len([value for value in confidence_values if 0.55 <= value < 0.75])
    low_count = len([value for value in confidence_values if value < 0.55])

    if low_count:
        recommendation = "Review low-confidence rows before using these predictions in production."
    elif average_confidence < 0.75:
        recommendation = "Predictions are usable, but review medium-confidence rows for business-critical decisions."
    else:
        recommendation = "Prediction confidence is strong for this batch."

    return {
        "available": True,
        "problem_type": problem_type,
        "total_rows": len(details),
        "average_confidence": average_confidence,
        "minimum_confidence": minimum_confidence,
        "high_confidence_rows": high_count,
        "medium_confidence_rows": medium_count,
        "low_confidence_rows": low_count,
        "low_confidence_row_numbers": low_confidence_rows[:25],
        "prediction_distribution": [
            {"label": label, "count": int(count)}
            for label, count in distribution.items()
        ],
        "recommendation": recommendation,
    }


def predict_rows_with_details(artifact: dict, rows: list[dict], threshold: float | None = None) -> dict:
    input_df = prepare_prediction_frame(artifact, rows)
    pipeline = artifact["pipeline"]
    threshold_value = normalize_threshold(threshold)
    details = []
    probabilities = None
    classes = None

    if artifact.get("problem_type") == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(input_df)
            classes = getattr(pipeline, "classes_", None)
            if classes is None:
                classes = getattr(pipeline.named_steps.get("model"), "classes_", None)
        except Exception:
            probabilities = None
            classes = None

    if threshold_value is not None and probabilities is not None and classes is not None and len(classes) == 2:
        predictions = build_threshold_predictions(probabilities, classes, threshold_value)
    else:
        predictions = pipeline.predict(input_df)

    serialized_predictions = [serialize_prediction_value(value) for value in predictions]

    for index, prediction in enumerate(serialized_predictions):
        detail = {
            "row": index + 1,
            "prediction": prediction,
        }
        if probabilities is not None and classes is not None:
            row_probabilities = {
                str(label): round(float(probability), 4)
                for label, probability in zip(classes, probabilities[index])
            }
            confidence = row_probabilities.get(str(prediction))
            if confidence is None and row_probabilities:
                confidence = max(row_probabilities.values())
            detail["confidence"] = confidence
            detail["confidence_band"] = confidence_band(confidence)
            detail["low_confidence"] = bool(confidence is not None and confidence < 0.55)
            detail["probabilities"] = row_probabilities
            if threshold_value is not None and len(classes) == 2:
                detail["threshold_applied"] = threshold_value
                detail["positive_class"] = str(classes[1])
        details.append(detail)

    result = {
        "predictions": serialized_predictions,
        "prediction_details": details,
        "confidence_summary": build_prediction_confidence_summary(details, artifact.get("problem_type")),
    }
    if threshold_value is not None:
        result["threshold"] = threshold_value
    return result
