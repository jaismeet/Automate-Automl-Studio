from typing import Any


def _numeric_score(record: dict) -> float | None:
    score = record.get("rank_score")
    return float(score) if isinstance(score, (int, float)) else None


def _group_key(record: dict) -> tuple[Any, Any, Any]:
    return (
        record.get("target_column"),
        record.get("problem_type"),
        record.get("primary_metric"),
    )


def _summary(record: dict, champion_score: float | None = None, rank: int | None = None) -> dict:
    score = _numeric_score(record)
    delta = round(score - champion_score, 4) if score is not None and champion_score is not None else None
    return {
        "model_id": record.get("model_id"),
        "dataset_id": record.get("dataset_id"),
        "best_model": record.get("best_model"),
        "target_column": record.get("target_column"),
        "problem_type": record.get("problem_type"),
        "primary_metric": record.get("primary_metric"),
        "rank_score": score,
        "score_delta_vs_champion": delta,
        "quality_label": record.get("quality_label"),
        "prediction_api": record.get("prediction_api"),
        "created_at": record.get("created_at"),
        "rank": rank,
        "run_context": record.get("run_context") or {},
    }


def _promotion_for_group(selected: dict, promotions: list[dict] | None) -> dict | None:
    if not promotions:
        return None
    target, problem, metric = _group_key(selected)
    return next(
        (
            promotion
            for promotion in promotions
            if promotion.get("target_column") == target
            and promotion.get("problem_type") == problem
            and promotion.get("primary_metric") == metric
            and promotion.get("status") == "active"
        ),
        None,
    )


def _actions(
    selected: dict,
    champion: dict | None,
    selected_rank: int | None,
    group_size: int,
    production: dict | None = None,
) -> list[str]:
    if not champion:
        return ["Train another model with the same target to create a champion/challenger comparison."]

    if production and selected.get("model_id") == production.get("model_id"):
        return [
            "This model is manually promoted to production. Keep monitoring drift, confidence, and alerts.",
            "Retrain from drift when recent labeled data shows a stronger challenger.",
        ]

    if selected.get("model_id") == champion.get("model_id"):
        actions = [
            "Keep this model as champion while monitoring drift, confidence, and audit coverage.",
            "Use retrain-from-drift when fresh labeled data becomes available.",
        ]
        if not production:
            actions.insert(0, "Promote this validated champion when it is ready for production traffic.")
        return actions

    selected_score = _numeric_score(selected)
    champion_score = _numeric_score(champion)
    if selected_score is None or champion_score is None:
        return ["Retrain or rerun this challenger so it has a comparable score."]

    gap = champion_score - selected_score
    if gap <= 0.02:
        return [
            "This challenger is close to champion performance; validate it with batch scoring before promotion.",
            "Compare drift and confidence on recent data before switching APIs.",
        ]

    actions = [
        f"Champion remains stronger by {round(gap, 4)} {champion.get('primary_metric') or 'score'} point(s).",
        "Use the challenger as experiment evidence or retrain with richer labeled data.",
    ]
    if production and production.get("model_id") != selected.get("model_id"):
        actions.append(f"Production is currently assigned to {production.get('best_model') or 'another model'}.")
    return actions


def build_champion_challenger(selected: dict, models: list[dict], promotions: list[dict] | None = None) -> dict:
    group_key = _group_key(selected)
    group = [
        model
        for model in models
        if _group_key(model) == group_key and _numeric_score(model) is not None
    ]
    group.sort(key=lambda item: _numeric_score(item) or float("-inf"), reverse=True)

    champion = group[0] if group else None
    promotion = _promotion_for_group(selected, promotions)
    production = next(
        (model for model in group if promotion and model.get("model_id") == promotion.get("model_id")),
        None,
    )
    champion_score = _numeric_score(champion or {})
    selected_rank = next(
        (index for index, model in enumerate(group, start=1) if model.get("model_id") == selected.get("model_id")),
        None,
    )
    selected_is_champion = bool(champion and selected.get("model_id") == champion.get("model_id"))
    selected_is_production = bool(production and selected.get("model_id") == production.get("model_id"))
    challengers = [
        _summary(model, champion_score, index)
        for index, model in enumerate(group, start=1)
        if not champion or model.get("model_id") != champion.get("model_id")
    ]

    status = "production" if selected_is_production else "champion" if selected_is_champion else "challenger"
    if selected_rank is None:
        status = "unscored"

    return {
        "available": bool(group),
        "group_key": {
            "target_column": selected.get("target_column"),
            "problem_type": selected.get("problem_type"),
            "primary_metric": selected.get("primary_metric"),
        },
        "status": status,
        "selected_model": _summary(selected, champion_score, selected_rank),
        "production_model": _summary(production, champion_score, None) if production else None,
        "promotion": promotion,
        "champion": _summary(champion, champion_score, 1) if champion else None,
        "challengers": challengers[:8],
        "group_size": len(group),
        "selected_rank": selected_rank,
        "actions": _actions(selected, champion, selected_rank, len(group), production),
    }


def attach_champion_signals(models: list[dict], promotions: list[dict] | None = None) -> list[dict]:
    enriched = []
    for model in models:
        comparison = build_champion_challenger(model, models, promotions)
        champion = comparison.get("champion") or {}
        production = comparison.get("production_model") or {}
        promotion = comparison.get("promotion") or {}
        selected = comparison.get("selected_model") or {}
        enriched.append(
            {
                **model,
                "champion": {
                    "status": comparison.get("status"),
                    "champion_model_id": champion.get("model_id"),
                    "rank": selected.get("rank"),
                    "group_size": comparison.get("group_size"),
                    "score_delta_vs_champion": selected.get("score_delta_vs_champion"),
                },
                "production": {
                    "status": "production" if production.get("model_id") == model.get("model_id") else "not_production",
                    "production_model_id": production.get("model_id"),
                    "promoted_at": promotion.get("promoted_at"),
                },
            }
        )
    return enriched
