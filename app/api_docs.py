import json


def _json_block(value: dict) -> str:
    return json.dumps(value, indent=2)


def build_model_api_docs(record: dict, playground: dict, production: dict | None = None) -> dict:
    endpoint = record.get("prediction_api") or f"/api/predict/{record.get('model_id')}"
    sample_payload = playground.get("request_template") or {"rows": [{}]}
    model_id = record.get("model_id")
    base_url = "http://127.0.0.1:8001"
    full_endpoint = f"{base_url}{endpoint}"
    response_example = {
        "model_id": model_id,
        "predictions": ["<prediction>"],
        "prediction_details": [
            {
                "row": 1,
                "prediction": "<prediction>",
                "confidence": 0.91,
                "confidence_band": "high",
                "probabilities": {"class_a": 0.09, "class_b": 0.91},
            }
        ],
        "confidence_summary": {
            "available": record.get("problem_type") == "classification",
            "total_rows": 1,
            "average_confidence": 0.91,
            "low_confidence_rows": 0,
        },
        "audit_id": "<saved-audit-id>",
    }

    return {
        "available": True,
        "model_id": model_id,
        "model_name": record.get("best_model"),
        "target_column": record.get("target_column"),
        "problem_type": record.get("problem_type"),
        "primary_metric": record.get("primary_metric"),
        "rank_score": record.get("rank_score"),
        "is_production": bool(production and production.get("model_id") == model_id),
        "production_model_id": (production or {}).get("model_id"),
        "method": "POST",
        "endpoint": endpoint,
        "full_endpoint": full_endpoint,
        "content_type": "application/json",
        "request_schema": {
            "rows": "Array of objects. Each object must include the model feature columns.",
            "threshold": "Optional number for supported binary classifiers.",
        },
        "required_features": playground.get("feature_columns") or [],
        "sample_payload": sample_payload,
        "response_fields": [
            {"name": "model_id", "type": "string", "description": "Saved model identifier."},
            {"name": "predictions", "type": "array", "description": "Predicted values for each row."},
            {"name": "prediction_details", "type": "array", "description": "Row-level prediction and confidence details."},
            {"name": "confidence_summary", "type": "object", "description": "Confidence and low-confidence counts when available."},
            {"name": "audit_id", "type": "string", "description": "Saved prediction audit event id."},
        ],
        "response_example": response_example,
        "curl": (
            f"curl -X POST \"{full_endpoint}\" "
            "-H \"Content-Type: application/json\" "
            f"-d '{json.dumps(sample_payload)}'"
        ),
        "python": (
            "import requests\n\n"
            f"url = \"{full_endpoint}\"\n"
            f"payload = {_json_block(sample_payload)}\n"
            "response = requests.post(url, json=payload, timeout=30)\n"
            "response.raise_for_status()\n"
            "print(response.json())"
        ),
        "notes": [
            "Use the same feature names shown in required_features.",
            "Batch CSV scoring is available at the same endpoint with /batch appended.",
            "Every successful prediction is saved to the prediction audit log.",
        ],
    }
