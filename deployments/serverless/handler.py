"""
AWS Lambda handler for serverless ML inference.

Loads the ONNX model once per execution environment (warm start) and
serves predictions via API Gateway proxy integration.
"""

import json
import os
import sys
import time
import logging

import numpy as np

# Add project root to path for shared inference module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from inference.predict import ONNXPredictor

# ── Global model (persists across warm invocations) ──────────────────────────

_predictor = None
_cold_start_timestamp = None
_invocation_count = 0


def _get_predictor() -> ONNXPredictor:
    """Lazy-load model on first invocation; reuse on warm invocations."""
    global _predictor, _cold_start_timestamp
    if _predictor is None:
        model_path = os.environ.get(
            "ONNX_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "model", "model.onnx"),
        )
        _cold_start_timestamp = time.time()
        _predictor = ONNXPredictor(model_path)
    return _predictor


def lambda_handler(event, context):
    """
    AWS Lambda proxy integration handler.
    Supports API Gateway (v1), Function URL (v2), and direct test events.
    """
    global _invocation_count
    _invocation_count += 1
    request_start = time.perf_counter()

    # Determine HTTP method if invoked via HTTP (API Gateway or Function URL)
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method")

    # Parse body from direct test event or HTTP proxy body
    if "random" in event or "data" in event:
        body = event
    else:
        raw_body = event.get("body", "{}")
        if isinstance(raw_body, str):
            try:
                body = json.loads(raw_body) if raw_body else {}
            except Exception:
                body = {}
        elif isinstance(raw_body, dict):
            body = raw_body
        else:
            body = {}

    # Handle healthcheck GET requests
    if http_method == "GET" and not body.get("random"):
        predictor = _get_predictor()
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "status": "healthy",
                "deployment_mode": "serverless",
                "platform": "aws_lambda",
                "model_loaded": predictor._session is not None,
                "model_load_time_ms": predictor.model_load_time_ms,
                "input_shape": predictor.input_shape,
                "invocation_count": _invocation_count,
            })
        }

    try:
        predictor = _get_predictor()

        # Run inference
        if body.get("random", False):
            result = predictor.predict_from_random()
        else:
            input_data = np.array(body["data"], dtype=np.float32)
            if "shape" in body:
                input_data = input_data.reshape(body["shape"])
            result = predictor.predict(input_data)

        # Remove raw predictions array from response
        result.pop("predictions", None)

        # Add AWS Lambda-specific metadata
        total_time_ms = (time.perf_counter() - request_start) * 1000
        result.update({
            "deployment_mode": "serverless",
            "total_request_time_ms": round(total_time_ms, 3),
            "invocation_count": _invocation_count,
            "platform": "aws_lambda",
        })

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }

    except Exception as e:
        logging.error(f"Inference error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "deployment_mode": "serverless",
            })
        }


if __name__ == "__main__":
    print("Testing handler locally (without Lambda runtime)...")
    predictor = _get_predictor()
    result = predictor.predict_from_random()
    result.pop("predictions", None)
    print(json.dumps(result, indent=2))
