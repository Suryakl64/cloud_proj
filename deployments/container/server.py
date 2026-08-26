"""
FastAPI inference server for container-based ML deployment.

Runs as an always-on service on an EC2 instance (or any Docker host).
Represents the "container" deployment mode — fixed hourly cost regardless
of request volume. Model is loaded once at startup (no cold starts).
"""

import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# Add project root to path for shared inference module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from inference.predict import ONNXPredictor


# ── Request/Response Models ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    data: Optional[list] = None
    shape: Optional[list] = None
    random: bool = Field(default=False, description="Use random input for benchmarking")


class PredictResponse(BaseModel):
    top_class: int
    top_score: float
    inference_time_ms: float
    model_load_time_ms: float
    is_cold_start: bool
    deployment_mode: str = "container"
    total_request_time_ms: float = 0.0
    request_count: int = 0


# ── Application ──────────────────────────────────────────────────────────────

# Global state
_predictor: Optional[ONNXPredictor] = None
_request_count = 0
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; clean up at shutdown."""
    global _predictor
    model_path = os.environ.get(
        "ONNX_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "model", "model.onnx"),
    )
    print(f"Loading model from {model_path}...")
    _predictor = ONNXPredictor(model_path)
    print(f"Model loaded in {_predictor.model_load_time_ms:.1f} ms")
    yield
    print("Shutting down inference server.")


app = FastAPI(
    title="ML Inference Server (Container Mode)",
    description="Always-on container inference for crossover analysis",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run inference on the loaded ONNX model.

    Send {"random": true} for benchmarking with random input.
    """
    global _request_count
    _request_count += 1

    request_start = time.perf_counter()

    try:
        if request.random:
            result = _predictor.predict_from_random()
        else:
            if request.data is None:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide 'data' array or set 'random': true",
                )
            input_data = np.array(request.data, dtype=np.float32)
            if request.shape:
                input_data = input_data.reshape(request.shape)
            result = _predictor.predict(input_data)

        total_time_ms = (time.perf_counter() - request_start) * 1000

        return PredictResponse(
            top_class=result["top_class"],
            top_score=result["top_score"],
            inference_time_ms=result["inference_time_ms"],
            model_load_time_ms=result["model_load_time_ms"],
            is_cold_start=result["is_cold_start"],
            deployment_mode="container",
            total_request_time_ms=round(total_time_ms, 3),
            request_count=_request_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    uptime_seconds = time.time() - _start_time
    return JSONResponse({
        "status": "healthy",
        "deployment_mode": "container",
        "model_loaded": _predictor is not None,
        "model_load_time_ms": _predictor.model_load_time_ms if _predictor else 0,
        "input_shape": _predictor.input_shape if _predictor else [],
        "request_count": _request_count,
        "uptime_seconds": round(uptime_seconds, 1),
    })


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "ML Inference Server",
        "deployment_mode": "container",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
