"""
Shared ONNX Runtime inference wrapper.

This module provides a unified inference interface used by all three
deployment modes (Lambda, Container, SageMaker). It handles model loading,
input preprocessing, inference execution, and timing instrumentation.
"""

import os
import time
from typing import Optional

import numpy as np
import onnxruntime as ort


class ONNXPredictor:
    """
    ONNX Runtime inference wrapper with timing instrumentation.

    Tracks model load time (for cold-start measurement) and per-request
    inference latency. Thread-safe for use with concurrent request handlers.
    """

    def __init__(self, model_path: str):
        """
        Load an ONNX model and create an inference session.

        Args:
            model_path: Absolute path to the .onnx model file.
        """
        self.model_path = model_path
        self._session: Optional[ort.InferenceSession] = None
        self.model_load_time_ms: float = 0.0
        self._is_cold_start: bool = True
        self._load_model()

    def _load_model(self):
        """Load the ONNX model and record load time."""
        start = time.perf_counter()
        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.model_load_time_ms = (time.perf_counter() - start) * 1000

        # Cache input/output metadata
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        self._output_name = self._session.get_outputs()[0].name

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def input_shape(self) -> list:
        return self._input_shape

    def predict(self, input_data: np.ndarray) -> dict:
        """
        Run inference and return prediction with timing metadata.

        Args:
            input_data: NumPy array matching the model's expected input shape.
                        For MobileNetV2: shape (1, 3, 224, 224), dtype float32.

        Returns:
            Dictionary with keys:
                - predictions: Raw model output (logits or probabilities)
                - top_class: Index of the highest-scoring class
                - top_score: Score of the highest-scoring class
                - inference_time_ms: Time for this inference call
                - model_load_time_ms: Time to load the model (0 if warm)
                - is_cold_start: Whether this was the first invocation
        """
        # Ensure correct dtype
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # Run inference with timing
        start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: input_data})
        inference_time_ms = (time.perf_counter() - start) * 1000

        predictions = outputs[0]
        top_class = int(np.argmax(predictions, axis=1)[0])
        top_score = float(np.max(predictions, axis=1)[0])

        # Track cold-start status
        was_cold_start = self._is_cold_start
        self._is_cold_start = False

        return {
            "predictions": predictions.tolist(),
            "top_class": top_class,
            "top_score": top_score,
            "inference_time_ms": round(inference_time_ms, 3),
            "model_load_time_ms": round(self.model_load_time_ms, 3),
            "is_cold_start": was_cold_start,
        }

    def predict_from_random(self) -> dict:
        """
        Run inference with a random input tensor (for benchmarking).

        Returns:
            Same dictionary as predict().
        """
        # Generate input matching model's expected shape
        shape = list(self._input_shape)
        # Replace dynamic axes (None or strings) with 1
        shape = [s if isinstance(s, int) else 1 for s in shape]
        random_input = np.random.randn(*shape).astype(np.float32)
        return self.predict(random_input)


def create_predictor(model_path: Optional[str] = None) -> ONNXPredictor:
    """
    Factory function to create a predictor with default model path.

    Searches for the model in standard locations:
    1. Provided model_path
    2. Environment variable ONNX_MODEL_PATH
    3. ./model/model.onnx (relative to project root)
    4. /opt/ml/model/model.onnx (SageMaker convention)
    """
    if model_path and os.path.exists(model_path):
        return ONNXPredictor(model_path)

    env_path = os.environ.get("ONNX_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return ONNXPredictor(env_path)

    # Search standard locations
    search_paths = [
        os.path.join(os.path.dirname(__file__), "..", "model", "model.onnx"),
        "/opt/ml/model/model.onnx",          # SageMaker
        "/var/task/model/model.onnx",         # Lambda
        "./model.onnx",                       # Current directory
    ]

    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return ONNXPredictor(abs_path)

    raise FileNotFoundError(
        f"ONNX model not found. Searched: {search_paths}. "
        "Set ONNX_MODEL_PATH environment variable or provide model_path."
    )
