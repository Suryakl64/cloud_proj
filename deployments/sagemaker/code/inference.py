"""
AWS SageMaker custom inference handler for MobileNetV2 (ONNX Runtime).

Defines standard SageMaker PyTorch / Triton container hooks:
- model_fn: Loads the ONNX runtime session once on container start
- input_fn: Deserializes the incoming JSON request payload
- predict_fn: Executes the ONNX inference session and measures latency
- output_fn: Serializes the predictions and telemetry into JSON
"""

import os
import json
import time
import numpy as np
import onnxruntime as ort

def model_fn(model_dir):
    """
    Loads the ONNX model from the model directory.
    Called once when the SageMaker container environment starts up.
    """
    start_time = time.perf_counter()
    
    # Try finding model.onnx or model_ir9.onnx
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model_ir9.onnx")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {model_dir}")

    # Configure ONNX Runtime execution session for CPU
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2

    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    load_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "session": session,
        "input_name": session.get_inputs()[0].name,
        "input_shape": session.get_inputs()[0].shape,
        "load_time_ms": load_time_ms,
        "is_first_request": True
    }


def input_fn(request_body, request_content_type="application/json"):
    """
    Parses the incoming request body into a numpy float32 tensor.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        # Support various payload formats
        if "random" in data and data["random"]:
            # Generate dummy 1x3x224x224 tensor
            return np.random.randn(1, 3, 224, 224).astype(np.float32)
        elif "input" in data:
            tensor = np.array(data["input"], dtype=np.float32).reshape((1, 3, 224, 224))
            return tensor
        elif "data" in data:
            shape = data.get("shape", [1, 3, 224, 224])
            tensor = np.array(data["data"], dtype=np.float32).reshape(shape)
            return tensor
        else:
            # Fallback to random tensor
            return np.zeros((1, 3, 224, 224), dtype=np.float32)
            
    elif request_content_type == "application/x-npy":
        return np.load(request_body).astype(np.float32)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_tensor, model_dict):
    """
    Executes model inference and measures execution time.
    """
    session = model_dict["session"]
    input_name = model_dict["input_name"]
    is_cold = model_dict["is_first_request"]
    model_dict["is_first_request"] = False

    t_start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time_ms = (time.perf_counter() - t_start) * 1000

    # Top-1 prediction class
    predicted_class = int(np.argmax(outputs[0]))
    confidence = float(np.max(outputs[0]))

    return {
        "status": "success",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "inference_time_ms": round(inference_time_ms, 2),
        "model_load_time_ms": round(model_dict["load_time_ms"], 2),
        "is_cold_start": is_cold,
        "platform": "aws_sagemaker_serverless"
    }


def output_fn(prediction, accept="application/json"):
    """
    Serializes prediction dictionary to JSON string.
    """
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
