"""
Export MobileNetV2 to ONNX format for cross-platform inference.

This script downloads a pre-trained MobileNetV2 model from torchvision,
exports it to ONNX format, validates the export, and saves a sample
input tensor for consistent benchmarking across all deployment modes.
"""

import os
import sys
import time
import json

import numpy as np
import torch
import torchvision.models as models
import onnx
import onnxruntime as ort


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "mobilenet_v2"
INPUT_SHAPE = (1, 3, 224, 224)  # Batch=1, RGB, 224x224
ONNX_OPSET = 13
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(OUTPUT_DIR, "model.onnx")
SAMPLE_INPUT_PATH = os.path.join(OUTPUT_DIR, "sample_input.npy")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")


def export_to_onnx():
    """Export MobileNetV2 to ONNX format."""
    print(f"[1/4] Loading pre-trained {MODEL_NAME}...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    # Generate deterministic sample input
    torch.manual_seed(42)
    sample_input = torch.randn(*INPUT_SHAPE)

    print(f"[2/4] Exporting to ONNX (opset {ONNX_OPSET})...")
    start = time.time()
    torch.onnx.export(
        model,
        sample_input,
        ONNX_PATH,
        opset_version=ONNX_OPSET,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    export_time = time.time() - start
    print(f"    Export completed in {export_time:.2f}s")

    # Save sample input for consistent benchmarking
    np.save(SAMPLE_INPUT_PATH, sample_input.numpy())
    print(f"    Sample input saved to {SAMPLE_INPUT_PATH}")

    return model, sample_input


def validate_onnx(pytorch_model, sample_input):
    """Validate ONNX model produces same outputs as PyTorch."""
    print("[3/4] Validating ONNX export...")

    # Check ONNX model structure
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("    ONNX model structure: OK")

    # Compare PyTorch vs ONNX Runtime outputs
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input).numpy()

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"input": sample_input.numpy()})[0]

    max_diff = np.max(np.abs(pytorch_output - ort_output))
    print(f"    Max output difference: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("    Output validation: OK")
    else:
        print("    Output validation: FAIL (difference exceeds threshold)")
        sys.exit(1)

    return session


def benchmark_inference(session, sample_input_np, n_runs=50):
    """Benchmark ONNX Runtime inference latency."""
    print(f"[4/4] Benchmarking inference ({n_runs} runs)...")

    # Warm-up
    for _ in range(5):
        session.run(None, {"input": sample_input_np})

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"input": sample_input_np})
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    stats = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
    }

    print(f"    Mean:   {stats['mean_ms']:.2f} ms")
    print(f"    Median: {stats['median_ms']:.2f} ms")
    print(f"    P95:    {stats['p95_ms']:.2f} ms")
    print(f"    P99:    {stats['p99_ms']:.2f} ms")

    return stats


def save_metadata(inference_stats):
    """Save model metadata for use by other components."""
    model_size_bytes = os.path.getsize(ONNX_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "input_shape": list(INPUT_SHAPE),
        "onnx_opset": ONNX_OPSET,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": round(model_size_bytes / (1024 * 1024), 2),
        "inference_stats": inference_stats,
        "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Size:  {metadata['model_size_mb']} MB")
    print(f"Files: {ONNX_PATH}")
    print(f"       {SAMPLE_INPUT_PATH}")
    print(f"       {METADATA_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    pytorch_model, sample_input = export_to_onnx()
    session = validate_onnx(pytorch_model, sample_input)
    stats = benchmark_inference(session, sample_input.numpy())
    save_metadata(stats)
    print("\n[OK] Model export complete. Ready for deployment.")
