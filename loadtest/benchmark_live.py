"""
Live asynchronous benchmark runner for ML Inference endpoints:
1. AWS EC2 Container (Baseline IaaS - t3.small)
2. AWS Lambda (Serverless FaaS - 1024 MB)
3. AWS SageMaker Serverless (Managed ML PaaS - 2048 MB)

Uses httpx and Python standard library (csv, statistics) for zero-dependency portability.
"""

import os
import sys
import time
import json
import csv
import asyncio
import argparse
import subprocess
import statistics
from typing import List, Dict, Any

import httpx


LAMBDA_URL = "https://xhxnqryeserx4cixmm74qr5k3i0sfiic.lambda-url.ap-southeast-2.on.aws/"
EC2_URL = "http://3.26.184.234:8080/predict"
SAGEMAKER_ENDPOINT = "sagemaker-mobilenet-serverless"
REGION = "ap-southeast-2"


def _invoke_sagemaker_sync() -> Dict[str, Any]:
    """Synchronous invocation of SageMaker Serverless endpoint via AWS CLI."""
    start_time = time.perf_counter()
    status_code = 0
    error_msg = ""
    server_inference_time = 0.0
    server_load_time = 0.0
    is_cold = False

    payload = json.dumps({"random": True})
    
    # Unique temp files per thread
    import threading
    task_id = f"{os.getpid()}_{threading.get_ident()}_{time.perf_counter_ns()}"
    payload_file = f"temp_payload_{task_id}.json"
    output_file = f"temp_output_{task_id}.json"

    try:
        with open(payload_file, "w") as f:
            f.write(payload)

        cmd = [
            "aws", "sagemaker-runtime", "invoke-endpoint",
            "--endpoint-name", SAGEMAKER_ENDPOINT,
            "--content-type", "application/json",
            "--body", f"fileb://{payload_file}",
            "--region", REGION,
            output_file
        ]
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
        round_trip_ms = (time.perf_counter() - start_time) * 1000

        if res.returncode == 0 and os.path.exists(output_file):
            status_code = 200
            with open(output_file) as f:
                data = json.load(f)
            server_inference_time = data.get("inference_time_ms", 0.0)
            server_load_time = data.get("model_load_time_ms", 0.0)
            is_cold = data.get("is_cold_start", False)
        else:
            status_code = 500
            error_msg = res.stderr.strip() or "Unknown error"

    except Exception as e:
        round_trip_ms = (time.perf_counter() - start_time) * 1000
        status_code = 500
        error_msg = str(e)
    finally:
        for f in [payload_file, output_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    return {
        "timestamp": time.time(),
        "mode": "sagemaker",
        "round_trip_ms": round(round_trip_ms, 2),
        "server_inference_time_ms": round(server_inference_time, 2),
        "server_load_time_ms": round(server_load_time, 2),
        "is_cold_start": is_cold,
        "status_code": status_code,
        "error": error_msg,
    }


async def send_http_request(client: httpx.AsyncClient, url: str, mode: str) -> Dict[str, Any]:
    """Send a single inference request over HTTP to EC2 or Lambda."""
    start_time = time.perf_counter()
    status_code = 0
    error_msg = ""
    server_inference_time = 0.0
    server_load_time = 0.0
    is_cold = False

    try:
        response = await client.post(
            url,
            json={"random": True},
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        status_code = response.status_code
        round_trip_ms = (time.perf_counter() - start_time) * 1000

        if status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "body" in data and isinstance(data["body"], str):
                try:
                    data = json.loads(data["body"])
                except Exception:
                    pass

            server_inference_time = data.get("inference_time_ms", 0.0)
            server_load_time = data.get("model_load_time_ms", 0.0)
            is_cold = data.get("is_cold_start", False)
        else:
            error_msg = f"HTTP {status_code}"

    except Exception as e:
        round_trip_ms = (time.perf_counter() - start_time) * 1000
        status_code = 500
        error_msg = str(e)

    return {
        "timestamp": time.time(),
        "mode": mode,
        "round_trip_ms": round(round_trip_ms, 2),
        "server_inference_time_ms": round(server_inference_time, 2),
        "server_load_time_ms": round(server_load_time, 2),
        "is_cold_start": is_cold,
        "status_code": status_code,
        "error": error_msg,
    }


def _calc_percentile(data: List[float], p: float) -> float:
    """Calculates percentile using standard library."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int((len(data_sorted) - 1) * (p / 100.0))
    return data_sorted[idx]


async def run_stage(target: str, mode: str, concurrency: int, total_requests: int) -> List[Dict[str, Any]]:
    """Run a batch of requests with a fixed concurrency level."""
    print(f"\n[Benchmarking {mode.upper()}] Concurrency: {concurrency} | Total requests: {total_requests}...")
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    if mode == "sagemaker":
        async def sagemaker_worker():
            async with semaphore:
                return await asyncio.to_thread(_invoke_sagemaker_sync)

        t0 = time.perf_counter()
        tasks = [sagemaker_worker() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    else:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async def http_worker():
                async with semaphore:
                    return await send_http_request(client, target, mode)

            t0 = time.perf_counter()
            tasks = [http_worker() for _ in range(total_requests)]
            results = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0

    success_count = sum(1 for r in results if r["status_code"] == 200)
    latencies = [r["round_trip_ms"] for r in results if r["status_code"] == 200]
    throughput = len(results) / elapsed if elapsed > 0 else 0

    print(f"  Completed in {elapsed:.2f}s ({throughput:.1f} req/s)")
    if latencies:
        p50 = _calc_percentile(latencies, 50)
        p90 = _calc_percentile(latencies, 90)
        p99 = _calc_percentile(latencies, 99)
        print(f"  Success: {success_count}/{total_requests} ({(success_count/total_requests)*100:.1f}%)")
        print(f"  Latency: p50={p50:.1f}ms | p90={p90:.1f}ms | p99={p99:.1f}ms")
    else:
        print(f"  All requests failed. Errors: {set(r['error'] for r in results)}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Live 3-Way Benchmark Runner")
    parser.add_argument("--mode", choices=["container", "serverless", "sagemaker", "both", "all"], default="all")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--requests-per-stage", type=int, default=15)
    args = parser.parse_args()

    all_records = []
    if args.mode == "all":
        modes_to_test = ["container", "serverless", "sagemaker"]
    elif args.mode == "both":
        modes_to_test = ["container", "serverless"]
    else:
        modes_to_test = [args.mode]

    for m in modes_to_test:
        if m == "container":
            target = EC2_URL
        elif m == "serverless":
            target = LAMBDA_URL
        else:
            target = SAGEMAKER_ENDPOINT

        print(f"\n{'='*65}")
        print(f"  STARTING 3-WAY BENCHMARK FOR {m.upper()} ({target})")
        print(f"{'='*65}")

        for c in args.concurrency:
            stage_records = await run_stage(target, m, c, args.requests_per_stage)
            all_records.extend(stage_records)
            await asyncio.sleep(2)  # Cooldown between concurrency levels

    output_dir = "analysis/results/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = int(time.time())
    csv_path = os.path.join(output_dir, f"live_benchmark_3way_{timestamp}.csv")
    
    fieldnames = [
        "timestamp", "mode", "round_trip_ms", "server_inference_time_ms",
        "server_load_time_ms", "is_cold_start", "status_code", "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n[OK] 3-Way Benchmark results saved to {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
