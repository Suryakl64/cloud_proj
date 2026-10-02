"""
Live asynchronous benchmark runner for ML Inference endpoints (AWS Lambda & EC2 Container).
Uses httpx to run controlled concurrency load tests without C-extension dependencies.
"""

import os
import sys
import time
import json
import asyncio
import argparse
from typing import List, Dict, Any

import httpx
import numpy as np
import pandas as pd


LAMBDA_URL = "https://xhxnqryeserx4cixmm74qr5k3i0sfiic.lambda-url.ap-southeast-2.on.aws/"
EC2_URL = "http://3.104.118.57:8080/predict"


async def send_request(client: httpx.AsyncClient, url: str, mode: str) -> Dict[str, Any]:
    """Send a single inference request and measure round-trip time and server timing."""
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
            # If wrapped in Lambda proxy format
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


async def run_stage(url: str, mode: str, concurrency: int, total_requests: int) -> List[Dict[str, Any]]:
    """Run a batch of requests with a fixed concurrency level."""
    print(f"\n[Benchmarking {mode.upper()}] Concurrency: {concurrency} | Total requests: {total_requests}...")
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def worker():
            async with semaphore:
                return await send_request(client, url, mode)

        tasks = [worker() for _ in range(total_requests)]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    success_count = sum(1 for r in results if r["status_code"] == 200)
    latencies = [r["round_trip_ms"] for r in results if r["status_code"] == 200]
    throughput = len(results) / elapsed

    print(f"  Completed in {elapsed:.2f}s ({throughput:.1f} req/s)")
    if latencies:
        print(f"  Success: {success_count}/{total_requests} ({(success_count/total_requests)*100:.1f}%)")
        print(f"  Latency: p50={np.percentile(latencies, 50):.1f}ms | p90={np.percentile(latencies, 90):.1f}ms | p99={np.percentile(latencies, 99):.1f}ms")
    else:
        print(f"  All requests failed. Errors: {set(r['error'] for r in results)}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Live Benchmark Runner")
    parser.add_argument("--mode", choices=["serverless", "container", "both"], default="serverless")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--requests-per-stage", type=int, default=30)
    args = parser.parse_args()

    all_records = []
    modes_to_test = ["serverless", "container"] if args.mode == "both" else [args.mode]

    for m in modes_to_test:
        url = LAMBDA_URL if m == "serverless" else EC2_URL
        print(f"\n{'='*60}")
        print(f"  STARTING BENCHMARK FOR {m.upper()} ({url})")
        print(f"{'='*60}")

        for c in args.concurrency:
            stage_records = await run_stage(url, m, c, args.requests_per_stage)
            all_records.extend(stage_records)
            await asyncio.sleep(2)  # Cooldown between concurrency levels

    output_dir = "analysis/results/raw"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, f"live_benchmark_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Benchmark results saved to {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
