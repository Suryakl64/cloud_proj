"""
Automated rate-sweep orchestrator for load testing.

Runs Locust in headless mode across all configured sweep rates and
deployment modes. Collects results into per-step CSV files for analysis.

Can also run in simulation mode (no actual endpoints needed) for
testing the analysis pipeline.
"""

import os
import sys
import time
import json
import subprocess
import argparse
import csv
import random
from datetime import datetime

import yaml
import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(config_path: str) -> dict:
    """Load sweep configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Live Sweep (Requires Deployed Endpoints) ─────────────────────────────────

def run_live_sweep(config: dict, modes: list = None):
    """
    Run Locust against live endpoints across all sweep rates.

    Args:
        config: Configuration dictionary from YAML.
        modes: List of modes to test. Defaults to all three.
    """
    if modes is None:
        modes = ["serverless", "container", "managed"]

    rates = config["sweep"]["rates_per_second"]
    duration = config["sweep"]["duration_seconds"]
    warmup = config["sweep"]["warmup_seconds"]
    cooldown = config["sweep"]["cooldown_seconds"]
    output_dir = config["measurement"]["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    total_steps = len(modes) * len(rates)
    step = 0

    for mode in modes:
        endpoint = config["endpoints"].get(mode, {})

        if mode == "managed":
            print(f"  [!] SageMaker endpoint uses boto3 invocation, not HTTP.")
            print(f"  Use sweep_sagemaker() for managed endpoints.")
            continue

        host = endpoint.get("url", "http://localhost:8080")

        for rps in rates:
            step += 1
            print(f"\n{'=' * 60}")
            print(f"  Step {step}/{total_steps}: {mode} @ {rps} req/s")
            print(f"  Duration: {duration}s (+ {warmup}s warmup)")
            print(f"{'=' * 60}")

            # Calculate users needed (1 user = 1 RPS with constant_throughput(1))
            num_users = max(1, int(rps))
            rps_per_user = rps / num_users

            env = os.environ.copy()
            env.update({
                "DEPLOYMENT_MODE": mode,
                "TARGET_RPS": str(rps),
                "LOCUST_RPS_PER_USER": str(rps_per_user),
                "RESULTS_DIR": output_dir,
            })

            locustfile = os.path.join(os.path.dirname(__file__), "locustfile.py")

            cmd = [
                sys.executable, "-m", "locust",
                "-f", locustfile,
                "--headless",
                "--host", host,
                "-u", str(num_users),
                "-r", str(min(num_users, 10)),  # spawn rate
                "-t", f"{duration + warmup}s",
                "--csv", os.path.join(output_dir, f"{mode}_rps{rps}"),
            ]

            if mode == "serverless":
                # Use LambdaInferenceUser for Lambda endpoints
                cmd.extend(["--class-picker"])

            print(f"  Command: {' '.join(cmd[:6])}...")

            try:
                subprocess.run(cmd, env=env, timeout=duration + warmup + 60)
            except subprocess.TimeoutExpired:
                print(f"  [!] Locust timed out for step {step}")
            except Exception as e:
                print(f"  [X] Error: {e}")

            # Cooldown between steps
            if step < total_steps:
                print(f"  Cooling down for {cooldown}s...")
                time.sleep(cooldown)

    print(f"\n[OK] Sweep complete. Results in {output_dir}/")


# ── Simulation Mode (No Cloud Required) ──────────────────────────────────────

def run_simulated_sweep(config: dict, output_dir: str = None):
    """
    Generate synthetic measurement data based on realistic latency
    distributions. Useful for developing the analysis pipeline without
    deploying to cloud.

    Simulates:
    - Serverless: Higher p99 at low rates (cold starts), ~50ms warm inference
    - Container: Consistent ~30ms latency, no cold starts
    - Managed: Consistent ~35ms latency (slight overhead), no cold starts
    """
    if output_dir is None:
        output_dir = config["measurement"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    rates = config["sweep"]["rates_per_second"]
    duration = config["sweep"]["duration_seconds"]

    print("Running simulated sweep (no cloud endpoints required)...")
    print(f"Output: {output_dir}/\n")

    all_records = []

    for mode in ["serverless", "container", "managed"]:
        for rps in rates:
            n_requests = int(rps * duration)
            if n_requests < 1:
                n_requests = max(1, int(rps * duration))

            records = _simulate_step(mode, rps, n_requests, duration)
            all_records.extend(records)

            # Write per-step CSV
            filename = f"{mode}_rps{rps}_{int(time.time())}.csv"
            filepath = os.path.join(output_dir, filename)
            _write_csv(filepath, records)

            cold_count = sum(1 for r in records if r["is_cold_start"])
            latencies = [r["latency_ms"] for r in records]
            print(
                f"  {mode:12s} @ {rps:7.3f} rps: "
                f"n={n_requests:5d}  "
                f"p50={np.percentile(latencies, 50):6.1f}ms  "
                f"p99={np.percentile(latencies, 99):6.1f}ms  "
                f"cold={cold_count}"
            )

    # Write combined CSV
    combined_path = os.path.join(output_dir, "combined_results.csv")
    _write_csv(combined_path, all_records)
    print(f"\n[OK] Simulated sweep complete. {len(all_records)} total records.")
    print(f"  Combined results: {combined_path}")

    return combined_path


def _simulate_step(mode: str, rps: float, n_requests: int, duration: float) -> list:
    """Generate synthetic latency data for one sweep step."""
    records = []
    base_time = time.time()

    for i in range(n_requests):
        timestamp = base_time + (i / max(rps, 0.001))

        if mode == "serverless":
            # Cold-start probability decreases with higher request rates
            # At low rates, ~20% cold starts; at high rates, ~1%
            cold_prob = max(0.01, 0.20 * np.exp(-rps / 2.0))
            is_cold = random.random() < cold_prob

            if is_cold:
                # Cold-start: model loading adds 800-2000ms
                latency = np.random.lognormal(mean=np.log(1200), sigma=0.3)
            else:
                # Warm: typical inference ~40-60ms
                latency = np.random.lognormal(mean=np.log(50), sigma=0.2)

        elif mode == "container":
            # Always-on: consistent low latency, no cold starts
            is_cold = False
            latency = np.random.lognormal(mean=np.log(30), sigma=0.15)

        elif mode == "managed":
            # Managed: slightly higher than container (managed overhead)
            is_cold = (i == 0)  # Only first request is "cold"
            if is_cold:
                latency = np.random.lognormal(mean=np.log(200), sigma=0.2)
            else:
                latency = np.random.lognormal(mean=np.log(35), sigma=0.15)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        records.append({
            "timestamp": timestamp,
            "deployment_mode": mode,
            "rps_target": rps,
            "rps_actual": rps,
            "latency_ms": round(latency, 3),
            "is_cold_start": is_cold,
            "status_code": 200,
            "error": "",
        })

    return records


def _write_csv(filepath: str, records: list):
    """Write records to CSV file."""
    if not records:
        return

    fieldnames = [
        "timestamp", "deployment_mode", "rps_target", "rps_actual",
        "latency_ms", "is_cold_start", "status_code", "error",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ML Inference Load Test Sweep Orchestrator"
    )
    parser.add_argument(
        "mode",
        choices=["live", "simulate"],
        help="'live' = test real endpoints, 'simulate' = generate synthetic data"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to sweep configuration YAML"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory for results"
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=["serverless", "container", "managed"],
        default=None,
        help="Deployment modes to test (default: all)"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config["measurement"]["output_dir"] = args.output_dir

    if args.mode == "simulate":
        run_simulated_sweep(config, args.output_dir)
    elif args.mode == "live":
        run_live_sweep(config, args.modes)


if __name__ == "__main__":
    main()
