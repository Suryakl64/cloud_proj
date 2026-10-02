# Cost and Latency Crossover Analysis of ML Inference Deployment Modes

A parameterized cost model and empirical validation framework for determining the exact request rate at which the most cost-effective Machine Learning inference deployment architecture changes.

[![AWS Cloud](https://img.shields.io/badge/AWS-Lambda%20%7C%20EC2-orange)](https://aws.amazon.com/)
[![ONNX Runtime](https://img.shields.io/badge/Model-MobileNetV2%20(ONNX)-blue)](https://onnxruntime.ai/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-green)](https://www.python.org/)

---

## 1. Problem Statement & Hypotheses

Cloud ML inference can be deployed using serverless functions, always-on containers, or managed endpoints. While serverless promises zero idle cost and automatic scaling, it introduces cold-start penalties and higher marginal costs per request under sustained traffic.

* **H1 (Cost Efficiency at Low Traffic):** Under sparse request traffic (< 3.19 req/s), serverless inference reduces cost per 1,000 inferences by at least 30% compared with an always-on container VM. *(Confirmed: up to 98.6% savings at sparse rates)*.
* **H2 (Crossover Point):** As request rates increase, the cost per 1,000 inferences diverges, creating a measurable crossover request rate where the container architecture becomes the least-cost option. *(Confirmed: mathematical and empirical crossover at $\lambda^* = 3.190$ req/s)*.

---

## 2. Experimental Testbed Architecture

Both architectures are deployed and evaluated in the same AWS region (**Sydney, `ap-southeast-2`**) running an ONNX-optimized **MobileNetV2** computer vision pipeline.

| Architecture | AWS Service | Hardware / Memory | Pricing Model | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Container)** | Docker on AWS EC2 | `t3.small` (2 vCPU, 2 GB RAM) | Fixed: ~$0.0208 / hour ($15.18/mo) | Baseline |
| **Proposed (Serverless)** | AWS Lambda | 1024 MB RAM, Python 3.11 | Pay-per-ms ($0.00001667/GB-s + $0.20/1M req) | Proposed |
| **Reference (Managed)** | SageMaker Endpoint | `ml.t3.medium` | Fixed: ~$0.05 / hour | Reference model |

---

## 3. Review 2: Empirical Validation & Live Benchmarks

We executed automated live load testing across both endpoints under concurrent workloads (1, 5, 10, and 20 concurrent workers) using `loadtest/benchmark_live.py`.

### Measured Performance Summary

| Architecture | Sample Size | Mean Inference Time | P50 Round-Trip | Cold-Start Overhead | Throughput @ 10 Concurrency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AWS EC2 Container (`t3.small`)** | 80 | **10.28 ms** (σ=0.32 ms) | 468.27 ms | **0 ms** (pre-loaded) | **18.7 req/s** |
| **AWS Lambda Serverless** | 71 | **27.92 ms** (σ=10.12 ms) | 409.25 ms | **394.21 ms** | **5.6 req/s** |

### Key Architectural Takeaways

1. **Inference Consistency:** The Dockerized FastAPI container on EC2 provided deterministic, rock-solid **10.2 ms** inference latency with zero cold-start variance.
2. **Cold-Start Penalty:** Serverless Lambda exhibited a **~394 ms model-loading penalty** whenever concurrent bursts triggered new container execution environments.
3. **Crossover Validation:** Below 3.19 req/s, Lambda is up to 98% cheaper by eliminating idle VM costs; above 3.19 req/s, the EC2 container is superior in throughput and cost efficiency.

---

## 4. Key Figures & Sensitivity Sweeps

All generated publication plots are stored in `analysis/results/figures/`:

* **Figure 1 (`fig1_cost_crossover.png`):** Analytical Cost vs. Request Rate curve proving the crossover at **3.190 req/s**.
* **Figure 2 (`fig2_latency_distribution.png`):** Latency distributions and cold-start tail latency comparisons.
* **Figure 3 (`fig3_cold_start_analysis.png`):** Cold-start frequency decay as traffic scales.
* **Figure 4 (`fig4_sensitivity_analysis.png`):** 3-panel parameter sweep demonstrating how crossover shifts with model duration (10–200ms), memory tier (128–3008MB), and VM hourly cost ($0.01–$0.10/hr).
* **Figure 5 (`fig5_model_validation.png`):** Validation comparing mathematical predictions against experimental measurements (< 5% error).
* **Live Benchmark (`live_comparison.png`):** Live empirical boxplots and latency breakdown from real AWS cloud endpoints.

---

## 5. Threats to Validity

1. **Network Transit Latency:** Client-to-Sydney internet transit introduces geographic network ping (~300 ms RTT) on top of raw server compute latency.
2. **Model Weight Scale:** Tested with lightweight MobileNetV2 (14 MB). Multi-gigabyte LLMs or heavy diffusion models will face substantially longer serverless cold starts (10–30s).
3. **Synthetic vs. Real-World Traffic:** Benchmark uses Poisson/fixed concurrency distributions, whereas real human traffic exhibits diurnal spikes causing more frequent cold starts.
4. **Secondary Cloud Overheads:** The core model focuses on compute, abstracting data egress and CloudWatch logging costs.

---

## 6. Quick Start & Reproduction

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Cost Model & Parameter Sweeps
```bash
# Run the analytical crossover solver
python cost_model/crossover.py

# Generate simulated sweep results & 5 publication figures
python loadtest/sweep_runner.py simulate
python analysis/plot_helpers.py
```

### 3. Run Live Cloud Benchmark
```bash
# Run automated live benchmark against both cloud endpoints
python loadtest/benchmark_live.py --mode both --concurrency 1 5 10 20

# Plot live comparison charts
python analysis/plot_live_benchmark.py
```

---

## 7. Project Structure
```
cloud_proj/
├── model/                     # MobileNetV2 ONNX model & export scripts
├── inference/                 # Shared ONNX Runtime predictor wrapper
├── deployments/
│   ├── container/             # Dockerfile & FastAPI server for EC2
│   └── serverless/            # Lambda handler & deployment scripts
├── loadtest/
│   ├── config.yaml            # Sweep configurations & endpoint URLs
│   ├── locustfile.py          # Locust user definitions & CSV loggers
│   ├── sweep_runner.py        # Automated rate-sweep orchestrator
│   └── benchmark_live.py      # Async live concurrency benchmark runner
├── cost_model/
│   ├── pricing.py             # AWS pricing parameters
│   ├── analytical_model.py    # Closed-form cost equations
│   └── crossover.py           # Mathematical crossover solver
└── analysis/
    ├── plot_helpers.py        # Generates publication figures 1-5
    ├── plot_live_benchmark.py # Plots live empirical cloud comparison
    └── results/               # Measured CSVs, sensitivity sweeps & figures
```
