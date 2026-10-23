# Cost and Latency Crossover Analysis of ML Inference Deployment Modes

A parameterized cost model and empirical validation framework for determining the exact request rate at which the most cost-effective Machine Learning inference deployment architecture changes.

[![AWS Cloud](https://img.shields.io/badge/AWS-EC2%20%7C%20Lambda%20%7C%20SageMaker-orange)](https://aws.amazon.com/)
[![ONNX Runtime](https://img.shields.io/badge/Model-MobileNetV2%20(ONNX)-blue)](https://onnxruntime.ai/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-green)](https://www.python.org/)

---

## 1. Problem Statement & Hypotheses

Machine Learning inference deployment represents up to 90% of total operational ML lifecycle expenses. Cloud practitioners typically choose between three paradigm tiers:
1. **Infrastructure-as-a-Service (IaaS):** Dedicated, always-on VM containers (e.g., Docker on AWS EC2).
2. **Function-as-a-Service (FaaS):** MicroVM serverless functions (e.g., AWS Lambda).
3. **Platform-as-a-Service (PaaS):** Specialized managed ML serverless endpoints (e.g., AWS SageMaker Serverless).

While serverless offerings promise auto-scaling to zero ($0 idle cost), they impose cold-start latencies and higher marginal per-request charges under sustained traffic.

* **H1 (Cost Efficiency under Sparse Traffic):** At sparse request rates ($\lambda < 3.19$ req/s), serverless architectures reduce inference costs by up to 98.6% by eliminating idle VM compute allocation ($15.18/mo). *(Confirmed)*.
* **H2 (Crossover Decision Boundaries):** As request rates scale, fixed-cost amortization creates mathematically and empirically verifiable crossover thresholds where always-on containers become superior in cost and throughput. *(Confirmed: $\lambda_1^* = 3.190$ req/s for Lambda vs. EC2, and $\lambda_2^* = 5.185$ req/s for SageMaker Serverless vs. EC2)*.
* **H3 (Cold-Start Tail Penalty):** FaaS and PaaS architectures exhibit a statistically significant initialization penalty (150–415 ms) during container allocation bursts compared to pre-warmed VM containers (0 ms delay). *(Confirmed)*.

---

## 2. Experimental Testbed Architecture

All three deployment paradigms were built, optimized, and deployed in the same AWS region (**Sydney, `ap-southeast-2`**) serving an ONNX-optimized **MobileNetV2** computer vision classification pipeline:

| Deployment Mode | Paradigm Tier | AWS Service | Hardware / Memory | Pricing Model | Idle Cost |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Option A (Baseline)** | **IaaS** | Docker on AWS EC2 | `t3.small` (2 vCPU, 2 GB RAM) | Fixed: $0.0208 / hr ($15.18/mo) | $15.18/mo |
| **Option B (FaaS)** | **FaaS** | AWS Lambda | 1024 MB RAM, Python 3.11 | $0.00001667/GB-s + $0.20/1M req | **$0.00** |
| **Option C (PaaS)** | **PaaS** | SageMaker Serverless | 2048 MB RAM, PyTorch 2.0 DLC | $0.00002000/GB-s + $0.20/1M req | **$0.00** |
| **Reference (Dedicated)** | **PaaS Real-Time** | SageMaker Real-Time | `ml.t2.medium` (Dedicated VM) | Fixed: $0.0650 / hr ($47.45/mo) | $47.45/mo |

---

## 3. Empirical Live Cloud Benchmarks (3-Way Evaluation)

We conducted multi-concurrency automated load testing across all three live endpoints using `loadtest/benchmark_live.py` under concurrent worker stages (1, 5, 10 workers; raw telemetry stored in `analysis/results/raw/live_benchmark_3way_1792696104.csv`).

### Empirical Telemetry Summary

| Metric | Option A: EC2 Container (IaaS) | Option B: AWS Lambda (FaaS) | Option C: SageMaker Serverless (PaaS) |
| :--- | :---: | :---: | :---: |
| **Mean Inference Time** | **10.31 ms** | **29.18 ms** | **20.84 ms** |
| **Inference Std. Dev (σ)** | **0.14 ms** | 12.93 ms | 16.61 ms |
| **Cold-Start Provisioning Delay** | **0.0 ms** (Pre-warmed) | **413.2 ms** (Cold init + load) | **155.6 ms** (Model load) |
| **Idle Hourly Cost** | $0.0208 / hr | **$0.00 / hr** | **$0.00 / hr** |
| **Max Tested Throughput** | **18.7 req/s** | 5.6 req/s | 9.8 req/s |

### Core Architectural Insights

1. **Deterministic Latency on EC2:** The Dockerized FastAPI container on EC2 provided deterministic, jitter-free **10.31 ms** inference with virtually zero standard deviation ($\sigma = 0.14$ ms) and zero cold-start delay.
2. **Cold-Start Differential:** SageMaker Serverless demonstrated a faster container model-loading initialization (**155.6 ms**) than Lambda's full runtime bootstrap (**413.2 ms**), benefiting from pre-baked deep learning container (DLC) environments.
3. **CPU Allocation vs. Inference Speed:** At 1024 MB, Lambda is allocated ~0.6 vCPU credits, yielding ~29.2 ms inference. SageMaker Serverless at 2048 MB provides 2 vCPU burst compute, executing warm inference in **20.84 ms** (1.4× faster than Lambda).

---

## 4. Analytical Cost Model & Decision Boundaries

The cost crossover equations model total cost per 1,000 inferences ($C_{1k}$) as a function of request arrival rate $\lambda$ (req/s):

$$C_{\text{serverless}}(\lambda) = 1000 \times \left( c_{\text{req}} + c_{\text{gb}} \cdot M \cdot \left[ (1 - P_{\text{cold}}(\lambda))\tau_{\text{warm}} + P_{\text{cold}}(\lambda)\tau_{\text{cold}} \right] \right)$$

$$C_{\text{container}}(\lambda) = \frac{C_{\text{hour}}}{3600 \cdot \lambda} \times 1000$$

### Crossover Decision Boundaries

Solving for $C_{\text{mode A}}(\lambda^*) = C_{\text{mode B}}(\lambda^*)$ yields three primary operational boundaries:

* **$\lambda_1^* = 3.190$ req/s (191 req/min) — Lambda vs. EC2:**
  Below 3.19 req/s, AWS Lambda is optimal, reducing cost by up to 98.6%. Above 3.19 req/s, EC2 Container becomes cheaper per request.
* **$\lambda_2^* = 5.185$ req/s (311 req/min) — SageMaker Serverless vs. EC2:**
  Below 5.19 req/s, SageMaker Serverless is more cost-effective than an always-on EC2 VM, while providing native AWS ML monitoring and IAM isolation.
* **$\lambda_3^* = 16.603$ req/s (996 req/min) — SageMaker Serverless vs. SageMaker Real-Time:**
  Within the SageMaker ecosystem, serverless endpoints remain cheaper than dedicated always-on `ml.t2.medium` instances until traffic surpasses ~16.6 req/s.

### Decision Matrix

```
   0.0 req/s           3.19 req/s          5.19 req/s                   16.6 req/s
───────┼───────────────────┼───────────────────┼────────────────────────────┼──────────>
       │   ZONE I (FaaS)   │   ZONE II (PaaS)  │       ZONE III (IaaS)      │
       │ AWS Lambda Wins   │ SageMaker Serv.   │   EC2 Container Dominates  │
       │ ($0 Idle, Low Cost│ (Zero Idle, Faster│   (Lowest Cost/req, Zero   │
       │  for sparse load) │  Warm Inference)  │    Cold-Start Jitter)      │
```

---

## 5. Artifacts & Publication Figures

All visual assets are rendered at 300 DPI in presentation-ready widescreen format and stored in `analysis/results/figures/`:

* **Figure 1 (`fig1_cost_crossover.png`):** Analytical Cost vs. Rate curves illustrating all 4 deployment modes and the 3 crossover decision boundaries.
* **Live Comparison (`live_comparison.png`):** Empirical bar chart comparing pure inference latency and cold-start provisioning overhead across EC2, Lambda, and SageMaker Serverless.
* **Architecture Diagram (`architecture_diagram.png`):** Detailed architectural schematic showing the client load generator, AWS VPC / IAM / S3 / DLC components, and telemetry analysis engine.
* **Simplified Architecture (`architecture_simple.png`):** Slide-ready 3-branch overview diagram for presentations.
* **Figures 2–5:** Latency distributions, cold-start decay modeling, parameter sensitivity sweeps, and mathematical model validation (< 5% MAPE).

---

## 6. Quick Start & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Suryakl64/cloud_proj.git
cd cloud_proj

# Install required dependencies
pip install -r requirements.txt
```

### 2. Solve Crossover Equations & Run Simulation
```bash
# Run the mathematical crossover solver
python -m cost_model.crossover

# Execute parametric sweep simulation and generate figures 1-5
python -m cost_model.simulator
python analysis/plot_helpers.py
```

### 3. Run 3-Way Live Cloud Benchmark
```bash
# Execute automated live benchmark against all 3 endpoints
python loadtest/benchmark_live.py --mode all --concurrency 1 5 10

# Generate updated 3-way live comparison chart
python analysis/plot_live_benchmark.py
```

---

## 7. Project Directory Map

```
cloud_proj/
├── model/                     # MobileNetV2 ONNX model export & verification
├── inference/                 # Shared ONNX Runtime prediction module
├── deployments/
│   ├── container/             # Dockerfile & FastAPI server for AWS EC2
│   ├── serverless/            # AWS Lambda handler, packaging & deploy scripts
│   └── sagemaker/             # SageMaker Serverless inference handler & packaging
├── loadtest/
│   ├── config.yaml            # Endpoint URLs & concurrency parameters
│   └── benchmark_live.py      # Async 3-way live load testing engine
├── cost_model/
│   ├── pricing.py             # Parameterized pricing for EC2, Lambda, and SageMaker
│   ├── analytical_model.py    # Closed-form cost equations & monthly estimators
│   ├── crossover.py           # Numerical and analytical crossover solvers
│   └── simulator.py           # Full sweep simulation & sensitivity generator
└── analysis/
    ├── generate_architecture_diagram.py  # Publication architecture diagram generator
    ├── generate_simple_architecture.py    # Slide-ready presentation diagram generator
    ├── plot_helpers.py                   # Generates publication figures 1-5
    ├── plot_live_benchmark.py            # Generates 3-way live comparison bar charts
    └── results/                          # Raw CSV logs, sweep data & 300 DPI figures
```
