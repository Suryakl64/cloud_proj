# Cost and Latency Crossover Analysis of ML Inference Deployment Modes

## Serverless vs Container vs Managed Endpoint (Azure)

A parameterised cost model and experimental framework for determining the
request rate at which the cheapest ML inference deployment mode changes.

| Mode | Azure Service | Instance | Hourly Cost |
|------|--------------|----------|-------------|
| **Serverless** | Azure Functions (Consumption) | — | Pay-per-execution |
| **Container** | Docker on Azure VM | Standard_B2s | $0.0416/hr |
| **Managed** | Azure ML Online Endpoint | Standard_DS1_v2 | $0.0693/hr |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Export the Model

```bash
python model/export_model.py
```

This exports MobileNetV2 (~14 MB) to ONNX format and benchmarks local
inference latency.

### 3. Run the Cost Simulation (No Cloud Required)

```bash
# Generate simulated load test data
python loadtest/sweep_runner.py simulate

# Run the cost model analysis
python cost_model/simulator.py

# Generate publication figures
python analysis/plot_helpers.py
```

### 4. (Optional) Deploy to Azure

```bash
# Login to Azure
az login

# Deploy Azure Functions (serverless)
bash deployments/serverless/deploy.sh ml-inference-rg eastus

# Deploy Docker container on Azure VM
bash deployments/container/deploy.sh ml-inference-rg eastus

# Deploy Azure ML Managed Online Endpoint
python deployments/managed/deploy_endpoint.py create \
    --subscription-id YOUR_SUBSCRIPTION_ID

# Run live load tests against deployed endpoints
python loadtest/sweep_runner.py live
```

---

## Project Structure

```
cloud_proj/
├── model/                    # Phase 1: Model preparation
│   └── export_model.py       # Export MobileNetV2 → ONNX
├── inference/                # Shared inference wrapper
│   └── predict.py            # ONNX Runtime predictor
├── deployments/              # Phase 2: Three deployment modes
│   ├── serverless/           # Azure Functions (Consumption Plan)
│   ├── container/            # Docker on Azure VM (Standard_B2s)
│   └── managed/              # Azure ML Managed Online Endpoint
├── loadtest/                 # Phase 3: Load testing
│   ├── locustfile.py         # Locust test definition
│   ├── sweep_runner.py       # Rate-sweep orchestrator
│   └── config.yaml           # Sweep configuration
├── cost_model/               # Phase 4: Cost modelling
│   ├── pricing.py            # Parameterised pricing (Azure defaults)
│   ├── analytical_model.py   # Cost equations
│   ├── crossover.py          # Crossover solver + sensitivity
│   └── simulator.py          # Full simulation engine
└── analysis/                 # Phase 5: Visualisation
    ├── plot_helpers.py        # Publication-quality plots
    └── notebook.ipynb         # Interactive analysis notebook
```

## Cost Model

### Cost Per 1000 Inferences

| Mode | Formula | Behaviour |
|------|---------|-----------|
| **Serverless** | `1000 × (c_req + c_gb_sec × mem × dur)` | Constant (per-use) |
| **Container** | `c_hour / (λ × 3600) × 1000` | Decreases with λ |
| **Managed** | `c_ml_hour / (λ × 3600) × 1000` | Decreases with λ |

### Crossover Formula

The serverless ↔ container crossover occurs at:

```
λ* = c_hour / ((c_req + c_gb_sec × mem × dur) × 3600)
```

With default Azure pricing (Functions 1024MB, 50ms inference, B2s VM):
**λ* ≈ 12.6 req/s** (≈ 756 req/min)

Below this rate → serverless is cheaper.
Above this rate → container is cheaper.

## Evaluation Metrics

- **Cost per 1000 inferences** (USD)
- **P50 and P99 latency** (milliseconds)
- **Cold-start frequency** (% of invocations)
- **Cold-start penalty** (additional latency in ms)
- **Crossover request rate** (λ* in req/s)
- **Model prediction error** (MAPE %)

## License

MIT
