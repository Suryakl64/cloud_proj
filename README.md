# Cost and Latency Crossover Analysis of ML Inference Deployment Modes

A parameterised cost model and experimental framework for determining the exact request rate at which the most cost-effective ML inference deployment mode changes.

## 1. Problem Statement & Hypotheses

Cloud ML inference can be deployed using serverless functions, always-on containers, or managed inference endpoints. However, the most cost-effective option may change with request traffic, and the request rate at which this change occurs is not known for a given model and pricing configuration.

**H1 (Cost Efficiency at Low Traffic):** Under sparse request traffic, serverless inference will reduce cost per 1,000 inferences by at least 30% compared with the always-on Docker container running on a small virtual machine, when the same ML model and request workload are used.

**H2 (Crossover Point):** As request rate increases, the cost per 1,000 inferences for serverless, always-on container, and managed inference will vary differently, resulting in a measurable crossover request rate at which the least-cost deployment changes.

## 2. Experimental Testbed

The experimental testbed uses **AWS** as the target cloud provider, executing an ONNX-optimized **MobileNetV2** model.

| Mode | AWS Service | Instance | Hourly Cost |
|------|--------------|----------|-------------|
| **Serverless** | AWS Lambda | 1024 MB | Pay-per-execution |
| **Container** | Docker on EC2 | t3.small | ~$0.0208/hr |
| **Managed** | SageMaker Endpoint | ml.t2.medium | ~$0.05/hr |

*The container testbed (baseline) has been successfully deployed and verified on an EC2 `t3.small` instance, demonstrating fast model load times and sub-40ms inference latency.*

## 3. Baseline Results (Simulated)

Using the parameterised cost model based on the testbed performance, the following crossover points were found:
- **Serverless ↔ Container Crossover:** ~3.190 requests/second
- Below ~3.190 req/s: Serverless is cheaper (proving H1).
- Above ~3.190 req/s: Container becomes the most cost-effective option (proving H2).

*(See `analysis/results/figures/` for the 5 generated publication-quality graphs detailing cost crossover, latency distribution, cold starts, and sensitivity analysis).*

---

## Quick Start (Local & AWS)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Export the Model
```bash
python model/export_model.py
```

### 3. Run the Cost Simulation (No Cloud Required)
```bash
# Generate simulated load test data
python loadtest/sweep_runner.py simulate

# Run the cost model analysis
python cost_model/simulator.py

# Generate publication figures
python analysis/plot_helpers.py
```

### 4. Deploy the Baseline Container Testbed (AWS EC2)
```bash
sudo apt update && sudo apt install docker.io jq -y
git clone https://github.com/Suryakl64/cloud_proj.git
cd cloud_proj
sudo docker build -t baseline-container -f deployments/container/Dockerfile .
sudo docker run -p 8080:8080 baseline-container
```

## Project Structure
```
cloud_proj/
├── model/                    # Model preparation (MobileNetV2 → ONNX)
├── inference/                # Shared ONNX Runtime predictor wrapper
├── deployments/              # Deployment modes (Container, Serverless)
├── loadtest/                 # Locust test definition & rate-sweep orchestrator
├── cost_model/               # Cost equations, simulator, and pricing parameters
└── analysis/                 # Visualisation scripts and generated figures
```
