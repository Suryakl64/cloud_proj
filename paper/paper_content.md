# Cost and Latency Crossover Analysis of Machine Learning Inference Deployment Modes: An Empirical Study on AWS EC2, Lambda, and SageMaker

**Author:** [Your Name]  
**Department:** Department of Computer Science and Engineering  
**Institution:** [Your Institution / University Name]  
**Email:** [your_email@institution.edu]  

---

### Abstract
Operational inference constitutes up to 90% of the aggregate cost in production machine learning (ML) lifecycles. Cloud practitioners face a fundamental architectural trade-off when selecting deployment tiers: Infrastructure-as-a-Service (IaaS) always-on virtual machines, Function-as-a-Service (FaaS) serverless computing, or specialized Platform-as-a-Service (PaaS) managed ML endpoints. While serverless paradigms offer auto-scaling to zero with zero idle cost, they incur cold-start latency overheads and elevated marginal per-request costs under sustained traffic. In this paper, we formulate a closed-form analytical cost model and validate it against live empirical measurements across three production AWS deployment modes in Sydney (`ap-southeast-2`): Dockerized FastAPI on Amazon EC2 (`t3.small`), AWS Lambda (1024 MB), and Amazon SageMaker Serverless Inference (2048 MB), all serving an ONNX-optimized MobileNetV2 computer vision pipeline. Empirical load testing demonstrates that dedicated EC2 containers yield deterministic inference ($10.31 \pm 0.14$ ms) with zero cold-start delay, whereas Lambda and SageMaker Serverless incur cold initialization penalties of $413.2$ ms and $155.6$ ms, respectively. Closed-form and numerical optimization reveals three distinct operational decision boundaries: $\lambda_1^* = 3.190$ req/s (Lambda vs. EC2), $\lambda_2^* = 5.185$ req/s (SageMaker Serverless vs. EC2), and $\lambda_3^* = 16.603$ req/s (SageMaker Serverless vs. SageMaker Real-Time). Below $\lambda_1^*$, Lambda reduces cost by up to 98.6% over always-on VMs; beyond $\lambda_2^*$, fixed VM amortization dominates in both throughput and economic efficiency. Our framework provides actionable decision rules for optimizing ML inference deployments.

**Index Terms:** Serverless Computing, Machine Learning Inference, Cloud Economics, Cost Crossover Analysis, AWS Lambda, Amazon SageMaker, Amazon EC2, ONNX Runtime, Cold Start.

---

## I. Introduction

Machine learning (ML) models are increasingly integrated into real-time interactive applications, ranging from autonomous vision inspection to conversational assistants. However, while substantial research focuses on training efficiency, recent industry analyses reveal that operational serving and inference account for 80% to 90% of total ML cloud expenditures. Consequently, selecting the optimal deployment architecture is a critical engineering decision directly affecting operational budgets, quality of service (QoS), and latency service level agreements (SLAs).

Cloud providers currently offer three distinct paradigm tiers for hosting ML inference workloads:
1. **Infrastructure-as-a-Service (IaaS):** Always-on dedicated virtual machine (VM) instances (e.g., Docker containers on Amazon EC2). This paradigm offers deterministic compute allocation and zero cold starts, but incurs continuous 24/7 billing regardless of traffic volume.
2. **Function-as-a-Service (FaaS):** MicroVM serverless functions (e.g., AWS Lambda). This model provides millisecond-level pay-per-use billing and auto-scaling down to zero ($0 idle cost), but suffers from runtime cold starts and higher marginal costs per invocation.
3. **Platform-as-a-Service (PaaS):** Specialized managed ML endpoints (e.g., Amazon SageMaker Serverless Inference). This tier abstracts infrastructure operations while providing pre-packaged Deep Learning Containers (DLCs) and burst resource capacity.

Despite the practical importance of this architectural choice, cloud practitioners frequently rely on heuristic rules of thumb rather than rigorous mathematical or empirical thresholds. When traffic is sparse, always-on VMs waste capital on idle compute. Conversely, when sustained traffic is routed to serverless endpoints, cumulative per-request fees can quickly exceed the fixed lease of a dedicated VM.

In this work, we present a parameterised cost model and a reproducible empirical testbed that resolves this dilemma. Specifically, we address three core research questions:
* **RQ1 (Latency and Variance):** How do pure server-side inference latency, tail variance, and cold-start provisioning delays compare across IaaS, FaaS, and PaaS architectures running an identical neural network?
* **RQ2 (Mathematical Crossover):** At what precise request arrival rates ($\lambda^*$) do the cost-optimal deployment architectures shift?
* **RQ3 (Sensitivity and Robustness):** How do model duration, memory sizing, and cloud pricing variances shift these crossover boundaries?

---

## II. Related Work

### A. Serverless Cloud Computing
Serverless computing has emerged as a dominant execution model for event-driven computing. Wang et al. conducted extensive characterizations of AWS Lambda, Google Cloud Functions, and Azure Functions, revealing heterogeneity in microVM allocation, CPU throttling, and network throughput. Shahrad et al. analyzed production serverless traces at Microsoft Azure, demonstrating that over 80% of serverless functions are invoked infrequently, highlighting the economic attractiveness of scale-to-zero architectures.

### B. Machine Learning Inference Serving
Dedicated inference serving systems such as Clipper and Triton Inference Server focus on batching, queue management, and GPU hardware saturation. However, deploying these frameworks on dedicated cloud instances requires over-provisioning to accommodate peak traffic, leading to poor resource utilization during lulls. 

To overcome this, recent work has explored serverless ML inference. Romero et al. (INFless) and Yu et al. (Gillis) investigated GPU sharing and model partitioning across serverless workers. While existing literature demonstrates that serverless inference is viable for small-to-medium models, few prior studies empirically contrast IaaS, general FaaS, and specialized ML PaaS within an end-to-end economic cost crossover formulation.

---

## III. System Architecture and Deployment Testbed

To ensure rigorous empirical parity, the identical model architecture, input representation, and inference engine were deployed across all three AWS testbeds in the Sydney (`ap-southeast-2`) region.

### A. Model Pipeline and Serialization
We selected **MobileNetV2**, a standard 53-layer convolutional neural network widely adopted for computer vision classification. The PyTorch model weights were converted to an Open Neural Network Exchange (ONNX) representation with dynamic batching. ONNX Runtime was selected as the execution provider due to its graph fusion, constant folding, and vectorised SIMD instruction kernels. The total serialized model footprint is 14.0 MB. The input tensor is a pre-processed $1 \times 3 \times 224 \times 224$ single-precision floating-point array.

### B. Deployment Testbeds
1. **Option A: EC2 Container (IaaS Baseline):** An Amazon EC2 `t3.small` virtual machine (2 vCPUs, 2.0 GB RAM) running Ubuntu 22.04 LTS. The application is containerized via Docker using `python:3.11-slim`, exposing a high-performance FastAPI asynchronous ASGI server over port 8080. The model is loaded into memory during container initialization.
2. **Option B: AWS Lambda (FaaS Serverless):** An AWS Lambda function configured with 1024 MB RAM running Python 3.11. Invocations are routed directly via an AWS Lambda Function URL (AuthType: NONE), eliminating API Gateway latency and fees.
3. **Option C: SageMaker Serverless (PaaS Managed):** An Amazon SageMaker Serverless endpoint configured with 2048 MB RAM and a concurrency limit of 10. The endpoint utilizes AWS's official PyTorch 2.0.0 CPU Inference Deep Learning Container (DLC) and retrieves model weights via an S3 `model.tar.gz` bundle (12.35 MB).

---

## IV. Analytical Cost Formulation

Let $\lambda$ denote the average request arrival rate (requests per second).

### A. Always-On Container Cost (IaaS)
For a dedicated VM, compute costs are incurred 24/7 regardless of actual request traffic. Let $C_{\text{hour}}$ denote the hourly VM rate ($0.0208/hr for `t3.small`). The cost per 1,000 inferences at rate $\lambda$ is amortized across total throughput:

$$C_{\text{container}}(\lambda) = \frac{C_{\text{hour}}}{3600 \cdot \lambda} \times 1000$$

As $\lambda \to \infty$, $C_{\text{container}}(\lambda) \to 0$ asymptotically, demonstrating the power of fixed-cost amortization under high utilization.

### B. Serverless Cost Formulation (FaaS and PaaS)
Serverless platforms bill strictly for request quantity and memory-duration allocation. Total cost per 1,000 invocations is expressed as:

$$C_{\text{serverless}}(\lambda) = 1000 \times \left( c_{\text{req}} + c_{\text{gb}} \cdot M \cdot \bar{\tau}(\lambda) \right)$$

where $c_{\text{req}}$ is the per-request charge ($0.20 per $10^6$ requests), $c_{\text{gb}}$ is the compute charge per GB-second ($0.00001667 for Lambda; $0.000020 for SageMaker Serverless), and $M$ is memory allocation in GB.

The effective average duration $\bar{\tau}(\lambda)$ blends warm inference duration $\tau_{\text{warm}}$ and cold initialization duration $\tau_{\text{cold}}$:

$$\bar{\tau}(\lambda) = (1 - P_{\text{cold}}(\lambda))\tau_{\text{warm}} + P_{\text{cold}}(\lambda)\tau_{\text{cold}}$$

Following empirical queueing observations, container lifecycle reuse causes cold-start probability to decay exponentially with request rate:

$$P_{\text{cold}}(\lambda) = \min\left(0.5, \max\left(0.01, P_{\text{base}} \cdot e^{-\lambda / \lambda_{\text{decay}}}\right)\right)$$

### C. Crossover Decision Derivation
The mathematical crossover threshold $\lambda^*$ occurs where the cost curves intersect: $C_{\text{container}}(\lambda^*) = C_{\text{serverless}}(\lambda^*)$. Assuming negligible cold starts at moderate traffic ($P_{\text{cold}} \to 0$):

$$\lambda^* = \frac{C_{\text{hour}}}{3600 \cdot \left( c_{\text{req}} + c_{\text{gb}} \cdot M \cdot \tau_{\text{warm}} \right)}$$

---

## V. Empirical Evaluation and Results

### Table I: Empirical 3-Way Benchmark Telemetry (AWS Sydney ap-southeast-2)

| Metric | Option A: EC2 Container | Option B: AWS Lambda | Option C: SageMaker Serverless |
| :--- | :---: | :---: | :---: |
| **AWS Paradigm Tier** | **IaaS** | **FaaS** | **PaaS** |
| **Hardware / Memory** | `t3.small` (2 vCPU, 2 GB) | 1024 MB RAM | 2048 MB RAM |
| **Idle Cost** | $0.0208 / hr ($15.18/mo) | **$0.00 / hr** | **$0.00 / hr** |
| **Mean Inference Time** | **10.31 ms** | **29.18 ms** | **20.84 ms** |
| **Inference Std. Dev ($\sigma$)** | **0.14 ms** | 12.93 ms | 16.61 ms |
| **Median (P50)** | **10.29 ms** | 30.37 ms | 14.85 ms |
| **95th Percentile (P95)** | **10.58 ms** | 53.62 ms | 68.80 ms |
| **Cold Start Delay** | **0.0 ms** | 413.2 ms | 155.6 ms |
| **Max Throughput** | **18.7 req/s** | 5.6 req/s | 9.8 req/s |

### Key Findings:
1. **Deterministic Latency on EC2:** EC2 Container achieved the fastest inference latency (10.31 ms) with virtually zero jitter ($\sigma = 0.14$ ms) due to persistent RAM residency.
2. **CPU Allocation vs. Inference Speed:** Lambda at 1024 MB receives ~0.6 vCPU credits, causing 29.18 ms inference. SageMaker Serverless at 2048 MB receives 2 full burst vCPUs, executing inference in 20.84 ms (1.4× faster than Lambda).
3. **Cold-Start Differential:** SageMaker Serverless exhibited a faster model initialization (155.6 ms) than Lambda (413.2 ms), as SageMaker reuses pre-cached DLC base container layers.

---

## VI. Cost Crossover and Decision Boundaries

Solving the cost equilibrium equations yields three key operational decision boundaries:
* **$\lambda_1^* = 3.190$ req/s (191.4 req/min) — Lambda vs. EC2:** Below 3.19 req/s (**Zone I**), AWS Lambda is optimal, reducing monthly expenditure by up to **98.6%** for sparse workloads.
* **$\lambda_2^* = 5.185$ req/s (311.1 req/min) — SageMaker Serv. vs. EC2:** Between 3.19 and 5.19 req/s (**Zone II**), SageMaker Serverless provides lower cost than an always-on VM while offering faster inference than Lambda without VM management.
* **$\lambda_3^* = 16.603$ req/s (996.2 req/min) — SageMaker Serv. vs. Real-Time:** Within the managed SageMaker ecosystem, serverless endpoints remain more economical than a dedicated `ml.t2.medium` real-time endpoint ($0.065/hr) until throughput exceeds 16.6 req/s.

Beyond $\lambda_2^* = 5.185$ req/s (**Zone III**), the EC2 Docker container dominates unconditionally, providing lower cost per transaction, 2.4× faster inference, and immunity to cold starts.

---

## VII. Conclusion

This paper presented an empirical and analytical investigation of machine learning inference deployment modes across IaaS (EC2), FaaS (Lambda), and PaaS (SageMaker Serverless). Our findings confirm that serverless is mathematically optimal for sparse, intermittent workloads ($\lambda < 3.19$ req/s), saving up to 98% of operational costs by eliminating idle allocations. However, for sustained traffic ($\lambda > 5.19$ req/s), always-on containers deliver superior cost efficiency, 2.4× lower latency, and zero cold-start variance.
