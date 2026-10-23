"""
Generates a clean, simple, slide-ready 3-way architecture diagram for Final Review PPT.
Outputs: analysis/results/figures/architecture_simple.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_simple_diagram(output_path="analysis/results/figures/architecture_simple.png"):
    fig = plt.figure(figsize=(15.2, 8.6), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")

    # Colors
    c_client = "#4F46E5"    # Indigo
    c_ec2 = "#2563EB"       # Royal Blue
    c_lambda = "#EA580C"    # Warm Amber / Orange
    c_sm = "#0D9488"        # Teal
    c_eval = "#059669"      # Emerald Green
    c_aws_bg = "#F8FAFC"    # Clean light slate
    c_text_dark = "#0F172A"
    c_text_muted = "#64748B"

    # Header Title
    ax.text(7.6, 8.15, "System Architecture: 3-Way ML Inference Deployment Testbed", 
            ha="center", va="center", fontsize=18, fontweight="bold", color=c_text_dark)
    ax.text(7.6, 7.75, "Empirical Evaluation of IaaS (EC2) vs. FaaS (Lambda) vs. PaaS (SageMaker Serverless) on AWS", 
            ha="center", va="center", fontsize=11.5, color=c_text_muted)

    # 1. CLIENT (Left Box)
    client_box = FancyBboxPatch((0.6, 1.2), 2.7, 6.0, boxstyle="round,pad=0.08,rounding_size=0.18",
                                edgecolor=c_client, facecolor="#EEF2FF", linewidth=2, zorder=2)
    ax.add_patch(client_box)
    ax.text(1.95, 6.6, "Load Generator\n(Client)", ha="center", va="center", 
            fontsize=13, fontweight="bold", color="#1E1B4B")
    ax.text(1.95, 5.2, "• Async Benchmark\n  Runner (HTTPX)\n• Concurrency:\n  1 to 20 workers\n• Synthetic Poisson\n  & Burst traffic\n• MobileNetV2 Tensor\n  (224x224x3 Float)", 
            ha="center", va="center", fontsize=9.5, color="#3730A3", linespacing=1.55)
    
    # Client Sub-Badge
    pill_client = FancyBboxPatch((0.85, 1.5), 2.2, 0.65, boxstyle="round,pad=0.04,rounding_size=0.1",
                                 facecolor="#E0E7FF", edgecolor="#A5B4FC", linewidth=1, zorder=3)
    ax.add_patch(pill_client)
    ax.text(1.95, 1.82, "Telemetry Logger\n(RTT & Server ms)", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#312E81", zorder=4)

    # 2. AWS CLOUD BOX (Center)
    aws_box = FancyBboxPatch((3.9, 0.8), 7.2, 6.6, boxstyle="round,pad=0.08,rounding_size=0.2",
                             edgecolor="#FF9900", facecolor=c_aws_bg, linewidth=2, linestyle="--", zorder=1)
    ax.add_patch(aws_box)
    
    # AWS Label
    ax.text(7.5, 7.15, "AWS Cloud Region (Sydney ap-southeast-2)", ha="center", va="center", 
            fontsize=11.5, fontweight="bold", color="#FF9900")

    # 2A. TOP BRANCH: EC2 CONTAINER (IaaS)
    ec2_box = FancyBboxPatch((4.2, 5.0), 6.6, 1.75, boxstyle="round,pad=0.06,rounding_size=0.14",
                             edgecolor=c_ec2, facecolor="#EFF6FF", linewidth=2, zorder=2)
    ax.add_patch(ec2_box)
    ax.text(4.5, 6.35, "Option A: EC2 Container (IaaS Baseline)", fontsize=11.5, fontweight="bold", color="#1E3A8A")
    ax.text(4.5, 5.85, "• AWS EC2 (t3.small: 2 vCPU, 2 GB RAM) + Docker + FastAPI\n• Pre-warmed MobileNetV2 ONNX Runtime Engine", 
            fontsize=9, color="#1D4ED8", linespacing=1.35)
    # Metric Pill
    pill_ec2 = FancyBboxPatch((8.3, 5.15), 2.3, 0.52, boxstyle="round,pad=0.04,rounding_size=0.1",
                              facecolor="#DBEAFE", edgecolor="#93C5FD", linewidth=1, zorder=3)
    ax.add_patch(pill_ec2)
    ax.text(9.45, 5.41, "10.31 ms | $15.18/mo fixed", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#1E40AF", zorder=4)

    # 2B. MIDDLE BRANCH: AWS LAMBDA (FaaS)
    lam_box = FancyBboxPatch((4.2, 2.95), 6.6, 1.75, boxstyle="round,pad=0.06,rounding_size=0.14",
                             edgecolor=c_lambda, facecolor="#FFF7ED", linewidth=2, zorder=2)
    ax.add_patch(lam_box)
    ax.text(4.5, 4.30, "Option B: AWS Lambda (FaaS Serverless)", fontsize=11.5, fontweight="bold", color="#9A3412")
    ax.text(4.5, 3.80, "• Lambda Function URL + 1024 MB RAM (Python 3.11 runtime)\n• On-demand byte-patched ONNX (Cold: 413ms | Warm: 29ms)", 
            fontsize=9, color="#C2410C", linespacing=1.35)
    # Metric Pill
    pill_lam = FancyBboxPatch((8.3, 3.10), 2.3, 0.52, boxstyle="round,pad=0.04,rounding_size=0.1",
                              facecolor="#FED7AA", edgecolor="#FDBA74", linewidth=1, zorder=3)
    ax.add_patch(pill_lam)
    ax.text(9.45, 3.36, "29.18 ms | $0.00 Idle", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#9A3412", zorder=4)

    # 2C. BOTTOM BRANCH: SAGEMAKER SERVERLESS (PaaS)
    sm_box = FancyBboxPatch((4.2, 0.95), 6.6, 1.75, boxstyle="round,pad=0.06,rounding_size=0.14",
                            edgecolor=c_sm, facecolor="#F0FDFA", linewidth=2, zorder=2)
    ax.add_patch(sm_box)
    ax.text(4.5, 2.30, "Option C: SageMaker Serverless (PaaS Managed)", fontsize=11.5, fontweight="bold", color="#115E59")
    ax.text(4.5, 1.80, "• PyTorch DLC + S3 model.tar.gz + 2048 MB RAM (Max conc: 10)\n• Managed AWS PaaS endpoint (Cold: 155ms | Warm: 21ms)", 
            fontsize=9, color="#0F766E", linespacing=1.35)
    # Metric Pill
    pill_sm = FancyBboxPatch((8.3, 1.10), 2.3, 0.52, boxstyle="round,pad=0.04,rounding_size=0.1",
                             facecolor="#CCFBF1", edgecolor="#99F6E4", linewidth=1, zorder=3)
    ax.add_patch(pill_sm)
    ax.text(9.45, 1.36, "20.84 ms | $0.00 Idle", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#0F766E", zorder=4)

    # 3. EVALUATION / DECISION (Right Box)
    eval_box = FancyBboxPatch((11.7, 1.2), 2.9, 6.0, boxstyle="round,pad=0.08,rounding_size=0.18",
                              edgecolor=c_eval, facecolor="#ECFDF5", linewidth=2, zorder=2)
    ax.add_patch(eval_box)
    ax.text(13.15, 6.6, "Evaluation &\nDecision Engine", ha="center", va="center", 
            fontsize=13, fontweight="bold", color="#064E3B")
    ax.text(13.15, 5.0, "• Crossover Solver:\n  λ1* = 3.19 req/s\n  (Lambda vs EC2)\n\n  λ2* = 5.19 req/s\n  (SageMaker vs EC2)\n\n  λ3* = 16.6 req/s\n  (SM Serverless vs RT)", 
            ha="center", va="center", fontsize=9.2, color="#047857", linespacing=1.35)
    
    # Decision Pill
    pill_eval = FancyBboxPatch((12.0, 1.5), 2.3, 1.1, boxstyle="round,pad=0.04,rounding_size=0.1",
                               facecolor="#D1FAE5", edgecolor="#6EE7B7", linewidth=1, zorder=3)
    ax.add_patch(pill_eval)
    ax.text(13.15, 2.05, "Architectural Verdict:\n<3.2 req/s: Lambda\n3.2-5.2 req/s: SM Serv.\n>5.2 req/s: EC2", 
            ha="center", va="center", fontsize=8, fontweight="bold", color="#065F46", zorder=4)

    # ARROWS & CONNECTORS
    arrow_ec2 = dict(arrowstyle="-|>", color=c_ec2, lw=2.2, mutation_scale=15)
    arrow_lam = dict(arrowstyle="-|>", color=c_lambda, lw=2.2, mutation_scale=15)
    arrow_sm = dict(arrowstyle="-|>", color=c_sm, lw=2.2, mutation_scale=15)
    arrow_eval = dict(arrowstyle="-|>", color=c_eval, lw=2.2, mutation_scale=15)

    # Client to EC2
    ax.annotate("", xy=(4.2, 5.85), xytext=(3.3, 5.85), arrowprops=arrow_ec2, zorder=5)
    ax.text(3.75, 6.05, "HTTP", ha="center", fontsize=8, fontweight="bold", color=c_ec2)

    # Client to Lambda
    ax.annotate("", xy=(4.2, 3.80), xytext=(3.3, 3.80), arrowprops=arrow_lam, zorder=5)
    ax.text(3.75, 4.00, "HTTPS", ha="center", fontsize=8, fontweight="bold", color=c_lambda)

    # Client to SageMaker
    ax.annotate("", xy=(4.2, 1.80), xytext=(3.3, 1.80), arrowprops=arrow_sm, zorder=5)
    ax.text(3.75, 2.00, "Boto3/CLI", ha="center", fontsize=7.5, fontweight="bold", color=c_sm)

    # EC2 to Evaluation
    ax.annotate("", xy=(11.7, 5.85), xytext=(10.8, 5.85), arrowprops=arrow_eval, zorder=5)

    # Lambda to Evaluation
    ax.annotate("", xy=(11.7, 3.80), xytext=(10.8, 3.80), arrowprops=arrow_eval, zorder=5)

    # SageMaker to Evaluation
    ax.annotate("", xy=(11.7, 1.80), xytext=(10.8, 1.80), arrowprops=arrow_eval, zorder=5)

    ax.text(11.25, 3.95, "Telemetry\nLogs", ha="center", fontsize=8, fontweight="bold", color=c_eval)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    print(f"Simple 3-way architecture diagram saved to {output_path}")

if __name__ == "__main__":
    create_simple_diagram()
