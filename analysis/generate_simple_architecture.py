"""
Generates a clean, simple, slide-ready architecture diagram for Review 2 PPT.
Outputs: analysis/results/figures/architecture_simple.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_simple_diagram(output_path="analysis/results/figures/architecture_simple.png"):
    fig = plt.figure(figsize=(14, 7), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")

    # Colors
    c_client = "#4F46E5"    # Indigo
    c_ec2 = "#2563EB"       # Royal Blue
    c_lambda = "#D97706"    # Warm Amber
    c_eval = "#059669"      # Emerald Green
    c_aws_bg = "#F8FAFC"    # Clean light slate
    c_aws_border = "#E2E8F0"
    c_text_dark = "#0F172A"

    # Title
    ax.text(7, 6.45, "System Architecture: ML Inference Deployment", 
            ha="center", va="center", fontsize=18, fontweight="bold", color=c_text_dark)
    ax.text(7, 6.05, "Comparing Always-On Container vs. Serverless Function on AWS", 
            ha="center", va="center", fontsize=12, color="#64748B")

    # 1. CLIENT (Left)
    client_box = FancyBboxPatch((0.6, 2.0), 2.5, 3.2, boxstyle="round,pad=0.08,rounding_size=0.18",
                                edgecolor=c_client, facecolor="#EEF2FF", linewidth=2, zorder=2)
    ax.add_patch(client_box)
    ax.text(1.85, 4.4, "Load Generator\n(Client)", ha="center", va="center", 
            fontsize=13, fontweight="bold", color="#1E1B4B")
    ax.text(1.85, 3.4, "• Synthetic Traffic\n• 1 to 20 Concurrency\n• Image Tensor Data", 
            ha="center", va="center", fontsize=10, color="#3730A3", linespacing=1.6)

    # 2. AWS CLOUD BOX (Center)
    aws_box = FancyBboxPatch((3.7, 0.8), 6.6, 4.8, boxstyle="round,pad=0.08,rounding_size=0.2",
                             edgecolor="#FF9900", facecolor=c_aws_bg, linewidth=2, linestyle="--", zorder=1)
    ax.add_patch(aws_box)
    
    # AWS Label
    ax.text(7.0, 5.25, "AWS Cloud (Sydney ap-southeast-2)", ha="center", va="center", 
            fontsize=11, fontweight="bold", color="#FF9900")

    # 2A. TOP BRANCH: EC2 CONTAINER
    ec2_box = FancyBboxPatch((4.0, 3.2), 6.0, 1.8, boxstyle="round,pad=0.06,rounding_size=0.15",
                             edgecolor=c_ec2, facecolor="#EFF6FF", linewidth=2, zorder=2)
    ax.add_patch(ec2_box)
    ax.text(4.3, 4.55, "Option A: EC2 Container (Baseline)", fontsize=12, fontweight="bold", color="#1E3A8A")
    ax.text(4.3, 4.15, "• AWS EC2 (t3.small) + Docker + FastAPI\n• Pre-warmed MobileNetV2 ONNX", 
            fontsize=9.5, color="#1D4ED8", linespacing=1.4)
    # Metric Pill
    pill_ec2 = FancyBboxPatch((7.7, 3.4), 2.1, 0.55, boxstyle="round,pad=0.04,rounding_size=0.1",
                              facecolor="#DBEAFE", edgecolor="#93C5FD", linewidth=1, zorder=3)
    ax.add_patch(pill_ec2)
    ax.text(8.75, 3.67, "10.2 ms | $15.18/mo", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#1E40AF", zorder=4)

    # 2B. BOTTOM BRANCH: AWS LAMBDA
    lam_box = FancyBboxPatch((4.0, 1.1), 6.0, 1.8, boxstyle="round,pad=0.06,rounding_size=0.15",
                             edgecolor=c_lambda, facecolor="#FEFCE8", linewidth=2, zorder=2)
    ax.add_patch(lam_box)
    ax.text(4.3, 2.45, "Option B: AWS Lambda (Serverless)", fontsize=12, fontweight="bold", color="#78350F")
    ax.text(4.3, 2.05, "• Lambda Function URL + 1024 MB RAM\n• On-demand MobileNetV2 ONNX", 
            fontsize=9.5, color="#92400E", linespacing=1.4)
    # Metric Pill
    pill_lam = FancyBboxPatch((7.7, 1.3), 2.1, 0.55, boxstyle="round,pad=0.04,rounding_size=0.1",
                              facecolor="#FEF08A", edgecolor="#FDE047", linewidth=1, zorder=3)
    ax.add_patch(pill_lam)
    ax.text(8.75, 1.57, "27.9 ms | $0.00 Idle", ha="center", va="center", 
            fontsize=8.5, fontweight="bold", color="#854D0E", zorder=4)

    # 3. EVALUATION / DECISION (Right)
    eval_box = FancyBboxPatch((10.9, 2.0), 2.5, 3.2, boxstyle="round,pad=0.08,rounding_size=0.18",
                              edgecolor=c_eval, facecolor="#ECFDF5", linewidth=2, zorder=2)
    ax.add_patch(eval_box)
    ax.text(12.15, 4.4, "Evaluation &\nDecision Model", ha="center", va="center", 
            fontsize=13, fontweight="bold", color="#064E3B")
    ax.text(12.15, 3.4, "• Cost vs. Latency\n• Crossover Point:\n  λ* = 3.19 req/s", 
            ha="center", va="center", fontsize=10, color="#047857", linespacing=1.6)

    # ARROWS & CONNECTORS
    arrow_ec2 = dict(arrowstyle="-|>", color=c_ec2, lw=2.2, mutation_scale=15)
    arrow_lam = dict(arrowstyle="-|>", color=c_lambda, lw=2.2, mutation_scale=15)
    arrow_eval = dict(arrowstyle="-|>", color=c_eval, lw=2.2, mutation_scale=15)

    # Client to EC2
    ax.annotate("", xy=(4.0, 4.1), xytext=(3.1, 4.1), arrowprops=arrow_ec2, zorder=5)
    ax.text(3.55, 4.3, "HTTP", ha="center", fontsize=8.5, fontweight="bold", color=c_ec2)

    # Client to Lambda
    ax.annotate("", xy=(4.0, 2.0), xytext=(3.1, 2.0), arrowprops=arrow_lam, zorder=5)
    ax.text(3.55, 2.2, "HTTPS", ha="center", fontsize=8.5, fontweight="bold", color=c_lambda)

    # EC2 to Evaluation
    ax.annotate("", xy=(10.9, 4.1), xytext=(10.0, 4.1), arrowprops=arrow_eval, zorder=5)

    # Lambda to Evaluation
    ax.annotate("", xy=(10.9, 2.0), xytext=(10.0, 2.0), arrowprops=arrow_eval, zorder=5)

    ax.text(10.45, 3.1, "Logs", ha="center", fontsize=8.5, fontweight="bold", color=c_eval)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    print(f"Simple architecture diagram saved to {output_path}")

if __name__ == "__main__":
    create_simple_diagram()
