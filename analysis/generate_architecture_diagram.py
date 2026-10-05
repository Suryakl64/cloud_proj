"""
Generates a publication-grade architecture diagram for the ML Inference Crossover Project.
Outputs: analysis/results/figures/architecture_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_diagram(output_path="analysis/results/figures/architecture_diagram.png"):
    fig = plt.figure(figsize=(16, 9.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Colors (Modern Executive Cloud Palette)
    c_bg = "#F8FAFC"          # Slate 50
    c_border = "#E2E8F0"      # Slate 200
    c_aws = "#232F3E"         # AWS Dark Blue
    c_aws_orange = "#FF9900"  # AWS Orange
    c_ec2 = "#3B82F6"         # Blue
    c_lambda = "#F59E0B"      # Amber
    c_client = "#6366F1"      # Indigo
    c_eval = "#10B981"        # Emerald
    c_card_bg = "#FFFFFF"     # White
    c_text_dark = "#0F172A"   # Slate 900
    c_text_muted = "#64748B"  # Slate 500

    fig.patch.set_facecolor(c_bg)

    # Title & Subtitle Header
    ax.text(8, 9.1, "System Architecture: ML Inference Deployment Dual Testbed", 
            ha="center", va="center", fontsize=19, fontweight="bold", color=c_text_dark)
    ax.text(8, 8.7, "Comparative Evaluation & Cost Crossover Framework for MobileNetV2 (ONNX Runtime)", 
            ha="center", va="center", fontsize=12, color=c_text_muted)

    def draw_card(x, y, w, h, title="", subtitle="", border_color="#CBD5E1", fill_color=c_card_bg, title_color=c_text_dark):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                             edgecolor=border_color, facecolor=fill_color, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        if title:
            ax.text(x + 0.25, y + h - 0.35, title, fontsize=12, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + 0.25, y + h - 0.65, subtitle, fontsize=9.5, color=c_text_muted, zorder=3)
        return box

    # ─────────────────────────────────────────────────────────────
    # SECTION 1: CLIENT & LOAD GENERATOR (Left)
    # ─────────────────────────────────────────────────────────────
    draw_card(0.6, 1.8, 3.2, 6.4, "Client & Load Generator", "Evaluation Harness", border_color=c_client)
    
    # Sub-items in Client
    sub_items_client = [
        ("Async Benchmark Runner", "loadtest/benchmark_live.py\nHTTPX Async concurrency (1-20)", "#EEF2FF", c_client),
        ("Workload Profiler", "Synthetic Poisson / Burst traffic\nParametric rate sweeps", "#F1F5F9", c_text_dark),
        ("Telemetry Collector", "Latency logger (P50/P95/P99)\nCold-start detection engine", "#F1F5F9", c_text_dark),
        ("ONNX Input Payload", "Synthesized 224x224x3 Tensor\nRaw JSON vector data", "#F1F5F9", c_text_dark)
    ]
    cur_y = 6.4
    for title, desc, bg, tc in sub_items_client:
        box = FancyBboxPatch((0.85, cur_y), 2.7, 1.05, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#CBD5E1", facecolor=bg, linewidth=1, zorder=3)
        ax.add_patch(box)
        ax.text(1.0, cur_y + 0.75, title, fontsize=9.5, fontweight="bold", color=tc, zorder=4)
        ax.text(1.0, cur_y + 0.35, desc, fontsize=7.5, color=c_text_muted, zorder=4)
        cur_y -= 1.3

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: AWS CLOUD REGION (Center Large Container)
    # ─────────────────────────────────────────────────────────────
    aws_bg = FancyBboxPatch((4.2, 1.8), 7.6, 6.4, boxstyle="round,pad=0.08,rounding_size=0.2",
                            edgecolor=c_aws_orange, facecolor="#F8FAFC", linewidth=2, linestyle="--", zorder=1)
    ax.add_patch(aws_bg)
    
    # AWS Region Label Badge
    aws_badge = FancyBboxPatch((4.4, 7.8), 3.2, 0.45, boxstyle="round,pad=0.05,rounding_size=0.1",
                               edgecolor=c_aws, facecolor=c_aws, zorder=3)
    ax.add_patch(aws_badge)
    ax.text(6.0, 8.02, "AWS Cloud Region (ap-southeast-2 Sydney)", ha="center", va="center", 
            fontsize=9.5, fontweight="bold", color="#FFFFFF", zorder=4)

    # ── BRANCH A: EC2 CONTAINER (Top Center) ──
    draw_card(4.5, 4.9, 7.0, 2.7, "Baseline: Always-On Container VM", "AWS EC2 (t3.small: 2 vCPU, 2 GB RAM) · $15.18/mo", 
              border_color=c_ec2, title_color="#1E40AF")
    
    # EC2 inner boxes
    ec2_boxes = [
        ("Security Group", "Port 8080 Open\nPublic IP Ingress", 4.7, 5.15, 1.8, 1.3),
        ("Docker Engine", "python:3.11-slim\nFastAPI + Uvicorn", 6.8, 5.15, 2.1, 1.3),
        ("ONNX Runtime", "MobileNetV2 (RAM)\n10.28 ms mean (σ=0.3)", 9.2, 5.15, 2.1, 1.3)
    ]
    for b_title, b_sub, bx, by, bw, bh in ec2_boxes:
        box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#93C5FD", facecolor="#EFF6FF", linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(bx + 0.15, by + bh - 0.35, b_title, fontsize=9.5, fontweight="bold", color="#1E3A8A", zorder=4)
        ax.text(bx + 0.15, by + bh - 0.85, b_sub, fontsize=8, color="#3B82F6", zorder=4)

    # ── BRANCH B: SERVERLESS LAMBDA (Bottom Center) ──
    draw_card(4.5, 2.0, 7.0, 2.7, "Proposed: Serverless FaaS", "AWS Lambda (1024 MB RAM, Python 3.11) · Pay-per-use", 
              border_color=c_lambda, title_color="#B45309")
    
    # Lambda inner boxes
    lambda_boxes = [
        ("Lambda Function URL", "AuthType: NONE\nCORS / HTTPS endpoint", 4.7, 2.25, 1.8, 1.3),
        ("Runtime & S3 Package", "/var/task handler\nS3 Deployment Zip (67MB)", 6.8, 2.25, 2.1, 1.3),
        ("Patched ONNX Engine", "IR v9 Byte-patched\nCold: 394ms | Warm: 28ms", 9.2, 2.25, 2.1, 1.3)
    ]
    for b_title, b_sub, bx, by, bw, bh in lambda_boxes:
        box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#FDE68A", facecolor="#FEFCE8", linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(bx + 0.15, by + bh - 0.35, b_title, fontsize=9.5, fontweight="bold", color="#92400E", zorder=4)
        ax.text(bx + 0.15, by + bh - 0.85, b_sub, fontsize=8, color="#D97706", zorder=4)

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: ANALYSIS & CROSSOVER ENGINE (Right)
    # ─────────────────────────────────────────────────────────────
    draw_card(12.2, 1.8, 3.2, 6.4, "Analysis & Crossover", "Evaluation Pipeline", border_color=c_eval)

    sub_items_eval = [
        ("Empirical Log", "live_benchmark.csv\n80 EC2 & 71 Lambda runs", "#ECFDF5", c_eval),
        ("Crossover Solver", "λ* = 3.190 req/s\nBelow: Lambda wins\nAbove: EC2 wins", "#F0FDF4", "#065F46"),
        ("Sensitivity Sweeps", "Duration (10-200ms)\nMemory (128-3008MB)\nVM pricing curves", "#F8FAFC", c_text_dark),
        ("Visual Artifacts", "6 Publication figures\nCost & Latency curves", "#F8FAFC", c_text_dark)
    ]
    cur_y = 6.4
    for title, desc, bg, tc in sub_items_eval:
        box = FancyBboxPatch((12.45, cur_y), 2.7, 1.05, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#A7F3D0", facecolor=bg, linewidth=1, zorder=3)
        ax.add_patch(box)
        ax.text(12.6, cur_y + 0.75, title, fontsize=9.5, fontweight="bold", color=tc, zorder=4)
        ax.text(12.6, cur_y + 0.35, desc, fontsize=7.5, color=c_text_muted, zorder=4)
        cur_y -= 1.3

    # ─────────────────────────────────────────────────────────────
    # CONNECTING ARROWS & TRAFFIC FLOW
    # ─────────────────────────────────────────────────────────────
    arrow_props_ec2 = dict(arrowstyle="-|>", color=c_ec2, lw=2.2, mutation_scale=16)
    arrow_props_lam = dict(arrowstyle="-|>", color=c_lambda, lw=2.2, mutation_scale=16)
    arrow_props_eval = dict(arrowstyle="-|>", color=c_eval, lw=2.2, mutation_scale=16)

    # Client to EC2
    ax.annotate("", xy=(4.5, 6.2), xytext=(3.8, 6.2), arrowprops=arrow_props_ec2, zorder=5)
    ax.text(4.15, 6.45, "HTTP POST\n:8080/predict", ha="center", fontsize=7.5, fontweight="bold", color=c_ec2)

    # Client to Lambda
    ax.annotate("", xy=(4.5, 3.4), xytext=(3.8, 3.4), arrowprops=arrow_props_lam, zorder=5)
    ax.text(4.15, 3.65, "HTTPS POST\nFunction URL", ha="center", fontsize=7.5, fontweight="bold", color=c_lambda)

    # EC2 inner flow
    ax.annotate("", xy=(6.8, 5.8), xytext=(6.5, 5.8), arrowprops=dict(arrowstyle="->", color=c_ec2, lw=1.5), zorder=5)
    ax.annotate("", xy=(9.2, 5.8), xytext=(8.9, 5.8), arrowprops=dict(arrowstyle="->", color=c_ec2, lw=1.5), zorder=5)

    # Lambda inner flow
    ax.annotate("", xy=(6.8, 2.9), xytext=(6.5, 2.9), arrowprops=dict(arrowstyle="->", color=c_lambda, lw=1.5), zorder=5)
    ax.annotate("", xy=(9.2, 2.9), xytext=(8.9, 2.9), arrowprops=dict(arrowstyle="->", color=c_lambda, lw=1.5), zorder=5)

    # EC2 to Evaluation
    ax.annotate("", xy=(12.2, 5.8), xytext=(11.5, 5.8), arrowprops=arrow_props_eval, zorder=5)
    # Lambda to Evaluation
    ax.annotate("", xy=(12.2, 3.2), xytext=(11.5, 3.2), arrowprops=arrow_props_eval, zorder=5)
    ax.text(11.85, 4.5, "Performance\n& Cost Telemetry", ha="center", fontsize=7.5, fontweight="bold", color=c_eval)

    # Bottom Footer Summary Ribbon
    footer = FancyBboxPatch((0.6, 0.4), 14.8, 0.9, boxstyle="round,pad=0.04,rounding_size=0.12",
                            edgecolor="#CBD5E1", facecolor="#FFFFFF", linewidth=1.2, zorder=2)
    ax.add_patch(footer)
    ax.text(1.0, 0.85, "Key Takeaway:", fontsize=9.5, fontweight="bold", color=c_text_dark, zorder=3)
    ax.text(2.35, 0.85, "Below λ* = 3.190 req/s, AWS Lambda avoids fixed VM idle costs ($15.18/mo) with up to 98% savings.", 
            fontsize=9, color=c_text_muted, zorder=3)
    ax.text(2.35, 0.55, "Above λ* = 3.190 req/s, EC2 Docker container provides 3x lower per-request cost and zero cold-start variance (10.2 ms vs 394 ms).", 
            fontsize=9, color=c_text_muted, zorder=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    create_diagram()
