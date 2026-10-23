"""
Generates a publication-grade 3-way architecture diagram for the ML Inference Crossover Project.
Outputs: analysis/results/figures/architecture_diagram.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_diagram(output_path="analysis/results/figures/architecture_diagram.png"):
    fig = plt.figure(figsize=(16.8, 11.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 11.2)
    ax.axis("off")

    # Colors (Modern Executive Cloud Palette)
    c_bg = "#F8FAFC"          # Slate 50
    c_aws = "#232F3E"         # AWS Dark Blue
    c_aws_orange = "#FF9900"  # AWS Orange
    c_ec2 = "#2563EB"         # Blue
    c_lambda = "#EA580C"      # Amber / Orange
    c_sm = "#0D9488"          # Teal
    c_client = "#4F46E5"      # Indigo
    c_eval = "#059669"        # Emerald
    c_card_bg = "#FFFFFF"     # White
    c_text_dark = "#0F172A"   # Slate 900
    c_text_muted = "#64748B"  # Slate 500

    fig.patch.set_facecolor(c_bg)

    # Title & Subtitle Header
    ax.text(8.4, 10.75, "System Architecture: 3-Way ML Inference Deployment Testbed", 
            ha="center", va="center", fontsize=19, fontweight="bold", color=c_text_dark)
    ax.text(8.4, 10.35, "Comparative Evaluation & Cost Crossover Framework for MobileNetV2 (ONNX Runtime) across IaaS, FaaS, and PaaS", 
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
    draw_card(0.6, 1.4, 3.2, 8.5, "Client & Load Generator", "Evaluation Harness", border_color=c_client)
    
    sub_items_client = [
        ("Async Benchmark Runner", "loadtest/benchmark_live.py\nHTTPX & Boto3 asyncio pool", "#EEF2FF", c_client),
        ("Workload Profiler", "Synthetic Poisson & burst traffic\nConcurrency sweep (1, 5, 10, 20)", "#F1F5F9", c_text_dark),
        ("Telemetry Engine", "Live RTT & server latency log\nCold-start detection engine", "#F1F5F9", c_text_dark),
        ("Input Tensor Payload", "MobileNetV2: 224x224x3 Float32\nOptimized binary/JSON payload", "#F1F5F9", c_text_dark),
        ("Empirical CSV Logger", "Timestamp, mode, RTT, server_ms\nSaved to analysis/results/raw/", "#F1F5F9", c_text_dark)
    ]
    cur_y = 8.35
    for title, desc, bg, tc in sub_items_client:
        box = FancyBboxPatch((0.85, cur_y), 2.7, 1.05, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#CBD5E1", facecolor=bg, linewidth=1, zorder=3)
        ax.add_patch(box)
        ax.text(1.0, cur_y + 0.72, title, fontsize=9.5, fontweight="bold", color=tc, zorder=4)
        ax.text(1.0, cur_y + 0.32, desc, fontsize=7.5, color=c_text_muted, zorder=4)
        cur_y -= 1.35

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: AWS CLOUD REGION (Center Large Container)
    # ─────────────────────────────────────────────────────────────
    aws_bg = FancyBboxPatch((4.2, 1.4), 8.0, 8.5, boxstyle="round,pad=0.08,rounding_size=0.2",
                            edgecolor=c_aws_orange, facecolor="#F8FAFC", linewidth=2, linestyle="--", zorder=1)
    ax.add_patch(aws_bg)
    
    # AWS Region Label Badge
    aws_badge = FancyBboxPatch((4.4, 9.45), 3.6, 0.45, boxstyle="round,pad=0.05,rounding_size=0.1",
                               edgecolor=c_aws, facecolor=c_aws, zorder=3)
    ax.add_patch(aws_badge)
    ax.text(6.2, 9.68, "AWS Cloud Region (ap-southeast-2 Sydney)", ha="center", va="center", 
            fontsize=9.5, fontweight="bold", color="#FFFFFF", zorder=4)

    # ── BRANCH A: EC2 CONTAINER (Top Center) ──
    draw_card(4.5, 7.15, 7.4, 2.2, "Option A: EC2 Container (IaaS Baseline)", 
              "AWS EC2 (t3.small: 2 vCPU, 2 GB RAM) · $15.18/mo fixed", 
              border_color=c_ec2, title_color="#1E40AF")
    
    ec2_boxes = [
        ("Security Group", "Port 8080 Ingress\nPublic IPv4 VPC", 4.7, 7.35, 2.0, 1.15),
        ("Docker Engine", "python:3.11-slim\nFastAPI + Uvicorn", 7.1, 7.35, 2.2, 1.15),
        ("ONNX Runtime", "In-RAM MobileNetV2\n10.31 ms (σ = 0.14 ms)", 9.6, 7.35, 2.1, 1.15)
    ]
    for b_title, b_sub, bx, by, bw, bh in ec2_boxes:
        box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#93C5FD", facecolor="#EFF6FF", linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(bx + 0.12, by + bh - 0.32, b_title, fontsize=9, fontweight="bold", color="#1E3A8A", zorder=4)
        ax.text(bx + 0.12, by + bh - 0.75, b_sub, fontsize=7.5, color="#2563EB", zorder=4)

    # ── BRANCH B: SERVERLESS LAMBDA (Middle Center) ──
    draw_card(4.5, 4.45, 7.4, 2.2, "Option B: AWS Lambda (FaaS Serverless)", 
              "AWS Lambda (1024 MB RAM, Python 3.11) · Pay-per-use ($0 Idle)", 
              border_color=c_lambda, title_color="#9A3412")
    
    lambda_boxes = [
        ("Function URL", "AuthType: NONE\nCORS / HTTPS direct", 4.7, 4.65, 2.0, 1.15),
        ("S3 Deployment Zip", "67MB runtime bundle\n/var/task cold load", 7.1, 4.65, 2.2, 1.15),
        ("Patched Engine", "ONNX IR v9 session\nCold: 413ms | Warm: 29ms", 9.6, 4.65, 2.1, 1.15)
    ]
    for b_title, b_sub, bx, by, bw, bh in lambda_boxes:
        box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#FDBA74", facecolor="#FFF7ED", linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(bx + 0.12, by + bh - 0.32, b_title, fontsize=9, fontweight="bold", color="#9A3412", zorder=4)
        ax.text(bx + 0.12, by + bh - 0.75, b_sub, fontsize=7.5, color="#EA580C", zorder=4)

    # ── BRANCH C: SAGEMAKER SERVERLESS (Bottom Center) ──
    draw_card(4.5, 1.75, 7.4, 2.2, "Option C: SageMaker Serverless (PaaS Managed)", 
              "PyTorch DLC + ONNX Runtime (2048 MB RAM) · Pay-per-use ($0 Idle)", 
              border_color=c_sm, title_color="#115E59")
    
    sm_boxes = [
        ("Serverless Endpoint", "Auto-scale (0 to 10 conc)\nBoto3 InvokeEndpoint", 4.7, 1.95, 2.0, 1.15),
        ("S3 model.tar.gz", "12.35 MB Gzip pkg\nPyTorch 2.0 CPU DLC", 7.1, 1.95, 2.2, 1.15),
        ("Managed ONNX", "Custom inference handler\nCold: 155ms | Warm: 21ms", 9.6, 1.95, 2.1, 1.15)
    ]
    for b_title, b_sub, bx, by, bw, bh in sm_boxes:
        box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#99F6E4", facecolor="#F0FDFA", linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(bx + 0.12, by + bh - 0.32, b_title, fontsize=9, fontweight="bold", color="#0F766E", zorder=4)
        ax.text(bx + 0.12, by + bh - 0.75, b_sub, fontsize=7.5, color="#0D9488", zorder=4)

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: ANALYSIS & CROSSOVER ENGINE (Right)
    # ─────────────────────────────────────────────────────────────
    draw_card(12.6, 1.4, 3.6, 8.5, "Analysis & Crossover Engine", "Evaluation Pipeline", border_color=c_eval)

    sub_items_eval = [
        ("3-Way Empirical Log", "live_benchmark_3way.csv\nEC2, Lambda & SageMaker", "#ECFDF5", c_eval),
        ("Decision Boundary λ1*", "λ1* = 3.190 req/s\nBelow: Lambda wins\nAbove: EC2 wins", "#F0FDF4", "#065F46"),
        ("Decision Boundary λ2*", "λ2* = 5.185 req/s\nBelow: SageMaker Serverless\nAbove: EC2 Container", "#F0FDF4", "#0F766E"),
        ("Decision Boundary λ3*", "λ3* = 16.603 req/s\nBelow: SM Serverless\nAbove: SM Real-Time", "#F0FDF4", "#7C3AED"),
        ("6 Publication Figures", "fig1_cost_crossover.png\nlive_comparison.png, figs 2-5", "#F8FAFC", c_text_dark)
    ]
    cur_y = 8.35
    for title, desc, bg, tc in sub_items_eval:
        box = FancyBboxPatch((12.85, cur_y), 3.1, 1.05, boxstyle="round,pad=0.04,rounding_size=0.1",
                             edgecolor="#A7F3D0", facecolor=bg, linewidth=1, zorder=3)
        ax.add_patch(box)
        ax.text(13.0, cur_y + 0.72, title, fontsize=9.5, fontweight="bold", color=tc, zorder=4)
        ax.text(13.0, cur_y + 0.32, desc, fontsize=7.5, color=c_text_muted, zorder=4)
        cur_y -= 1.35

    # ─────────────────────────────────────────────────────────────
    # CONNECTING ARROWS & TRAFFIC FLOW
    # ─────────────────────────────────────────────────────────────
    arrow_props_ec2 = dict(arrowstyle="-|>", color=c_ec2, lw=2.2, mutation_scale=15)
    arrow_props_lam = dict(arrowstyle="-|>", color=c_lambda, lw=2.2, mutation_scale=15)
    arrow_props_sm = dict(arrowstyle="-|>", color=c_sm, lw=2.2, mutation_scale=15)
    arrow_props_eval = dict(arrowstyle="-|>", color=c_eval, lw=2.2, mutation_scale=15)

    # Client to EC2
    ax.annotate("", xy=(4.5, 8.25), xytext=(3.8, 8.25), arrowprops=arrow_props_ec2, zorder=5)
    ax.text(4.15, 8.45, "HTTP", ha="center", fontsize=7.5, fontweight="bold", color=c_ec2)

    # Client to Lambda
    ax.annotate("", xy=(4.5, 5.55), xytext=(3.8, 5.55), arrowprops=arrow_props_lam, zorder=5)
    ax.text(4.15, 5.75, "HTTPS", ha="center", fontsize=7.5, fontweight="bold", color=c_lambda)

    # Client to SageMaker
    ax.annotate("", xy=(4.5, 2.85), xytext=(3.8, 2.85), arrowprops=arrow_props_sm, zorder=5)
    ax.text(4.15, 3.05, "Boto3", ha="center", fontsize=7.5, fontweight="bold", color=c_sm)

    # EC2 to Evaluation
    ax.annotate("", xy=(12.6, 8.25), xytext=(11.9, 8.25), arrowprops=arrow_props_eval, zorder=5)
    # Lambda to Evaluation
    ax.annotate("", xy=(12.6, 5.55), xytext=(11.9, 5.55), arrowprops=arrow_props_eval, zorder=5)
    # SageMaker to Evaluation
    ax.annotate("", xy=(12.6, 2.85), xytext=(11.9, 2.85), arrowprops=arrow_props_eval, zorder=5)

    ax.text(12.25, 5.65, "Telemetry\nLogs", ha="center", fontsize=7.5, fontweight="bold", color=c_eval)

    # Bottom Footer Summary Ribbon
    footer = FancyBboxPatch((0.6, 0.3), 15.6, 0.85, boxstyle="round,pad=0.04,rounding_size=0.12",
                            edgecolor="#CBD5E1", facecolor="#FFFFFF", linewidth=1.2, zorder=2)
    ax.add_patch(footer)
    ax.text(0.9, 0.72, "Key Architectural Finding:", fontsize=9.5, fontweight="bold", color=c_text_dark, zorder=3)
    ax.text(3.1, 0.72, "Sparse Traffic (λ < 3.19 req/s): AWS Lambda provides up to 98% savings by eliminating VM idle costs ($15.18/mo).", 
            fontsize=8.5, color=c_text_muted, zorder=3)
    ax.text(3.1, 0.45, "Moderate Bursts (3.19 ≤ λ < 5.19 req/s): SageMaker Serverless balances zero idle cost with isolated PaaS burst capacity (20.8ms vs 29.2ms).", 
            fontsize=8.5, color=c_text_muted, zorder=3)
    ax.text(3.1, 0.18, "Sustained Traffic (λ ≥ 5.19 req/s): EC2 Docker Container amortizes fixed costs, providing 2.4x faster latency (10.3ms) and zero cold-start delay.", 
            fontsize=8.5, color=c_text_muted, zorder=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Publication architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    create_diagram()
