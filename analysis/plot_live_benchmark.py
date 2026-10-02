"""
Generate comparative visualization of live Serverless vs. Container benchmark results.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Find latest live benchmark file
csv_files = sorted(glob.glob("analysis/results/raw/live_benchmark_*.csv"))
if not csv_files:
    print("No benchmark files found.")
    exit(1)

latest_csv = csv_files[-1]
df = pd.read_csv(latest_csv)
df = df[df["status_code"] == 200]

palette = {"serverless": "#ff7f0e", "container": "#1f77b4"}

# Plot 1: Round Trip Latency vs Mode
sns.boxplot(
    data=df,
    x="mode",
    y="round_trip_ms",
    palette=palette,
    ax=axes[0],
    boxprops=dict(alpha=0.8),
    showmeans=True,
    meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"}
)
axes[0].set_title("Live Round-Trip Latency (End-to-End)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Deployment Mode", fontsize=11)
axes[0].set_ylabel("Latency (ms)", fontsize=11)

# Plot 2: Pure Server-Side Inference Time
sns.barplot(
    data=df,
    x="mode",
    y="server_inference_time_ms",
    palette=palette,
    ax=axes[1],
    capsize=0.1,
    err_kws={"linewidth": 1.5}
)
axes[1].set_title("Live Server-Side Model Inference Time", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Deployment Mode", fontsize=11)
axes[1].set_ylabel("Inference Time (ms)", fontsize=11)

for ax in axes:
    ax.tick_params(labelsize=10)

plt.suptitle("Empirical Validation: AWS Lambda (Serverless) vs. EC2 Container (MobileNetV2)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

os.makedirs("analysis/results/figures", exist_ok=True)
output_img = "analysis/results/figures/live_comparison.png"
plt.savefig(output_img, dpi=300, bbox_inches="tight")
print(f"[OK] Live comparison plot saved to {output_img}")
