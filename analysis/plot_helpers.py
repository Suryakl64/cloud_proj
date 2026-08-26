"""
Publication-quality plotting utilities for the crossover analysis.

Generates 5 key figures:
1. Cost crossover plot (per-1000 inferences vs request rate)
2. Latency distribution (box plots across rate steps)
3. Cold-start analysis (frequency and penalty vs rate)
4. Sensitivity analysis (crossover shift with parameters)
5. Model validation (predicted vs measured with error bars)
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style Configuration ──────────────────────────────────────────────────────

# Publication-quality settings
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
COLORS = {
    "serverless": "#FF6B35",   # Orange
    "container": "#004E89",    # Deep blue
    "managed": "#2A9D8F",      # Teal
    "crossover": "#E63946",    # Red accent
    "grid": "#CCCCCC",
    "background": "#FAFAFA",
}

MODE_LABELS = {
    "serverless": "Serverless (Lambda)",
    "container": "Container (EC2)",
    "managed": "Managed (SageMaker)",
}

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "results", "figures")


def ensure_figure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


# ── Figure 1: Cost Crossover Plot ────────────────────────────────────────────

def plot_cost_crossover(
    sweep_df: pd.DataFrame,
    crossovers: Dict = None,
    comparison_df: pd.DataFrame = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot cost per 1000 inferences vs request rate (log scale).

    Shows the three cost curves and annotates crossover points.

    Args:
        sweep_df: DataFrame with columns: rate_rps, serverless_per_1k,
                  container_per_1k, managed_per_1k
        crossovers: Dict from CrossoverSolver.find_all_crossovers()
        comparison_df: Optional measured data points to overlay
        save_path: Path to save figure (default: figures/cost_crossover.png)
    """
    ensure_figure_dir()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor(COLORS["background"])

    # Plot cost curves
    rates = sweep_df["rate_rps"].values

    for mode, col in [("serverless", "serverless_per_1k"),
                       ("container", "container_per_1k"),
                       ("managed", "managed_per_1k")]:
        ax.plot(rates, sweep_df[col].values,
                color=COLORS[mode], label=MODE_LABELS[mode],
                linewidth=2.5, zorder=3)

    # Annotate crossover points
    if crossovers:
        for pair_name, data in crossovers.items():
            xover = data.get("numerical_crossover_rps")
            cost = data.get("cost_at_crossover_per_1k")
            if xover and cost and 0.01 <= xover <= 100:
                ax.axvline(x=xover, color=COLORS["crossover"],
                          linestyle="--", alpha=0.7, linewidth=1.5, zorder=2)
                ax.scatter([xover], [cost], color=COLORS["crossover"],
                          s=100, zorder=5, marker="o", edgecolors="white",
                          linewidths=2)
                ax.annotate(
                    f"λ* = {xover:.2f} req/s\n({xover*60:.0f} req/min)",
                    xy=(xover, cost),
                    xytext=(xover * 2.5, cost * 1.5),
                    fontsize=9, fontweight="bold",
                    color=COLORS["crossover"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["crossover"],
                                   lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                             edgecolor=COLORS["crossover"], alpha=0.9),
                    zorder=6,
                )

    # Overlay measured data points
    if comparison_df is not None:
        for mode in ["serverless", "container", "managed"]:
            mode_data = comparison_df[comparison_df["deployment_mode"] == mode]
            if len(mode_data) > 0:
                ax.scatter(
                    mode_data["rate_rps"],
                    mode_data["measured_cost_per_1k"],
                    color=COLORS[mode], s=60, zorder=4,
                    marker="D", edgecolors="white", linewidths=1.5,
                    label=f"{MODE_LABELS[mode]} (measured)",
                )

    # Add shaded regions indicating cheapest mode
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Request Rate (requests/second)", fontweight="bold")
    ax.set_ylabel("Cost per 1000 Inferences (USD)", fontweight="bold")
    ax.set_title("ML Inference Cost Crossover Analysis",
                fontsize=16, fontweight="bold", pad=15)

    ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
    ax.grid(True, which="both", alpha=0.2)

    # Format tick labels
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))

    if save_path is None:
        save_path = os.path.join(FIGURE_DIR, "fig1_cost_crossover.png")
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")

    return fig


# ── Figure 2: Latency Distribution ──────────────────────────────────────────

def plot_latency_distribution(
    measured_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Box plots of latency at each request rate, grouped by deployment mode.

    Args:
        measured_df: DataFrame with columns: deployment_mode, rps_target,
                     latency_ms
    """
    ensure_figure_dir()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Latency Distribution by Request Rate",
                fontsize=16, fontweight="bold", y=1.02)

    for idx, mode in enumerate(["serverless", "container", "managed"]):
        ax = axes[idx]
        mode_data = measured_df[measured_df["deployment_mode"] == mode].copy()

        if len(mode_data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(MODE_LABELS[mode], color=COLORS[mode], fontweight="bold")
            continue

        # Create rate labels for x-axis
        mode_data["rate_label"] = mode_data["rps_target"].apply(
            lambda r: f"{r:.3f}" if r < 1 else f"{r:.0f}"
        )

        rates = sorted(mode_data["rps_target"].unique())
        bp_data = [
            mode_data[mode_data["rps_target"] == r]["latency_ms"].values
            for r in rates
        ]
        labels = [f"{r:.3f}" if r < 1 else f"{r:.0f}" for r in rates]

        bp = ax.boxplot(bp_data, tick_labels=labels, patch_artist=True,
                       showfliers=False, widths=0.6)

        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[mode])
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(2)

        # Add p50 and p99 annotations
        for i, data in enumerate(bp_data):
            if len(data) > 0:
                p50 = np.percentile(data, 50)
                p99 = np.percentile(data, 99)
                ax.annotate(f"p99:{p99:.0f}", xy=(i + 1, p99),
                          fontsize=7, ha="center", va="bottom",
                          color=COLORS[mode])

        ax.set_title(MODE_LABELS[mode], color=COLORS[mode], fontweight="bold")
        ax.set_xlabel("Request Rate (req/s)")
        if idx == 0:
            ax.set_ylabel("Latency (ms)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURE_DIR, "fig2_latency_distribution.png")
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")

    return fig


# ── Figure 3: Cold-Start Analysis ────────────────────────────────────────────

def plot_cold_start_analysis(
    measured_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Two-panel plot:
    - Left: Cold-start frequency (%) vs request rate
    - Right: Cold-start penalty (ms overhead) vs request rate
    """
    ensure_figure_dir()

    serverless = measured_df[measured_df["deployment_mode"] == "serverless"].copy()

    if len(serverless) == 0:
        print("  No serverless data for cold-start analysis")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cold-Start Analysis (Serverless Mode)",
                fontsize=16, fontweight="bold", y=1.02)

    rates = sorted(serverless["rps_target"].unique())

    # Panel 1: Cold-start frequency
    cold_pcts = []
    for rate in rates:
        rate_data = serverless[serverless["rps_target"] == rate]
        cold_pct = rate_data["is_cold_start"].sum() / len(rate_data) * 100
        cold_pcts.append(cold_pct)

    ax1.bar(range(len(rates)), cold_pcts, color=COLORS["serverless"],
           alpha=0.7, edgecolor=COLORS["serverless"])
    ax1.set_xticks(range(len(rates)))
    ax1.set_xticklabels([f"{r:.3f}" if r < 1 else f"{r:.0f}" for r in rates],
                       rotation=45, ha="right")
    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("Cold-Start Frequency (%)")
    ax1.set_title("Cold-Start Probability vs Request Rate")

    # Add value labels
    for i, pct in enumerate(cold_pcts):
        ax1.annotate(f"{pct:.1f}%", xy=(i, pct), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    # Panel 2: Latency penalty
    warm_latencies = []
    cold_latencies = []
    for rate in rates:
        rate_data = serverless[serverless["rps_target"] == rate]
        warm = rate_data[rate_data["is_cold_start"] == False]["latency_ms"]
        cold = rate_data[rate_data["is_cold_start"] == True]["latency_ms"]
        warm_latencies.append(warm.median() if len(warm) > 0 else 0)
        cold_latencies.append(cold.median() if len(cold) > 0 else 0)

    x = np.arange(len(rates))
    width = 0.35
    ax2.bar(x - width / 2, warm_latencies, width, label="Warm", color="#2A9D8F",
           alpha=0.7, edgecolor="#2A9D8F")
    ax2.bar(x + width / 2, cold_latencies, width, label="Cold", color="#E63946",
           alpha=0.7, edgecolor="#E63946")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r:.3f}" if r < 1 else f"{r:.0f}" for r in rates],
                       rotation=45, ha="right")
    ax2.set_xlabel("Request Rate (req/s)")
    ax2.set_ylabel("Median Latency (ms)")
    ax2.set_title("Cold-Start Penalty: Warm vs Cold Latency")
    ax2.legend()
    ax2.set_yscale("log")

    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURE_DIR, "fig3_cold_start_analysis.png")
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")

    return fig


# ── Figure 4: Sensitivity Analysis ──────────────────────────────────────────

def plot_sensitivity(
    sensitivity_data: Dict[str, pd.DataFrame],
    save_path: str = None,
) -> plt.Figure:
    """
    Multi-panel plot showing how the crossover point shifts with
    different parameters.
    """
    ensure_figure_dir()

    n_params = len(sensitivity_data)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    fig.suptitle("Sensitivity Analysis: How Crossover Shifts with Parameters",
                fontsize=16, fontweight="bold", y=1.05)

    param_labels = {
        "avg_duration_sec": ("Inference Duration (seconds)", "Duration (s)"),
        "memory_mb": ("Lambda Memory (MB)", "Memory (MB)"),
        "container_cost_per_hour": ("EC2 Hourly Cost ($)", "Cost ($/hr)"),
    }

    for idx, (param, df) in enumerate(sensitivity_data.items()):
        ax = axes[idx]
        title, xlabel = param_labels.get(param, (param, param))

        valid = df.dropna(subset=["crossover_rps"])
        if len(valid) > 0:
            ax.plot(valid[param], valid["crossover_rps"],
                   color=COLORS["crossover"], linewidth=2.5)
            ax.fill_between(valid[param], valid["crossover_rps"],
                          alpha=0.1, color=COLORS["crossover"])

        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel("Crossover Rate (req/s)", fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add reference line for default crossover
        if len(valid) > 0:
            mid_idx = len(valid) // 2
            mid_val = valid["crossover_rps"].iloc[mid_idx]
            ax.axhline(y=mid_val, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURE_DIR, "fig4_sensitivity_analysis.png")
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")

    return fig


# ── Figure 5: Model Validation ──────────────────────────────────────────────

def plot_model_validation(
    analysis_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Scatter plot of predicted vs measured costs with perfect-prediction line.
    """
    ensure_figure_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cost Model Validation: Predicted vs Measured",
                fontsize=16, fontweight="bold", y=1.02)

    # Panel 1: Scatter plot
    for mode in ["serverless", "container", "managed"]:
        mode_data = analysis_df[analysis_df["deployment_mode"] == mode]
        if len(mode_data) > 0:
            ax1.scatter(
                mode_data["measured_cost_per_1k"],
                mode_data["predicted_cost_per_1k"],
                c=COLORS[mode], s=80, label=MODE_LABELS[mode],
                edgecolors="white", linewidths=1.5, zorder=3,
            )

    # Perfect prediction line
    all_vals = np.concatenate([
        analysis_df["measured_cost_per_1k"].values,
        analysis_df["predicted_cost_per_1k"].values,
    ])
    min_val, max_val = all_vals.min() * 0.5, all_vals.max() * 2
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5,
            linewidth=1.5, label="Perfect prediction")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Measured Cost ($/1k inferences)", fontweight="bold")
    ax1.set_ylabel("Predicted Cost ($/1k inferences)", fontweight="bold")
    ax1.set_title("Predicted vs Measured")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # Panel 2: Error by rate
    for mode in ["serverless", "container", "managed"]:
        mode_data = analysis_df[analysis_df["deployment_mode"] == mode]
        if len(mode_data) > 0:
            ax2.plot(mode_data["rate_rps"], mode_data["error_pct"],
                    color=COLORS[mode], marker="o", linewidth=2,
                    markersize=8, label=MODE_LABELS[mode])

    ax2.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="10% error")
    ax2.set_xscale("log")
    ax2.set_xlabel("Request Rate (req/s)", fontweight="bold")
    ax2.set_ylabel("Prediction Error (%)", fontweight="bold")
    ax2.set_title("Prediction Error vs Request Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Add overall MAPE annotation
    overall_mape = analysis_df["error_pct"].mean()
    ax2.annotate(
        f"Overall MAPE: {overall_mape:.1f}%",
        xy=(0.95, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                 edgecolor="orange"),
    )

    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURE_DIR, "fig5_model_validation.png")
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")

    return fig


# ── Generate All Figures ─────────────────────────────────────────────────────

def generate_all_figures(results_dir: str = None):
    """
    Generate all 5 publication figures from available data.

    Args:
        results_dir: Directory containing analysis results CSVs.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")

    ensure_figure_dir()
    print("\n" + "=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from cost_model.crossover import CrossoverSolver

    solver = CrossoverSolver()
    crossovers = solver.find_all_crossovers()

    # Figure 1: Cost Crossover
    sweep_path = os.path.join(results_dir, "cost_sweep_smooth.csv")
    if os.path.exists(sweep_path):
        sweep_df = pd.read_csv(sweep_path)
        analysis_path = os.path.join(results_dir, "measured_analysis.csv")
        analysis_df = pd.read_csv(analysis_path) if os.path.exists(analysis_path) else None
        plot_cost_crossover(sweep_df, crossovers, analysis_df)
    else:
        print("  ⚠ No sweep data found. Run simulator first.")

    # Figure 2: Latency Distribution
    measured_path = os.path.join(results_dir, "raw", "combined_results.csv")
    if os.path.exists(measured_path):
        measured_df = pd.read_csv(measured_path)
        plot_latency_distribution(measured_df)
        # Figure 3: Cold-Start Analysis
        plot_cold_start_analysis(measured_df)
    else:
        print("  ⚠ No measured data found for latency/cold-start plots.")

    # Figure 4: Sensitivity Analysis
    sensitivity_files = {
        "avg_duration_sec": os.path.join(results_dir, "sensitivity_avg_duration_sec.csv"),
        "memory_mb": os.path.join(results_dir, "sensitivity_memory_mb.csv"),
        "container_cost_per_hour": os.path.join(results_dir, "sensitivity_container_cost_per_hour.csv"),
    }
    sensitivity_data = {}
    for param, path in sensitivity_files.items():
        if os.path.exists(path):
            sensitivity_data[param] = pd.read_csv(path)
    if sensitivity_data:
        plot_sensitivity(sensitivity_data)
    else:
        print("  ⚠ No sensitivity data found.")

    # Figure 5: Model Validation
    analysis_path = os.path.join(results_dir, "measured_analysis.csv")
    if os.path.exists(analysis_path):
        analysis_df = pd.read_csv(analysis_path)
        plot_model_validation(analysis_df)
    else:
        print("  ⚠ No validation data found.")

    print(f"\n  Figures saved to: {FIGURE_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
