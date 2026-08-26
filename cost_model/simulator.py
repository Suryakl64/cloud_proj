"""
Cost model sweep simulator.

Generates comprehensive cost comparison data across the full request rate
sweep. Can operate on measured data (from load tests) or use the analytical
model with simulated latency data. Outputs tables and data ready for
visualisation.
"""

import os
import sys
import json
import csv
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure cost_model package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cost_model.pricing import PricingConfig, get_default_pricing
from cost_model.analytical_model import CostModel
from cost_model.crossover import CrossoverSolver, print_crossover_analysis


class CostSimulator:
    """
    Simulates costs across the full request rate sweep for all modes.

    Can ingest measured latency data from the load test to produce
    measured costs, and compares against analytical predictions.
    """

    def __init__(self, pricing: PricingConfig = None):
        self.pricing = pricing or get_default_pricing()
        self.model = CostModel(self.pricing)
        self.solver = CrossoverSolver(self.pricing)

    def simulate_full_sweep(
        self,
        rates: np.ndarray = None,
        output_dir: str = None,
    ) -> pd.DataFrame:
        """
        Run the full cost simulation across all rates and modes.

        Args:
            rates: Request rates to evaluate (req/s).
            output_dir: Directory to save output files.

        Returns:
            DataFrame with cost comparison data.
        """
        if rates is None:
            rates = np.array([0.017, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0])

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "analysis", "results"
            )
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print("  COST SIMULATION — Full Sweep")
        print("=" * 70)

        # Build comparison table
        rows = []
        for rate in rates:
            s_cost = self.model.serverless_cost(rate)
            c_cost = self.model.container_cost(rate)
            m_cost = self.model.managed_cost(rate)

            s_monthly = self.model.serverless_monthly(rate)
            c_monthly = self.model.container_monthly(rate)
            m_monthly = self.model.managed_monthly(rate)

            cheapest, _ = self.model.find_cheapest(rate)

            rows.append({
                "rate_rps": rate,
                "rate_rpm": rate * 60,
                "rate_rph": rate * 3600,
                "serverless_per_1k": round(s_cost, 6),
                "container_per_1k": round(c_cost, 6),
                "managed_per_1k": round(m_cost, 6),
                "serverless_monthly": round(s_monthly, 2),
                "container_monthly": round(c_monthly, 2),
                "managed_monthly": round(m_monthly, 2),
                "cheapest_mode": cheapest,
                "savings_vs_2nd_pct": self._savings_percent(s_cost, c_cost, m_cost),
            })

        df = pd.DataFrame(rows)

        # Save to CSV
        csv_path = os.path.join(output_dir, "cost_comparison.csv")
        df.to_csv(csv_path, index=False)

        # Print summary table
        self._print_table(df)

        # Find and report crossovers
        crossovers = self.solver.find_all_crossovers()
        self._print_crossovers(crossovers)

        # Generate high-resolution sweep for smooth plotting
        smooth_rates = np.logspace(-2, 2, 500)
        sweep_data = self.model.evaluate_sweep(smooth_rates)

        # Save sweep data for plotting
        sweep_df = pd.DataFrame({
            "rate_rps": smooth_rates,
            "serverless_per_1k": sweep_data["per_1k"]["serverless"],
            "container_per_1k": sweep_data["per_1k"]["container"],
            "managed_per_1k": sweep_data["per_1k"]["managed"],
            "serverless_monthly": sweep_data["monthly"]["serverless"],
            "container_monthly": sweep_data["monthly"]["container"],
            "managed_monthly": sweep_data["monthly"]["managed"],
        })
        sweep_csv = os.path.join(output_dir, "cost_sweep_smooth.csv")
        sweep_df.to_csv(sweep_csv, index=False)

        print(f"\n  Output files:")
        print(f"    {csv_path}")
        print(f"    {sweep_csv}")

        return df

    def analyze_measured_data(
        self,
        data_path: str,
        output_dir: str = None,
    ) -> pd.DataFrame:
        """
        Analyze measured latency data from load tests and compute
        actual costs based on measured durations.

        Args:
            data_path: Path to combined_results.csv from sweep_runner.
            output_dir: Output directory for analysis results.

        Returns:
            DataFrame with measured vs predicted costs.
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "analysis", "results"
            )
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nAnalyzing measured data from: {data_path}")
        df = pd.read_csv(data_path)

        # Compute statistics per mode per rate
        analysis_rows = []

        for mode in df["deployment_mode"].unique():
            mode_df = df[df["deployment_mode"] == mode]

            for rate in sorted(mode_df["rps_target"].unique()):
                rate_df = mode_df[mode_df["rps_target"] == rate]

                latencies = rate_df["latency_ms"].values
                cold_starts = rate_df["is_cold_start"].sum()
                n_requests = len(rate_df)

                # Compute measured cost based on actual latencies
                if mode == "serverless":
                    p = self.pricing.serverless
                    avg_duration_sec = np.mean(latencies) / 1000.0
                    compute = p.cost_per_gb_second * p.memory_gb * avg_duration_sec
                    measured_per_1k = (p.cost_per_request + compute) * 1000
                elif mode == "container":
                    measured_per_1k = self.model.container_cost(rate)
                elif mode == "managed":
                    measured_per_1k = self.model.managed_cost(rate)
                else:
                    measured_per_1k = 0

                predicted_per_1k = getattr(self.model, f"{mode}_cost")(rate)

                analysis_rows.append({
                    "deployment_mode": mode,
                    "rate_rps": rate,
                    "n_requests": n_requests,
                    "latency_p50_ms": round(np.percentile(latencies, 50), 2),
                    "latency_p99_ms": round(np.percentile(latencies, 99), 2),
                    "latency_mean_ms": round(np.mean(latencies), 2),
                    "latency_std_ms": round(np.std(latencies), 2),
                    "cold_start_count": cold_starts,
                    "cold_start_pct": round(cold_starts / n_requests * 100, 1),
                    "measured_cost_per_1k": round(measured_per_1k, 6),
                    "predicted_cost_per_1k": round(predicted_per_1k, 6),
                    "error_pct": round(
                        abs(predicted_per_1k - measured_per_1k) / max(measured_per_1k, 1e-10) * 100,
                        2,
                    ),
                })

        analysis_df = pd.DataFrame(analysis_rows)

        # Save analysis
        analysis_path = os.path.join(output_dir, "measured_analysis.csv")
        analysis_df.to_csv(analysis_path, index=False)

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"  MEASURED vs PREDICTED COST ANALYSIS")
        print(f"{'=' * 80}")
        print(f"\n  {'Mode':>12s}  {'Rate':>8s}  {'P50':>8s}  {'P99':>8s}  "
              f"{'Meas$/1k':>10s}  {'Pred$/1k':>10s}  {'Error%':>7s}")
        print(f"  {'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 8}  "
              f"{'─' * 10}  {'─' * 10}  {'─' * 7}")

        for _, row in analysis_df.iterrows():
            print(
                f"  {row['deployment_mode']:>12s}  "
                f"{row['rate_rps']:>8.3f}  "
                f"{row['latency_p50_ms']:>7.1f}  "
                f"{row['latency_p99_ms']:>7.1f}  "
                f"${row['measured_cost_per_1k']:>9.4f}  "
                f"${row['predicted_cost_per_1k']:>9.4f}  "
                f"{row['error_pct']:>6.1f}%"
            )

        # Overall MAPE
        mape = analysis_df["error_pct"].mean()
        print(f"\n  Overall MAPE: {mape:.1f}%")
        print(f"  Output: {analysis_path}")

        return analysis_df

    def generate_sensitivity_report(
        self,
        output_dir: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """Generate sensitivity analysis data for all key parameters."""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "analysis", "results"
            )
        os.makedirs(output_dir, exist_ok=True)

        parameters = [
            ("avg_duration_sec", np.linspace(0.01, 0.5, 50)),
            ("memory_mb", np.array([128, 256, 512, 1024, 2048, 3072])),
            ("container_cost_per_hour", np.linspace(0.005, 0.10, 50)),
        ]

        results = {}
        print(f"\n{'=' * 60}")
        print(f"  SENSITIVITY ANALYSIS")
        print(f"{'=' * 60}")

        for param, values in parameters:
            sensitivity = self.solver.sensitivity_analysis(
                parameter=param, values=values
            )

            df = pd.DataFrame({
                param: sensitivity["values"],
                "crossover_rps": sensitivity["crossover_rps"],
            })

            csv_path = os.path.join(output_dir, f"sensitivity_{param}.csv")
            df.to_csv(csv_path, index=False)
            results[param] = df

            # Print summary
            valid = df.dropna(subset=["crossover_rps"])
            if len(valid) > 0:
                print(f"\n  {param}:")
                print(f"    Range: {values[0]:.4f} → {values[-1]:.4f}")
                print(f"    Crossover range: {valid['crossover_rps'].min():.3f} → "
                      f"{valid['crossover_rps'].max():.3f} req/s")

        return results

    # ── Private Helpers ──────────────────────────────────────────────────

    def _savings_percent(self, s: float, c: float, m: float) -> float:
        """Calculate savings of cheapest vs second cheapest."""
        costs = sorted([s, c, m])
        if costs[0] == 0:
            return 0.0
        return round((1 - costs[0] / costs[1]) * 100, 1)

    def _print_table(self, df: pd.DataFrame):
        """Print formatted cost comparison table."""
        print(f"\n  {'Rate':>10s}  {'Serverless':>12s}  {'Container':>12s}  "
              f"{'Managed':>12s}  {'Cheapest':>12s}  {'Savings':>8s}")
        print(f"  {'(req/s)':>10s}  {'($/1k inf)':>12s}  {'($/1k inf)':>12s}  "
              f"{'($/1k inf)':>12s}  {'':>12s}  {'':>8s}")
        print(f"  {'─' * 10}  {'─' * 12}  {'─' * 12}  "
              f"{'─' * 12}  {'─' * 12}  {'─' * 8}")

        for _, row in df.iterrows():
            print(
                f"  {row['rate_rps']:>10.3f}  "
                f"${row['serverless_per_1k']:>10.4f}  "
                f"${row['container_per_1k']:>10.4f}  "
                f"${row['managed_per_1k']:>10.4f}  "
                f"{row['cheapest_mode']:>12s}  "
                f"{row['savings_vs_2nd_pct']:>6.1f}%"
            )

        # Monthly costs
        print(f"\n  Monthly Cost Summary:")
        print(f"  {'Rate':>10s}  {'Serverless':>12s}  {'Container':>12s}  {'Managed':>12s}")
        print(f"  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
        for _, row in df.iterrows():
            print(
                f"  {row['rate_rps']:>10.3f}  "
                f"${row['serverless_monthly']:>10.2f}  "
                f"${row['container_monthly']:>10.2f}  "
                f"${row['managed_monthly']:>10.2f}"
            )

    def _print_crossovers(self, crossovers: Dict):
        """Print crossover analysis results."""
        print(f"\n  Crossover Points:")
        for name, data in crossovers.items():
            numerical = data.get("numerical_crossover_rps")
            if numerical:
                print(f"    {data['mode_a']} ↔ {data['mode_b']}: "
                      f"{numerical:.3f} req/s ({numerical * 60:.1f} req/min)")
            else:
                print(f"    {data['mode_a']} ↔ {data['mode_b']}: No crossover found")


def main():
    """Run the full simulation pipeline."""
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "analysis", "results"
    )

    simulator = CostSimulator()

    # 1. Run cost simulation
    print("\n" + "#" * 70)
    print("  STEP 1: Cost Model Simulation")
    print("#" * 70)
    df = simulator.simulate_full_sweep(output_dir=output_dir)

    # 2. Run sensitivity analysis
    print("\n" + "#" * 70)
    print("  STEP 2: Sensitivity Analysis")
    print("#" * 70)
    simulator.generate_sensitivity_report(output_dir=output_dir)

    # 3. Print crossover analysis
    print("\n" + "#" * 70)
    print("  STEP 3: Crossover Analysis")
    print("#" * 70)
    print_crossover_analysis()

    # 4. Check for measured data
    measured_path = os.path.join(output_dir, "raw", "combined_results.csv")
    if os.path.exists(measured_path):
        print("\n" + "#" * 70)
        print("  STEP 4: Measured Data Analysis")
        print("#" * 70)
        simulator.analyze_measured_data(measured_path, output_dir)
    else:
        print(f"\n  [i] No measured data found at {measured_path}")
        print(f"    Run 'python loadtest/sweep_runner.py simulate' to generate synthetic data")
        print(f"    Then re-run this simulator to analyze measured vs predicted costs.")

    print(f"\n{'=' * 70}")
    print(f"  [OK] Simulation complete. Results in: {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
