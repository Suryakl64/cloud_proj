"""
Crossover point solver for ML inference cost analysis.

Solves for the request rate λ* where the cheapest deployment mode changes.
Provides both analytical closed-form solutions and numerical solvers,
plus sensitivity analysis showing how λ* shifts with pricing parameters.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, List, Tuple, Optional

from cost_model.pricing import PricingConfig, get_default_pricing
from cost_model.analytical_model import CostModel


class CrossoverSolver:
    """
    Finds crossover points where cost curves intersect.

    Key crossovers:
    1. Serverless ↔ Container: The primary crossover (most cited)
    2. Serverless ↔ Managed: Where managed becomes cheaper than serverless
    3. Container ↔ Managed: Where container beats managed (always, since cheaper hourly)
    """

    def __init__(self, pricing: PricingConfig = None):
        self.pricing = pricing or get_default_pricing()
        self.model = CostModel(self.pricing)

    # ── Analytical Closed-Form Solutions ──────────────────────────────────

    def analytical_crossover_serverless_container(self) -> float:
        """
        Closed-form crossover between serverless and container.

        At the crossover:
            C_serverless(λ*) = C_container(λ*)

        Without cold-start effects:
            1000 × (c_req + c_gb × mem × dur) = c_hour / (λ* × 3600) × 1000

        Solving for λ*:
            λ* = c_hour / ((c_req + c_gb × mem × dur) × 3600)

        Returns:
            Crossover rate in requests per second.
        """
        p_s = self.pricing.serverless
        p_c = self.pricing.container

        # Serverless cost per invocation (warm, no cold-start)
        compute_cost = p_s.cost_per_gb_second * p_s.memory_gb * p_s.avg_duration_sec
        per_invocation = p_s.cost_per_request + compute_cost

        # Crossover
        lambda_star = p_c.cost_per_hour / (per_invocation * 3600)

        return lambda_star

    def analytical_crossover_serverless_managed(self) -> float:
        """
        Closed-form crossover between serverless and managed endpoint.

        Same formula as above but using managed hourly rate.

        Returns:
            Crossover rate in requests per second.
        """
        p_s = self.pricing.serverless
        p_m = self.pricing.managed

        compute_cost = p_s.cost_per_gb_second * p_s.memory_gb * p_s.avg_duration_sec
        per_invocation = p_s.cost_per_request + compute_cost

        lambda_star = p_m.cost_per_hour / (per_invocation * 3600)

        return lambda_star

    # ── Numerical Solver (Accounts for Cold Starts) ──────────────────────

    def numerical_crossover(
        self,
        mode_a: str = "serverless",
        mode_b: str = "container",
        search_range: Tuple[float, float] = (0.001, 1000.0),
    ) -> Optional[float]:
        """
        Numerically find the crossover point between two modes.

        Accounts for cold-start probability that varies with request rate.

        Args:
            mode_a: First deployment mode.
            mode_b: Second deployment mode.
            search_range: (min_rps, max_rps) to search within.

        Returns:
            Crossover rate λ* in requests/second, or None if no crossover.
        """
        cost_fns = {
            "serverless": self.model.serverless_cost,
            "container": self.model.container_cost,
            "managed": self.model.managed_cost,
        }

        fn_a = cost_fns[mode_a]
        fn_b = cost_fns[mode_b]

        # Difference function: f(λ) = C_a(λ) - C_b(λ)
        def diff(lam):
            return fn_a(lam) - fn_b(lam)

        # Check if crossover exists in range
        lo, hi = search_range
        try:
            val_lo = diff(lo)
            val_hi = diff(hi)
        except (ValueError, ZeroDivisionError):
            return None

        if val_lo * val_hi > 0:
            # Same sign at both ends — no crossover in range
            return None

        try:
            lambda_star = brentq(diff, lo, hi, xtol=1e-6)
            return lambda_star
        except ValueError:
            return None

    # ── Find All Crossovers ──────────────────────────────────────────────

    def find_all_crossovers(self) -> Dict[str, dict]:
        """
        Find all pairwise crossover points.

        Returns:
            Dictionary mapping pair names to crossover details.
        """
        pairs = [
            ("serverless", "container"),
            ("serverless", "managed"),
            ("container", "managed"),
        ]

        results = {}
        for mode_a, mode_b in pairs:
            pair_name = f"{mode_a}_vs_{mode_b}"

            # Analytical (simplified, no cold-start effects)
            if mode_a == "serverless" and mode_b == "container":
                analytical = self.analytical_crossover_serverless_container()
            elif mode_a == "serverless" and mode_b == "managed":
                analytical = self.analytical_crossover_serverless_managed()
            else:
                analytical = None

            # Numerical (accounts for cold starts)
            numerical = self.numerical_crossover(mode_a, mode_b)

            # Cost at crossover
            if numerical:
                cost_at_crossover = self.model.serverless_cost(numerical)
                cheapest_below = mode_a if self.model.serverless_cost(0.01) < self.model.container_cost(0.01) else mode_b
                cheapest_above = mode_b if cheapest_below == mode_a else mode_a
            else:
                cost_at_crossover = None
                cheapest_below = None
                cheapest_above = None

            results[pair_name] = {
                "mode_a": mode_a,
                "mode_b": mode_b,
                "analytical_crossover_rps": analytical,
                "numerical_crossover_rps": numerical,
                "cost_at_crossover_per_1k": cost_at_crossover,
                "cheapest_below_crossover": cheapest_below,
                "cheapest_above_crossover": cheapest_above,
            }

        return results

    # ── Sensitivity Analysis ─────────────────────────────────────────────

    def sensitivity_analysis(
        self,
        parameter: str = "avg_duration_sec",
        values: np.ndarray = None,
        mode_pair: Tuple[str, str] = ("serverless", "container"),
    ) -> Dict[str, np.ndarray]:
        """
        Show how the crossover point shifts as a parameter changes.

        Args:
            parameter: Pricing parameter to vary. Options:
                - "avg_duration_sec": Inference duration
                - "memory_mb": Lambda memory
                - "container_cost_per_hour": EC2 hourly rate
                - "model_size_factor": Scales duration proportionally
            values: Array of values to test.
            mode_pair: Which pair of modes to find crossover for.

        Returns:
            Dictionary with parameter values and corresponding crossover rates.
        """
        if values is None:
            if parameter == "avg_duration_sec":
                values = np.linspace(0.01, 0.5, 50)
            elif parameter == "memory_mb":
                values = np.array([128, 256, 512, 1024, 2048, 3072])
            elif parameter == "container_cost_per_hour":
                values = np.linspace(0.005, 0.10, 50)
            elif parameter == "model_size_factor":
                values = np.linspace(0.1, 5.0, 50)
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

        crossovers = []
        import copy

        for val in values:
            # Clone pricing and modify parameter
            p = copy.deepcopy(self.pricing)

            if parameter == "avg_duration_sec":
                p.serverless.avg_duration_sec = val
                p.serverless.cold_start_duration_sec = val * 20  # Scale cold start
            elif parameter == "memory_mb":
                p.serverless.memory_mb = int(val)
                p.serverless.memory_gb = val / 1024.0
            elif parameter == "container_cost_per_hour":
                p.container.cost_per_hour = val
            elif parameter == "model_size_factor":
                base_duration = self.pricing.serverless.avg_duration_sec
                p.serverless.avg_duration_sec = base_duration * val
                p.serverless.cold_start_duration_sec = base_duration * val * 20

            solver = CrossoverSolver(p)
            xover = solver.numerical_crossover(mode_pair[0], mode_pair[1])
            crossovers.append(xover if xover is not None else np.nan)

        return {
            "parameter": parameter,
            "values": values,
            "crossover_rps": np.array(crossovers),
        }

    # ── Model Validation ─────────────────────────────────────────────────

    def validate_against_measurements(
        self,
        measured_data: Dict[str, Dict[float, float]],
    ) -> Dict[str, dict]:
        """
        Compare predicted costs against measured data points.

        Args:
            measured_data: Dict of mode -> {rate: measured_cost_per_1k}

        Returns:
            Validation metrics including MAPE per mode and overall.
        """
        results = {}

        for mode, measurements in measured_data.items():
            cost_fn = getattr(self.model, f"{mode}_cost")

            predicted = []
            actual = []
            rates = []

            for rate, measured_cost in measurements.items():
                pred = cost_fn(rate)
                predicted.append(pred)
                actual.append(measured_cost)
                rates.append(rate)

            predicted = np.array(predicted)
            actual = np.array(actual)

            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            # RMSE
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))

            results[mode] = {
                "rates": rates,
                "predicted": predicted.tolist(),
                "actual": actual.tolist(),
                "mape_percent": round(mape, 2),
                "rmse": round(rmse, 6),
                "max_error_percent": round(
                    np.max(np.abs((actual - predicted) / actual)) * 100, 2
                ),
            }

        # Overall MAPE
        all_pred = []
        all_actual = []
        for mode in results:
            all_pred.extend(results[mode]["predicted"])
            all_actual.extend(results[mode]["actual"])

        all_pred = np.array(all_pred)
        all_actual = np.array(all_actual)
        overall_mape = np.mean(np.abs((all_actual - all_pred) / all_actual)) * 100

        return {
            "per_mode": results,
            "overall_mape_percent": round(overall_mape, 2),
        }


# ── Convenience Functions ────────────────────────────────────────────────────

def print_crossover_analysis(pricing: PricingConfig = None):
    """Print a formatted crossover analysis to stdout."""
    solver = CrossoverSolver(pricing)
    crossovers = solver.find_all_crossovers()

    print("\n" + "=" * 70)
    print("  CROSSOVER ANALYSIS — ML Inference Cost Model")
    print("=" * 70)

    print(f"\n  Pricing Configuration:")
    print(f"    Serverless: ${solver.pricing.serverless.cost_per_request}/req "
          f"+ ${solver.pricing.serverless.cost_per_gb_second}/GB-s "
          f"× {solver.pricing.serverless.memory_gb:.1f} GB "
          f"× {solver.pricing.serverless.avg_duration_sec*1000:.0f}ms")
    print(f"    Container:  ${solver.pricing.container.cost_per_hour}/hr "
          f"({solver.pricing.container.instance_type})")
    print(f"    Managed:    ${solver.pricing.managed.cost_per_hour}/hr "
          f"({solver.pricing.managed.instance_type})")

    print(f"\n  Crossover Points:")
    for name, data in crossovers.items():
        analytical = data.get("analytical_crossover_rps")
        numerical = data.get("numerical_crossover_rps")

        print(f"\n    {data['mode_a']} ↔ {data['mode_b']}:")
        if analytical:
            print(f"      Analytical (no cold starts): {analytical:.3f} req/s "
                  f"({analytical * 60:.1f} req/min)")
        if numerical:
            print(f"      Numerical (with cold starts): {numerical:.3f} req/s "
                  f"({numerical * 60:.1f} req/min)")
            print(f"      Cost at crossover: ${data['cost_at_crossover_per_1k']:.4f} / 1k inferences")
        else:
            print(f"      No crossover found in search range")

    # Recommendation table
    print(f"\n  Recommendation by Request Rate:")
    print(f"    {'Rate':>12s}  {'Cheapest':>12s}  {'Cost/1k':>12s}")
    print(f"    {'─' * 12}  {'─' * 12}  {'─' * 12}")

    model = solver.model
    test_rates = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    for rate in test_rates:
        cheapest, cost = model.find_cheapest(rate)
        print(f"    {rate:>9.3f}/s  {cheapest:>12s}  ${cost:>10.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_crossover_analysis()
