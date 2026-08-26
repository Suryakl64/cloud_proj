"""
Parameterised analytical cost model for ML inference deployment modes.

Provides closed-form cost functions for:
  - Serverless (AWS Lambda): per-invocation + per-GB-second
  - Container (EC2): fixed hourly rate amortised over requests
  - Managed (SageMaker): fixed hourly rate with managed premium

All functions express cost as $/1000 inferences as a function of
request rate λ (requests per second).
"""

import numpy as np
from typing import Tuple

from cost_model.pricing import PricingConfig, get_default_pricing


class CostModel:
    """
    Analytical cost model for comparing ML inference deployment modes.

    The core insight: serverless cost scales linearly with λ (pay per use),
    while container/managed costs are fixed and amortised (cost per inference
    decreases as λ increases). This creates a crossover point.
    """

    def __init__(self, pricing: PricingConfig = None):
        self.pricing = pricing or get_default_pricing()

    # ── Per-1000-Inference Cost Functions ─────────────────────────────────

    def serverless_cost(self, lam: float) -> float:
        """
        Cost per 1000 inferences for serverless (Lambda).

        C_serverless = 1000 × (c_request + c_gb_sec × mem_gb × duration)

        Note: This is constant w.r.t. λ — each invocation costs the same
        regardless of how many there are (above free tier).

        Args:
            lam: Request rate (requests/second). Used only for cold-start
                 probability adjustment.

        Returns:
            Cost in USD per 1000 inferences.
        """
        p = self.pricing.serverless

        # Cold-start probability decreases with request rate
        # Model: P(cold) = base_prob × exp(-λ / decay_rate)
        cold_prob = min(0.5, max(0.01, 0.20 * np.exp(-lam / 2.0)))

        # Effective average duration (blending warm and cold)
        effective_duration = (
            (1 - cold_prob) * p.avg_duration_sec +
            cold_prob * p.cold_start_duration_sec
        )

        # Cost per invocation
        compute_cost = p.cost_per_gb_second * p.memory_gb * effective_duration
        per_invocation_cost = p.cost_per_request + compute_cost

        return per_invocation_cost * 1000

    def container_cost(self, lam: float) -> float:
        """
        Cost per 1000 inferences for always-on container (EC2).

        C_container = (c_hour / (λ × 3600)) × 1000

        The container runs 24/7 regardless of load. Cost per inference
        is the hourly cost spread across all requests in that hour.

        Args:
            lam: Request rate (requests/second). Must be > 0.

        Returns:
            Cost in USD per 1000 inferences.
        """
        if lam <= 0:
            return float("inf")

        p = self.pricing.container
        requests_per_hour = lam * 3600
        cost_per_request = p.cost_per_hour / requests_per_hour

        return cost_per_request * 1000

    def managed_cost(self, lam: float) -> float:
        """
        Cost per 1000 inferences for managed endpoint (SageMaker).

        C_managed = (c_sm_hour / (λ × 3600)) × 1000

        Same structure as container but with a higher hourly rate
        reflecting the managed service premium.

        Args:
            lam: Request rate (requests/second). Must be > 0.

        Returns:
            Cost in USD per 1000 inferences.
        """
        if lam <= 0:
            return float("inf")

        p = self.pricing.managed
        requests_per_hour = lam * 3600
        cost_per_request = p.cost_per_hour / requests_per_hour

        return cost_per_request * 1000

    # ── Monthly Cost Functions ───────────────────────────────────────────

    def serverless_monthly(self, lam: float) -> float:
        """Total monthly cost for serverless at rate λ."""
        p = self.pricing.serverless
        hours = self.pricing.hours_per_month
        total_requests = lam * 3600 * hours

        # Apply free tier
        billable_requests = max(0, total_requests - p.free_requests)

        cold_prob = min(0.5, max(0.01, 0.20 * np.exp(-lam / 2.0)))
        effective_duration = (
            (1 - cold_prob) * p.avg_duration_sec +
            cold_prob * p.cold_start_duration_sec
        )

        request_cost = billable_requests * p.cost_per_request
        compute_gb_sec = total_requests * p.memory_gb * effective_duration
        billable_gb_sec = max(0, compute_gb_sec - p.free_gb_seconds)
        compute_cost = billable_gb_sec * p.cost_per_gb_second

        return request_cost + compute_cost

    def container_monthly(self, lam: float) -> float:
        """Total monthly cost for container (always-on)."""
        p = self.pricing.container
        return p.cost_per_hour * self.pricing.hours_per_month

    def managed_monthly(self, lam: float) -> float:
        """Total monthly cost for managed endpoint (always-on)."""
        p = self.pricing.managed
        return p.cost_per_hour * self.pricing.hours_per_month

    # ── Evaluation Across Sweep ──────────────────────────────────────────

    def evaluate_sweep(self, rates: np.ndarray = None) -> dict:
        """
        Evaluate all cost functions across a sweep of request rates.

        Args:
            rates: Array of request rates (req/s). Defaults to logarithmic
                   sweep from 0.01 to 100.

        Returns:
            Dictionary with rates and cost arrays for each mode.
        """
        if rates is None:
            rates = np.logspace(-2, 2, 200)  # 0.01 to 100 req/s

        serverless = np.array([self.serverless_cost(r) for r in rates])
        container = np.array([self.container_cost(r) for r in rates])
        managed = np.array([self.managed_cost(r) for r in rates])

        # Monthly costs
        serverless_mo = np.array([self.serverless_monthly(r) for r in rates])
        container_mo = np.array([self.container_monthly(r) for r in rates])
        managed_mo = np.array([self.managed_monthly(r) for r in rates])

        return {
            "rates": rates,
            "per_1k": {
                "serverless": serverless,
                "container": container,
                "managed": managed,
            },
            "monthly": {
                "serverless": serverless_mo,
                "container": container_mo,
                "managed": managed_mo,
            },
        }

    def find_cheapest(self, lam: float) -> Tuple[str, float]:
        """
        Determine the cheapest deployment mode at rate λ.

        Returns:
            Tuple of (mode_name, cost_per_1k).
        """
        costs = {
            "serverless": self.serverless_cost(lam),
            "container": self.container_cost(lam),
            "managed": self.managed_cost(lam),
        }
        cheapest = min(costs, key=costs.get)
        return cheapest, costs[cheapest]
