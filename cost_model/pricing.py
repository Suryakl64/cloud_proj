"""
Cloud pricing constants for ML inference deployment modes.

All values are parameterised and can be overridden by the user
to match their specific region, instance type, or pricing tier.
Prices are in USD — defaults set to Azure pricing.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ServerlessPricing:
    """AWS Lambda pricing parameters."""

    # Per-request charge (AWS Lambda)
    cost_per_request: float = 0.0000002  # $0.20 per 1M executions

    # Compute charge (per GB-second)
    cost_per_gb_second: float = 0.0000166667  # AWS Lambda

    # Function configuration
    memory_mb: int = 1024
    memory_gb: float = field(init=False)

    # Measured/estimated inference duration (seconds)
    avg_duration_sec: float = 0.050  # 50ms warm inference
    cold_start_duration_sec: float = 1.2  # ~1200ms cold start
    cold_start_probability: float = 0.05  # Depends on request rate

    # Free tier (per month)
    free_requests: int = 1_000_000
    free_gb_seconds: float = 400_000

    def __post_init__(self):
        self.memory_gb = self.memory_mb / 1024.0


@dataclass
class ContainerPricing:
    """AWS EC2 container (always-on) pricing parameters."""

    # Hourly cost for the VM (AWS t3.small)
    cost_per_hour: float = 0.0208  # t3.small on-demand, us-east-1

    # Instance details (for reference)
    instance_type: str = "t3.small"
    vcpus: int = 2
    memory_gb: float = 2.0

    # Optional: Managed disk cost (per GB-month, EBS gp3)
    disk_cost_per_gb_month: float = 0.08
    disk_size_gb: int = 30


@dataclass
class ManagedPricing:
    """AWS SageMaker Managed endpoint pricing parameters."""

    # Hourly cost for the managed instance
    # ml.t2.medium: smallest managed option
    cost_per_hour: float = 0.065  # ml.t2.medium

    # Instance details
    instance_type: str = "ml.t2.medium"
    vcpus: int = 2
    memory_gb: float = 4.0

    # Managed premium (ratio over equivalent raw VM)
    managed_premium_factor: float = 1.3


@dataclass
class PricingConfig:
    """Complete pricing configuration for all three modes."""

    serverless: ServerlessPricing = field(default_factory=ServerlessPricing)
    container: ContainerPricing = field(default_factory=ContainerPricing)
    managed: ManagedPricing = field(default_factory=ManagedPricing)

    # Model characteristics
    model_name: str = "MobileNetV2"
    model_size_mb: float = 14.0
    avg_inference_ms: float = 50.0  # Warm inference time

    # Analysis period
    hours_per_month: float = 730.0  # ~365.25 * 24 / 12

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialisation."""
        return {
            "serverless": {
                "cost_per_request": self.serverless.cost_per_request,
                "cost_per_gb_second": self.serverless.cost_per_gb_second,
                "memory_mb": self.serverless.memory_mb,
                "avg_duration_sec": self.serverless.avg_duration_sec,
                "cold_start_duration_sec": self.serverless.cold_start_duration_sec,
            },
            "container": {
                "cost_per_hour": self.container.cost_per_hour,
                "instance_type": self.container.instance_type,
            },
            "managed": {
                "cost_per_hour": self.managed.cost_per_hour,
                "instance_type": self.managed.instance_type,
            },
            "model": {
                "name": self.model_name,
                "size_mb": self.model_size_mb,
                "avg_inference_ms": self.avg_inference_ms,
            },
        }


# ── Preset Configurations ───────────────────────────────────────────────────

def get_default_pricing() -> PricingConfig:
    """Default pricing with current Azure rates (East US region)."""
    return PricingConfig()


def get_aws_pricing() -> PricingConfig:
    """AWS equivalent pricing (approximate)."""
    return PricingConfig(
        serverless=ServerlessPricing(
            cost_per_request=0.0000002,  # Lambda
            cost_per_gb_second=0.0000166667,
            memory_mb=1024,
            avg_duration_sec=0.050,
        ),
        container=ContainerPricing(
            cost_per_hour=0.0208,  # t3.small
            instance_type="t3.small",
            vcpus=2,
            memory_gb=2.0,
        ),
        managed=ManagedPricing(
            cost_per_hour=0.065,  # ml.t2.medium
            instance_type="ml.t2.medium",
        ),
    )


def get_gcp_pricing() -> PricingConfig:
    """Google Cloud equivalent pricing (approximate)."""
    return PricingConfig(
        serverless=ServerlessPricing(
            cost_per_request=0.0000004,  # Cloud Functions
            cost_per_gb_second=0.0000025,
            memory_mb=1024,
            avg_duration_sec=0.050,
        ),
        container=ContainerPricing(
            cost_per_hour=0.0192,  # e2-small
            instance_type="e2-small",
            vcpus=2,
            memory_gb=2.0,
        ),
        managed=ManagedPricing(
            cost_per_hour=0.0759,  # n1-standard-1 (Vertex AI)
            instance_type="n1-standard-1",
        ),
    )
