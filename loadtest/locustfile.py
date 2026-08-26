"""
Locust load test definition for ML inference endpoints.

Defines a single HttpUser that sends inference requests at a controlled
throughput rate. Supports all three deployment modes via configuration.
Records per-request latency and cold-start metadata.
"""

import os
import json
import time

from locust import HttpUser, task, constant_throughput, events


class MLInferenceUser(HttpUser):
    """
    Simulates a client sending ML inference requests.

    Uses constant_throughput to ensure precise RPS control.
    The throughput rate is set via environment variable LOCUST_RPS_PER_USER.
    """

    # Default: 1 request per second per user
    # Override via environment variable for sweep control
    wait_time = constant_throughput(
        float(os.environ.get("LOCUST_RPS_PER_USER", "1"))
    )

    def on_start(self):
        """Called when a new user starts. Initialise per-user state."""
        self.deployment_mode = os.environ.get("DEPLOYMENT_MODE", "container")
        self.request_payload = json.dumps({"random": True})
        self.headers = {"Content-Type": "application/json"}
        self._user_request_count = 0

    @task
    def predict(self):
        """Send an inference request and record metadata."""
        self._user_request_count += 1

        with self.client.post(
            "/predict",
            data=self.request_payload,
            headers=self.headers,
            catch_response=True,
            name=f"predict [{self.deployment_mode}]",
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()

                    # Tag the response with cold-start info for analysis
                    is_cold_start = result.get("is_cold_start", False)
                    inference_time = result.get("inference_time_ms", 0)

                    # Mark as success with metadata
                    response.success()

                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")


class LambdaInferenceUser(HttpUser):
    """
    Specialised user for Lambda Function URL endpoints.

    Lambda endpoints return the response wrapped in a different
    JSON structure (body is a stringified JSON).
    """

    wait_time = constant_throughput(
        float(os.environ.get("LOCUST_RPS_PER_USER", "1"))
    )

    def on_start(self):
        self.request_payload = json.dumps({"random": True})
        self.headers = {"Content-Type": "application/json"}

    @task
    def predict(self):
        with self.client.post(
            "/",  # Lambda function URL uses root path
            data=self.request_payload,
            headers=self.headers,
            catch_response=True,
            name="predict [serverless]",
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    # Lambda wraps response in statusCode/body
                    if "body" in result:
                        body = json.loads(result["body"])
                    else:
                        body = result
                    response.success()
                except (json.JSONDecodeError, KeyError):
                    response.failure("Invalid Lambda response")
            else:
                response.failure(f"HTTP {response.status_code}")


# ── Custom event hooks for CSV logging ───────────────────────────────────────

_csv_file = None


@events.init.add_listener
def on_init(environment, **kwargs):
    """Open CSV file for raw result logging."""
    global _csv_file
    output_dir = os.environ.get("RESULTS_DIR", "analysis/results/raw")
    os.makedirs(output_dir, exist_ok=True)

    mode = os.environ.get("DEPLOYMENT_MODE", "unknown")
    rps = os.environ.get("TARGET_RPS", "0")
    filename = f"{mode}_rps{rps}_{int(time.time())}.csv"

    filepath = os.path.join(output_dir, filename)
    _csv_file = open(filepath, "w")
    _csv_file.write(
        "timestamp,deployment_mode,rps_target,rps_actual,latency_ms,"
        "is_cold_start,status_code,error\n"
    )


@events.request.add_listener
def on_request(request_type, name, response_time, response_length,
               response, exception, context, **kwargs):
    """Log each request to CSV for post-hoc analysis."""
    if _csv_file is None:
        return

    mode = os.environ.get("DEPLOYMENT_MODE", "unknown")
    target_rps = os.environ.get("TARGET_RPS", "0")

    # Try to extract cold-start from response
    is_cold_start = False
    if response is not None and hasattr(response, "text"):
        try:
            data = json.loads(response.text)
            if "body" in data:
                data = json.loads(data["body"])
            is_cold_start = data.get("is_cold_start", False)
        except (json.JSONDecodeError, TypeError):
            pass

    status_code = response.status_code if response is not None else 0
    error = str(exception) if exception else ""

    _csv_file.write(
        f"{time.time()},{mode},{target_rps},{0},{response_time},"
        f"{is_cold_start},{status_code},{error}\n"
    )
    _csv_file.flush()


@events.quitting.add_listener
def on_quit(environment, **kwargs):
    """Close CSV file on shutdown."""
    if _csv_file:
        _csv_file.close()
