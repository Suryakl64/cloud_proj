"""
Deploys MobileNetV2 to AWS SageMaker Serverless Inference in Sydney (ap-southeast-2).
Features:
- Automated IAM role setup with SageMaker & S3 policies
- Registers SageMaker Model with AWS Deep Learning Container
- Creates Serverless Endpoint Config (2048 MB, max concurrency 10, $0 idle cost)
- Launches and waits for endpoint to enter 'InService' state
- Executes verified live test invocation
"""

import json
import os
import subprocess
import sys
import time

REGION = "ap-southeast-2"
ACCOUNT_ID = "257247531891"
BUCKET_NAME = f"cloud-proj-deploy-{ACCOUNT_ID}-sydney"
S3_MODEL_KEY = "sagemaker/model.tar.gz"
S3_MODEL_URI = f"s3://{BUCKET_NAME}/{S3_MODEL_KEY}"

ROLE_NAME = "SageMaker-ExecutionRole-MLInference"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"

MODEL_NAME = "mobilenetv2-sagemaker-model"
CONFIG_NAME = "mobilenetv2-serverless-config"
ENDPOINT_NAME = "sagemaker-mobilenet-serverless"

# AWS Official Deep Learning Container for PyTorch CPU Inference in Sydney
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.0.0-cpu-py310"


def run_aws(cmd_list, check=True):
    """Executes an AWS CLI command and returns parsed JSON or stdout text."""
    cmd = ["aws"] + cmd_list + ["--region", REGION]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR ({result.returncode}):\n{result.stderr}")
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def ensure_iam_role():
    """Creates SageMaker execution role if it does not already exist."""
    print("\n--- 1. Checking IAM Execution Role ---")
    res = subprocess.run(["aws", "iam", "get-role", "--role-name", ROLE_NAME], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[OK] IAM Role {ROLE_NAME} exists.")
        return ROLE_ARN

    print(f"Creating IAM role: {ROLE_NAME}...")
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    subprocess.run([
        "aws", "iam", "create-role",
        "--role-name", ROLE_NAME,
        "--assume-role-policy-document", json.dumps(trust_policy)
    ], check=True)

    print("Attaching policies...")
    subprocess.run([
        "aws", "iam", "attach-role-policy",
        "--role-name", ROLE_NAME,
        "--policy-arn", "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    ], check=True)
    subprocess.run([
        "aws", "iam", "attach-role-policy",
        "--role-name", ROLE_NAME,
        "--policy-arn", "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    ], check=True)

    print("Waiting 10s for IAM propagation...")
    time.sleep(10)
    print(f"[OK] IAM Role created: {ROLE_ARN}")
    return ROLE_ARN


def create_model(role_arn):
    """Registers the model in SageMaker."""
    print("\n--- 2. Registering SageMaker Model ---")
    # Delete old model if it exists
    res = subprocess.run(["aws", "sagemaker", "describe-model", "--model-name", MODEL_NAME, "--region", REGION], capture_output=True)
    if res.returncode == 0:
        print(f"Removing previous model {MODEL_NAME}...")
        run_aws(["sagemaker", "delete-model", "--model-name", MODEL_NAME])

    primary_container = {
        "Image": IMAGE_URI,
        "ModelDataUrl": S3_MODEL_URI,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": S3_MODEL_URI
        }
    }

    run_aws([
        "sagemaker", "create-model",
        "--model-name", MODEL_NAME,
        "--primary-container", json.dumps(primary_container),
        "--execution-role-arn", role_arn
    ])
    print(f"[OK] SageMaker Model registered: {MODEL_NAME}")


def create_endpoint_config():
    """Creates the Serverless Endpoint Configuration."""
    print("\n--- 3. Creating Serverless Endpoint Config ---")
    res = subprocess.run(["aws", "sagemaker", "describe-endpoint-config", "--endpoint-config-name", CONFIG_NAME, "--region", REGION], capture_output=True)
    if res.returncode == 0:
        print(f"Removing previous endpoint config {CONFIG_NAME}...")
        run_aws(["sagemaker", "delete-endpoint-config", "--endpoint-config-name", CONFIG_NAME])

    serverless_config = {
        "MemorySizeInMB": 2048,
        "MaxConcurrency": 10
    }
    prod_variants = [
        {
            "VariantName": "AllTraffic",
            "ModelName": MODEL_NAME,
            "ServerlessConfig": serverless_config
        }
    ]

    run_aws([
        "sagemaker", "create-endpoint-config",
        "--endpoint-config-name", CONFIG_NAME,
        "--production-variants", json.dumps(prod_variants)
    ])
    print(f"[OK] Serverless Endpoint Config created: {CONFIG_NAME} (2048 MB, MaxConcurrency=10)")


def deploy_endpoint():
    """Deploys or updates the SageMaker Endpoint."""
    print("\n--- 4. Deploying SageMaker Endpoint ---")
    res = subprocess.run(["aws", "sagemaker", "describe-endpoint", "--endpoint-name", ENDPOINT_NAME, "--region", REGION], capture_output=True, text=True)
    
    if res.returncode == 0:
        print(f"Endpoint {ENDPOINT_NAME} already exists. Updating to new config...")
        run_aws([
            "sagemaker", "update-endpoint",
            "--endpoint-name", ENDPOINT_NAME,
            "--endpoint-config-name", CONFIG_NAME
        ])
    else:
        print(f"Creating new SageMaker endpoint: {ENDPOINT_NAME}...")
        run_aws([
            "sagemaker", "create-endpoint",
            "--endpoint-name", ENDPOINT_NAME,
            "--endpoint-config-name", CONFIG_NAME
        ])

    print("Waiting for endpoint to enter 'InService' status (this usually takes 2-3 minutes for serverless)...")
    while True:
        status_json = run_aws(["sagemaker", "describe-endpoint", "--endpoint-name", ENDPOINT_NAME])
        status = json.loads(status_json).get("EndpointStatus", "Unknown")
        print(f"  Current Status: {status}")
        if status == "InService":
            print(f"[OK] SageMaker Serverless Endpoint is LIVE and InService!")
            break
        elif status in ["Failed", "Deleting"]:
            raise RuntimeError(f"Endpoint deployment ended with status: {status}")
        time.sleep(15)


def test_endpoint():
    """Sends a test inference request to the live SageMaker endpoint."""
    print("\n--- 5. Testing Live Endpoint Inference ---")
    test_payload = json.dumps({"random": True})
    
    with open("test_payload.json", "w") as f:
        f.write(test_payload)

    t0 = time.perf_counter()
    res = run_aws([
        "sagemaker-runtime", "invoke-endpoint",
        "--endpoint-name", ENDPOINT_NAME,
        "--content-type", "application/json",
        "--body", "fileb://test_payload.json",
        "test_output.json"
    ])
    rtt_ms = (time.perf_counter() - t0) * 1000

    if os.path.exists("test_output.json"):
        with open("test_output.json") as f:
            resp_body = f.read()
        print(f"[SUCCESS] SageMaker Response (Round-Trip: {rtt_ms:.2f} ms):\n{resp_body}")
        try:
            os.remove("test_payload.json")
            os.remove("test_output.json")
        except Exception:
            pass


def main():
    print("=======================================================")
    print("  AWS SageMaker Serverless Deployment: MobileNetV2     ")
    print(f"  Region: {REGION} | Memory: 2048 MB | Max Concurrency: 10")
    print("=======================================================")
    role_arn = ensure_iam_role()
    create_model(role_arn)
    create_endpoint_config()
    deploy_endpoint()
    test_endpoint()
    print("\n[ALL COMPLETE] SageMaker Serverless testbed is ready for benchmarking!")


if __name__ == "__main__":
    main()
