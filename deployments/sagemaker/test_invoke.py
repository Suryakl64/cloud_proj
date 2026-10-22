import subprocess
import json
import time
import os

with open("test_payload.json", "w") as f:
    f.write(json.dumps({"random": True}))

cmd = [
    "aws", "sagemaker-runtime", "invoke-endpoint",
    "--endpoint-name", "sagemaker-mobilenet-serverless",
    "--content-type", "application/json",
    "--body", "fileb://test_payload.json",
    "--region", "ap-southeast-2",
    "output.json"
]

t0 = time.perf_counter()
res = subprocess.run(cmd, capture_output=True, text=True)
latency_ms = (time.perf_counter() - t0) * 1000

print(f"Return code: {res.returncode}")
print(f"Stdout:\n{res.stdout}")
print(f"Stderr:\n{res.stderr}")

if os.path.exists("output.json"):
    with open("output.json") as f:
        print(f"Output JSON ({latency_ms:.2f} ms):\n{f.read()}")
