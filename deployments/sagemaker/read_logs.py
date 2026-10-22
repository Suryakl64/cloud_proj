import subprocess
import os

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

cmd = [
    "aws", "logs", "get-log-events",
    "--log-group-name", "/aws/sagemaker/Endpoints/sagemaker-mobilenet-serverless",
    "--log-stream-name", "AllTraffic/7922ea00b130fc05bb000bb27c63a1f4-124f8bd92cdf47a0b589c634cf1748e7",
    "--region", "ap-southeast-2"
]
res = subprocess.run(cmd, capture_output=True, env=env)
with open("sagemaker_events.txt", "wb") as f:
    f.write(res.stdout)
    f.write(b"\n--- STDERR ---\n")
    f.write(res.stderr)

print("Saved raw bytes log output to sagemaker_events.txt")
