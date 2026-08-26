# Container image for always-on ML inference server
# Runs FastAPI + Uvicorn with ONNX Runtime

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    "onnxruntime>=1.17.0" \
    "numpy<2.0" \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    python-multipart==0.0.6

# Copy shared inference module
COPY inference/ /app/inference/

# Copy model (including external weights .onnx.data)
COPY model/ /app/model/

# Copy server code
COPY deployments/container/server.py /app/server.py

# Set environment
ENV ONNX_MODEL_PATH=/app/model/model.onnx
ENV PORT=8080

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run with single worker (simulating t3.small constraints)
CMD ["python", "server.py"]
