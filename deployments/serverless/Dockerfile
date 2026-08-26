# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Install only the necessary dependencies for inference (avoids huge PyTorch/Jupyter packages)
RUN pip install "numpy<2.0" onnxruntime --target "${LAMBDA_TASK_ROOT}"

# Copy the model file
COPY model/model.onnx ${LAMBDA_TASK_ROOT}/model/model.onnx

# Copy the inference module
COPY inference/ ${LAMBDA_TASK_ROOT}/inference/

# Copy the handler
COPY deployments/serverless/handler.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "handler.lambda_handler" ]
