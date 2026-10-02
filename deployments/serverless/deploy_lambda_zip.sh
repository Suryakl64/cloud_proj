#!/bin/bash
set -e
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export FUNCTION_NAME="ml-inference-serverless"
echo "Building Lambda Zip Package..."
rm -rf lambda_pkg function.zip
mkdir lambda_pkg
pip install --platform manylinux2014_x86_64 --target ./lambda_pkg --implementation cp --python-version 3.11 --only-binary=:all: "numpy<2.0" onnxruntime
cp model/model.onnx lambda_pkg/
cp -r inference lambda_pkg/
cp deployments/serverless/handler.py lambda_pkg/
cd lambda_pkg
zip -r ../function.zip .
cd ..
echo "Creating/Updating AWS Lambda Function..."
ROLE_ARN=$(aws iam get-role --role-name lambda-ex --query Role.Arn --output text 2>/dev/null || echo "")
if [ -z "$ROLE_ARN" ]; then
    echo "Creating basic execution role..."
    aws iam create-role --role-name lambda-ex --assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
    aws iam attach-role-policy --role-name lambda-ex --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    ROLE_ARN=$(aws iam get-role --role-name lambda-ex --query Role.Arn --output text)
    sleep 10
fi
aws lambda get-function --function-name $FUNCTION_NAME --region us-east-1 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    aws lambda update-function-code --function-name $FUNCTION_NAME --zip-file fileb://function.zip --region us-east-1
else
    aws lambda create-function --function-name $FUNCTION_NAME --runtime python3.11 --handler handler.handler --role $ROLE_ARN --zip-file fileb://function.zip --timeout 30 --memory-size 1024 --region us-east-1
fi
echo "Deploy Complete!"
