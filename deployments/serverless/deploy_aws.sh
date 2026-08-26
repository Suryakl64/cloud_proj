#!/bin/bash
set -e

# AWS Lambda Container Deployment Script
# Assumes you have run `aws configure` and have docker installed.

export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="us-east-1"
export ECR_REPO="ml-inference-lambda"
export IMAGE_TAG="latest"
export FUNCTION_NAME="ml-inference-serverless"

echo "=========================================================="
echo "  Deploying ML model to AWS Lambda (us-east-1)"
echo "=========================================================="

echo "1. Authenticating Docker to AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "2. Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION || \
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

echo "3. Building Docker image for Lambda..."
# Note: Build context is the project root to include model/ and inference/
cd ../..
docker build -t $ECR_REPO:$IMAGE_TAG -f deployments/serverless/Dockerfile .

echo "4. Tagging and pushing image to ECR..."
docker tag $ECR_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

echo "5. Updating/Creating Lambda function..."
# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME --region $AWS_REGION > /dev/null 2>&1; then
    echo "Updating existing function code..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG \
        --region $AWS_REGION
else
    echo "Creating new function..."
    # Need an execution role (create a basic basic role if it doesn't exist, this is a placeholder)
    ROLE_ARN=$(aws iam get-role --role-name lambda-ex --query Role.Arn --output text 2>/dev/null || echo "")
    
    if [ -z "$ROLE_ARN" ]; then
        echo "Creating basic execution role..."
        aws iam create-role --role-name lambda-ex --assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
        aws iam attach-role-policy --role-name lambda-ex --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        ROLE_ARN=$(aws iam get-role --role-name lambda-ex --query Role.Arn --output text)
        sleep 10 # Wait for role to propagate
    fi

    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG \
        --role $ROLE_ARN \
        --timeout 30 \
        --memory-size 1024 \
        --region $AWS_REGION
fi

echo "=========================================================="
echo "  Deploy Complete!"
echo "  To test, go to AWS Console -> Lambda -> ml-inference-serverless"
echo "  and click 'Test'."
echo "=========================================================="
