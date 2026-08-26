$ErrorActionPreference = "Continue"

$AWS_ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$AWS_REGION = "us-east-1"
$ECR_REPO = "ml-inference-lambda"
$IMAGE_TAG = "latest"
$FUNCTION_NAME = "ml-inference-serverless"

Write-Host "=========================================================="
Write-Host "  Deploying ML model to AWS Lambda (us-east-1)"
Write-Host "=========================================================="

Write-Host "1. Authenticating Docker to AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

Write-Host "2. Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Repository doesn't exist, creating..."
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION | Out-Null
}

Write-Host "3. Building Docker image for Lambda..."
Set-Location -Path "..\.."
docker build -t "$($ECR_REPO):$IMAGE_TAG" -f deployments/serverless/Dockerfile .

Write-Host "4. Tagging and pushing image to ECR..."
docker tag "$($ECR_REPO):$IMAGE_TAG" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$($ECR_REPO):$IMAGE_TAG"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$($ECR_REPO):$IMAGE_TAG"

Write-Host "5. Updating/Creating Lambda function..."
aws lambda get-function --function-name $FUNCTION_NAME --region $AWS_REGION 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Updating existing function code..."
    aws lambda update-function-code `
        --function-name $FUNCTION_NAME `
        --image-uri "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$($ECR_REPO):$IMAGE_TAG" `
        --region $AWS_REGION | Out-Null
} else {
    Write-Host "Creating new function..."
    
    $ROLE_ARN = aws iam get-role --role-name lambda-ex --query Role.Arn --output text 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ROLE_ARN)) {
        Write-Host "Creating basic execution role..."
        
        $policyJson = @'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
'@
        $policyJson | Out-File -FilePath "trust-policy.json" -Encoding ascii
        
        aws iam create-role --role-name lambda-ex --assume-role-policy-document file://trust-policy.json | Out-Null
        aws iam attach-role-policy --role-name lambda-ex --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole | Out-Null
        $ROLE_ARN = aws iam get-role --role-name lambda-ex --query Role.Arn --output text
        
        Remove-Item "trust-policy.json" -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 10
    }

    aws lambda create-function `
        --function-name $FUNCTION_NAME `
        --package-type Image `
        --code ImageUri="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$($ECR_REPO):$IMAGE_TAG" `
        --role $ROLE_ARN `
        --timeout 30 `
        --memory-size 1024 `
        --region $AWS_REGION | Out-Null
}

Write-Host "=========================================================="
Write-Host "  Deploy Complete!"
Write-Host "  To test, go to AWS Console -> Lambda -> ml-inference-serverless"
Write-Host "  and click 'Test'."
Write-Host "=========================================================="
