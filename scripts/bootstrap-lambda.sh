#!/usr/bin/env bash
set -euo pipefail

REGION="ap-southeast-1"
ACCOUNT_ID="886350084162"
ECR_REPO="isums/ocr-service"
LAMBDA_NAME="isums-ocr"
ROLE_NAME="isums-ocr-lambda-role"
RULE_NAME="isums-ocr-keep-warm"
SECRET_PARAM="/isums/ocr/shared-secret"
SCHEDULE_CRON="cron(*/4 0-15 ? * * *)"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || echo "")
if [ -z "$ROLE_ARN" ]; then
  TRUST=$(cat <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON
)
  ROLE_ARN=$(aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST" \
    --query 'Role.Arn' --output text)
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  sleep 10
fi
echo "Role: $ROLE_ARN"

SECRET=$(aws ssm get-parameter --name "$SECRET_PARAM" --with-decryption \
  --region "$REGION" --query 'Parameter.Value' --output text 2>/dev/null || echo "")
if [ -z "$SECRET" ]; then
  SECRET=$(openssl rand -hex 24)
  aws ssm put-parameter --name "$SECRET_PARAM" --type SecureString \
    --value "$SECRET" --region "$REGION" --overwrite
  echo "Generated new shared secret in SSM $SECRET_PARAM"
fi

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:lambda-latest"

if ! aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws lambda create-function \
    --function-name "$LAMBDA_NAME" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$ROLE_ARN" \
    --architectures x86_64 \
    --memory-size 10240 \
    --timeout 60 \
    --environment "Variables={OCR_SHARED_SECRET=$SECRET,OCR_CPU_THREADS=6,OCR_WORKER_THREADS=3,OCR_VERIFY_WORKERS=2}" \
    --region "$REGION"
else
  aws lambda update-function-configuration \
    --function-name "$LAMBDA_NAME" \
    --memory-size 10240 \
    --timeout 60 \
    --environment "Variables={OCR_SHARED_SECRET=$SECRET,OCR_CPU_THREADS=6,OCR_WORKER_THREADS=3,OCR_VERIFY_WORKERS=2}" \
    --region "$REGION" >/dev/null
fi

URL_CONF=$(aws lambda get-function-url-config --function-name "$LAMBDA_NAME" \
  --region "$REGION" 2>/dev/null || echo "")
if [ -z "$URL_CONF" ]; then
  aws lambda create-function-url-config \
    --function-name "$LAMBDA_NAME" \
    --auth-type NONE \
    --cors '{"AllowOrigins":["*"],"AllowMethods":["POST","GET"],"AllowHeaders":["*"]}' \
    --region "$REGION" >/dev/null
  aws lambda add-permission \
    --function-name "$LAMBDA_NAME" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE \
    --region "$REGION"
fi
FUNCTION_URL=$(aws lambda get-function-url-config --function-name "$LAMBDA_NAME" \
  --region "$REGION" --query 'FunctionUrl' --output text)
echo "Function URL: $FUNCTION_URL"

aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "$SCHEDULE_CRON" \
  --description "Keep OCR Lambda warm 07h-22h UTC+7" \
  --region "$REGION" >/dev/null

LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_NAME" \
  --region "$REGION" --query 'Configuration.FunctionArn' --output text)

aws events put-targets --rule "$RULE_NAME" --region "$REGION" --targets "[
  {\"Id\":\"warm-1\",\"Arn\":\"$LAMBDA_ARN\",\"Input\":\"{\\\"requestContext\\\":{\\\"http\\\":{\\\"method\\\":\\\"GET\\\",\\\"path\\\":\\\"/ping\\\"}},\\\"rawPath\\\":\\\"/ping\\\",\\\"warm\\\":1}\"},
  {\"Id\":\"warm-2\",\"Arn\":\"$LAMBDA_ARN\",\"Input\":\"{\\\"requestContext\\\":{\\\"http\\\":{\\\"method\\\":\\\"GET\\\",\\\"path\\\":\\\"/ping\\\"}},\\\"rawPath\\\":\\\"/ping\\\",\\\"warm\\\":2}\"}
]" >/dev/null

aws lambda add-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id "EventBridgeKeepWarm" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
  --region "$REGION" 2>/dev/null || true

echo ""
echo "Done. Function URL: $FUNCTION_URL"
echo "Shared secret param: $SECRET_PARAM"
echo "Schedule rule: $RULE_NAME ($SCHEDULE_CRON)"
