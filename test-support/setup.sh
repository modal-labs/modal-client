#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

echo "Deploying 'libmodal_test_support.py'..."
modal deploy test_support.py

echo "Deploying Secret 'libmodal-test-secret'..."
modal secret create --force libmodal-test-secret \
  a=1 b=2 c="hello world" >/dev/null

# Must be signed into AWS CLI for Modal Labs
echo "Deploying Secret 'libmodal-aws-ecr-test'..."
ecr_test_secret=$(aws secretsmanager get-secret-value \
  --secret-id test/libmodal/AwsEcrTest --query 'SecretString' --output text)
modal secret create --force libmodal-aws-ecr-test \
  AWS_ACCESS_KEY_ID="$(echo "$ecr_test_secret" | jq -r '.AWS_ACCESS_KEY_ID')" \
  AWS_SECRET_ACCESS_KEY="$(echo "$ecr_test_secret" | jq -r '.AWS_SECRET_ACCESS_KEY')" \
  AWS_REGION=us-east-1 \
  >/dev/null

echo "Deploying Secret 'libmodal-gcp-artifact-registry-test'..."
gcp_test_secret=$(aws secretsmanager get-secret-value \
  --secret-id test/libmodal/GcpArtifactRegistryTest --query 'SecretString' --output text)
modal secret create --force libmodal-gcp-artifact-registry-test \
  SERVICE_ACCOUNT_JSON="$(echo "$gcp_test_secret" | jq -r '.SERVICE_ACCOUNT_JSON')" \
  REGISTRY_USERNAME="_json_key" \
  REGISTRY_PASSWORD="$(echo "$gcp_test_secret" | jq -r '.SERVICE_ACCOUNT_JSON')" \
  >/dev/null

echo "Deploying Secret 'libmodal-anthropic-secret'..."
anthropic_api_key_secret=$(aws secretsmanager get-secret-value \
    --secret-id dev/libmodal/AnthropicApiKey --query 'SecretString' --output text | jq -r '.ANTHROPIC_API_KEY')
modal secret create --force libmodal-anthropic-secret \
  ANTHROPIC_API_KEY="$anthropic_api_key_secret" \
  >/dev/null

# deploy an app using an older version of Modal that's unsupported by libmodal
uv venv modal_1_1
uv pip install --python modal_1_1 "modal<1.2"
modal_1_1/bin/modal deploy test_support_1_1.py

echo
echo "NOTE! The tests also require a Proxy named 'libmodal-test-proxy', which cannot be created programmatically and must be created using the dashboard: https://modal.com/settings/modal-labs/proxy"
