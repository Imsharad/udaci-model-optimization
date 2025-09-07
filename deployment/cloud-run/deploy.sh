#!/bin/bash
# Deploy ML Model to Google Cloud Run
# Based on Second Brain deployment configuration

set -e

# Configuration variables - Using Second Brain project
PROJECT_ID="${GCP_PROJECT_ID:-second-brain-463904}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-udaci-model-optimization}"
AR_REPOSITORY="${AR_REPOSITORY:-second-brain}"

# Resource configuration
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-2}"
TIMEOUT="${TIMEOUT:-300}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
CONCURRENCY="${CONCURRENCY:-10}"

echo "🚀 Deploying ${SERVICE_NAME} to Google Cloud Run..."
echo "📍 Project: ${PROJECT_ID}"
echo "🌍 Region: ${REGION}"
echo "💾 Memory: ${MEMORY}"
echo "🖥️  CPU: ${CPU}"

# Check if required tools are installed
command -v gcloud >/dev/null 2>&1 || { echo "❌ gcloud CLI is required but not installed. Aborting." >&2; exit 1; }

# Authenticate and configure gcloud if needed
if [ -n "$GCP_SA_KEY" ]; then
    echo "🔑 Using service account authentication"
    echo "$GCP_SA_KEY" | gcloud auth activate-service-account --key-file=-
fi

gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "📦 Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "$AR_REPOSITORY" --location="$REGION" >/dev/null 2>&1; then
    echo "Creating Artifact Registry repository: $AR_REPOSITORY"
    gcloud artifacts repositories create "$AR_REPOSITORY" \
        --repository-format=docker \
        --location="$REGION" \
        --description="ML models repository"
fi

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Build and deploy with Cloud Build (source-based deployment)
echo "🏗️  Building and deploying with Cloud Run..."

# Prepare environment variables (add your own as needed)
ENV_VARS=""
if [ -n "$MODEL_PATH" ]; then
    ENV_VARS="${ENV_VARS}MODEL_PATH=${MODEL_PATH},"
fi
if [ -n "$API_KEY" ]; then
    ENV_VARS="${ENV_VARS}API_KEY=${API_KEY},"
fi
# Remove trailing comma
ENV_VARS=$(echo "$ENV_VARS" | sed 's/,$//')

# Deploy to Cloud Run
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --port 8080 \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --timeout "$TIMEOUT" \
    --max-instances "$MAX_INSTANCES" \
    --min-instances "$MIN_INSTANCES" \
    --concurrency "$CONCURRENCY" \
    ${ENV_VARS:+--set-env-vars "$ENV_VARS"} \
    --execution-environment gen2 \
    --cpu-boost

echo "✅ Deployment completed!"
echo ""
echo "🔍 Getting service information..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="get(status.url)")
echo "🔗 Service URL: $SERVICE_URL"

echo ""
echo "📊 Testing deployment..."
if curl -f "${SERVICE_URL}/health" >/dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "⚠️  Health check failed - service may still be starting up"
fi

echo ""
echo "🎉 Deployment successful!"
echo "📝 Service: $SERVICE_NAME"
echo "🔗 URL: $SERVICE_URL"
echo "📍 Region: $REGION"
echo "💾 Resources: $CPU CPU, $MEMORY memory"