#!/bin/bash
# Deploy Model Compression Service to Google Cloud Run
# Specialized for running ML compression experiments with GPU support

set -e

# Configuration variables - Using Second Brain project
PROJECT_ID="${GCP_PROJECT_ID:-second-brain-463904}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-udaci-compression-service}"
AR_REPOSITORY="${AR_REPOSITORY:-second-brain}"

# Resource configuration for ML workloads
MEMORY="${MEMORY:-8Gi}"
CPU="${CPU:-4}"
TIMEOUT="${TIMEOUT:-1800}"  # 30 minutes for long-running experiments
MAX_INSTANCES="${MAX_INSTANCES:-2}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
CONCURRENCY="${CONCURRENCY:-1}"  # Low concurrency for compute-intensive tasks

echo "🚀 Deploying ${SERVICE_NAME} to Google Cloud Run..."
echo "📍 Project: ${PROJECT_ID}"
echo "🌍 Region: ${REGION}"
echo "💾 Memory: ${MEMORY}"
echo "🖥️  CPU: ${CPU}"
echo "⏱️  Timeout: ${TIMEOUT}s"

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
        --description="ML compression models repository"
fi

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Prepare environment variables for ML workloads
ENV_VARS="PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1"

# Add GPU-related environment variables if needed
if [ "$ENABLE_GPU" = "true" ]; then
    ENV_VARS="${ENV_VARS},CUDA_VISIBLE_DEVICES=0"
fi

# Add experiment configuration
if [ -n "$EXPERIMENT_CONFIG" ]; then
    ENV_VARS="${ENV_VARS},EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}"
fi

# Deploy to Cloud Run with ML-optimized settings
echo "🏗️  Building and deploying compression service..."
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
    --set-env-vars "$ENV_VARS" \
    --execution-environment gen2 \
    --cpu-boost

echo "✅ Deployment completed!"
echo ""
echo "🔍 Getting service information..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="get(status.url)")
echo "🔗 Service URL: $SERVICE_URL"

echo ""
echo "📊 Testing compression service..."
if curl -f "${SERVICE_URL}/health" >/dev/null 2>&1; then
    echo "✅ Health check passed"
    
    # Test compression experiment endpoint
    echo "🧪 Testing experiment capabilities..."
    EXPERIMENT_STATUS=$(curl -s "${SERVICE_URL}/experiments/status" | jq -r '.status' 2>/dev/null || echo "unknown")
    echo "📊 Experiment status: $EXPERIMENT_STATUS"
else
    echo "⚠️  Health check failed - service may still be starting up"
fi

echo ""
echo "🎉 Compression service deployment successful!"
echo "📝 Service: $SERVICE_NAME"
echo "🔗 URL: $SERVICE_URL"
echo "📍 Region: $REGION"
echo "💾 Resources: $CPU CPU, $MEMORY memory"
echo "⏱️  Timeout: ${TIMEOUT}s"
echo ""
echo "🧪 Available endpoints:"
echo "  - Health: ${SERVICE_URL}/health"
echo "  - Start experiments: ${SERVICE_URL}/experiments/start"
echo "  - Check status: ${SERVICE_URL}/experiments/status"
echo "  - Get results: ${SERVICE_URL}/experiments/results"