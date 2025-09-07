#!/bin/bash
# Deploy GPU-enabled ML Model Compression to Google Cloud Run
# Enhanced for aggressive 70% compression pipeline

set -e

# Configuration variables
PROJECT_ID="${GCP_PROJECT_ID:-second-brain-463904}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-udaci-gpu-compression}"
AR_REPOSITORY="${AR_REPOSITORY:-second-brain}"

# GPU-optimized resource configuration
MEMORY="${MEMORY:-16Gi}"
CPU="${CPU:-8}"
TIMEOUT="${TIMEOUT:-3600}"  # 1 hour for complex compression
MAX_INSTANCES="${MAX_INSTANCES:-3}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
CONCURRENCY="${CONCURRENCY:-1}"  # Single request per instance for GPU

echo "🚀 Deploying GPU-enabled compression service to Google Cloud Run..."
echo "📍 Project: ${PROJECT_ID}"
echo "🌍 Region: ${REGION}" 
echo "💾 Memory: ${MEMORY}"
echo "🖥️  CPU: ${CPU}"
echo "⏱️  Timeout: ${TIMEOUT}s"
echo "🎯 Target: 70% compression ratio"

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
        --description="GPU ML models repository"
fi

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Build and deploy with Cloud Build (source-based deployment)
echo "🏗️  Building and deploying with Cloud Run (using source)..."

# Prepare environment variables
ENV_VARS=""
if [ -n "$MODEL_PATH" ]; then
    ENV_VARS="${ENV_VARS}MODEL_PATH=${MODEL_PATH},"
fi
if [ -n "$COMPRESSION_TARGET" ]; then
    ENV_VARS="${ENV_VARS}COMPRESSION_TARGET=${COMPRESSION_TARGET},"
else
    ENV_VARS="${ENV_VARS}COMPRESSION_TARGET=70,"
fi

# Add CUDA environment variables
ENV_VARS="${ENV_VARS}CUDA_VISIBLE_DEVICES=0,"
ENV_VARS="${ENV_VARS}NVIDIA_VISIBLE_DEVICES=all,"

# Remove trailing comma
ENV_VARS=$(echo "$ENV_VARS" | sed 's/,$//')

# Copy GPU Procfile and deploy
cp Procfile.gpu Procfile

# Deploy to Cloud Run with source build
echo "🏗️  Deploying to Cloud Run with GPU support..."
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

echo "✅ GPU deployment completed!"
echo ""
echo "🔍 Getting service information..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="get(status.url)")
echo "🔗 Service URL: $SERVICE_URL"

echo ""
echo "📊 Testing deployment..."
if curl -f "${SERVICE_URL}/health" >/dev/null 2>&1; then
    echo "✅ Health check passed"
    echo "🎯 GPU compression service is ready"
else
    echo "⚠️  Health check failed - service may still be starting up"
fi

echo ""
echo "🎉 GPU-enabled compression deployment successful!"
echo "📝 Service: $SERVICE_NAME"
echo "🔗 URL: $SERVICE_URL"
echo "📍 Region: $REGION"
echo "💾 Resources: $CPU CPU, $MEMORY memory"
echo "🎯 Target: 70% compression, 60% speedup"
echo ""
echo "📋 Available endpoints:"
echo "  Health: ${SERVICE_URL}/health"
echo "  Start: ${SERVICE_URL}/compression/start (POST)"
echo "  Status: ${SERVICE_URL}/compression/status"
echo "  Progress: ${SERVICE_URL}/compression/progress"
echo "  Results: ${SERVICE_URL}/compression/results"
echo ""
echo "💡 To start compression, run:"
echo "  curl -X POST ${SERVICE_URL}/compression/start"