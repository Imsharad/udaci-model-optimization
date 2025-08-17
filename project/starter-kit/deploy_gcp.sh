#!/bin/bash
# Deploy to Google Cloud Run

echo "🚀 Deploying Model Compression to Google Cloud Run"
echo "=================================================="

# Set your GCP project ID
PROJECT_ID="your-gcp-project-id"
SERVICE_NAME="model-compression"
REGION="us-central1"

echo "📝 Please update PROJECT_ID in this script with your actual GCP project ID"
echo "Current PROJECT_ID: $PROJECT_ID"
read -p "Press Enter to continue or Ctrl+C to exit..."

# Build and deploy
echo "🔨 Building and deploying..."

gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 1 \
    --allow-unauthenticated

echo "✅ Deployment complete!"
echo "🔗 Service URL will be shown above"