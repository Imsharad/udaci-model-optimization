#!/bin/bash

# Run compression notebook on Vertex AI Workbench
# Usage: ./run_notebook_vertex.sh

set -e

PROJECT_ID="your-project-id"
REGION="us-central1"
INSTANCE_NAME="udaci-compression-notebook"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="NVIDIA_TESLA_T4"
GPU_COUNT=1

echo "Creating Vertex AI Workbench instance..."

gcloud notebooks instances create $INSTANCE_NAME \
    --vm-image-project=deeplearning-platform-release \
    --vm-image-family=pytorch-latest-gpu \
    --machine-type=$MACHINE_TYPE \
    --accelerator-type=$GPU_TYPE \
    --accelerator-core-count=$GPU_COUNT \
    --location=$REGION \
    --project=$PROJECT_ID

echo "Instance created. Getting instance details..."

# Get the JupyterLab URL
JUPYTER_URL=$(gcloud notebooks instances describe $INSTANCE_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(proxyUri)")

echo "JupyterLab URL: $JUPYTER_URL"
echo "Instance will be ready in 2-3 minutes."
echo ""
echo "To delete the instance when done:"
echo "gcloud notebooks instances delete $INSTANCE_NAME --location=$REGION --project=$PROJECT_ID"