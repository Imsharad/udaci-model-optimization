#!/bin/bash

# This script submits a headless Quantization-Aware Training job to Google Cloud Vertex AI.

# --- Configuration ---
# Please update these values to match your environment.

# Your Google Cloud Project ID.
PROJECT_ID="second-brain-463904"

# The Google Cloud region to run the job in.
REGION="us-central1"

# A unique name for your training job.
JOB_NAME="udacisense_qat_headless_$(date +%Y%m%d_%H%M%S)"

# The path to the project directory to be packaged.
# This should be the directory containing the 'setup.py' file.
PACKAGE_PATH="."

# The Python module to execute as the entry point for the training job.
PYTHON_MODULE="scripts.run_qat_headless"

# --- Worker Pool Configuration ---
# Specifies the machine type and GPU for the training job.

MACHINE_TYPE="n1-standard-4"
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT=1

# The pre-built PyTorch container image to use for training.
# Using a PyTorch 1.13 image with Python 3.10 and GPU support.
EXECUTOR_IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"


# --- Job Submission ---

echo "Submitting Vertex AI Custom Training Job..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Job Name: $JOB_NAME"
echo "Package Path: $PACKAGE_PATH"
echo "Python Module: $PYTHON_MODULE"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $ACCELERATOR_COUNT x $ACCELERATOR_TYPE"

gcloud ai custom-jobs create \
  --display-name="$JOB_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --worker-pool-spec="machine-type=$MACHINE_TYPE,replica-count=1,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERATOR_COUNT,executor-image-uri=$EXECUTOR_IMAGE_URI,local-package-path=$PACKAGE_PATH,python-module=$PYTHON_MODULE"

echo "✅ Job submitted."
echo "Monitor its progress in the Google Cloud Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
