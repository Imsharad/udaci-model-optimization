#!/bin/bash

# Setup GCP Compute Engine instance with GPU for headless notebook execution
# Usage: ./setup_gcp_instance.sh

set -e

PROJECT_ID="your-project-id"
ZONE="us-central1-a"
INSTANCE_NAME="udaci-compression-gpu"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "Creating GCP instance with GPU..."

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --boot-disk-device-name=$INSTANCE_NAME \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo "Instance created successfully!"
echo ""
echo "To connect to the instance:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "To run the compression experiments:"
echo "1. SSH into the instance"
echo "2. git clone https://github.com/Imsharad/udaci-model-optimization.git"
echo "3. cd udaci-model-optimization/project/starter-kit/"
echo "4. python run_compression_headless.py"
echo ""
echo "To copy results back:"
echo "gcloud compute scp $INSTANCE_NAME:~/udaci-model-optimization/project/starter-kit/compression_results_summary.json . --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "To delete the instance when done:"
echo "gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"