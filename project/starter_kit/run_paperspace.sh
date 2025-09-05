#!/bin/bash

# Run compression notebook on Paperspace Gradient
# Requires: pip install gradient

set -e

echo "Starting Paperspace Gradient job..."

gradient jobs create \
    --name "udaci-compression-experiments" \
    --projectId "your-project-id" \
    --machineType "P4000" \
    --container "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime" \
    --command "git clone https://github.com/Imsharad/udaci-model-optimization.git && cd udaci-model-optimization/project/starter-kit && python run_compression_headless.py" \
    --workspace "none"

echo "Job submitted to Paperspace Gradient"
echo "Monitor progress with: gradient jobs list"