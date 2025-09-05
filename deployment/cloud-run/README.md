# Google Cloud Run Deployment for Udacity Model Optimization

This directory contains all the necessary files to deploy your ML model compression experiments to Google Cloud Run, extracted and adapted from the Second Brain project's deployment configuration.

## 🚀 Quick Start

### 1. Setup Google Cloud Project

```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id-here"

# Login and set project
gcloud auth login
gcloud config set project $GCP_PROJECT_ID
```

### 2. Deploy the Basic Model Service

```bash
# Deploy general-purpose model serving service
./deploy.sh
```

### 3. Deploy the Compression Experiment Service

```bash
# Deploy compression experiment service (for running headless notebooks)
./deploy-compression.sh
```

## 📁 Files Overview

### Core Deployment Files

- **`Dockerfile`** - Multi-purpose container for ML model serving
- **`main.py`** - FastAPI service for model inference and health checks
- **`requirements.txt`** - Python dependencies for model serving
- **`deploy.sh`** - Deployment script for general model serving

### Compression-Specific Files

- **`compression-main.py`** - FastAPI service for running compression experiments
- **`compression-requirements.txt`** - Dependencies for compression experiments
- **`deploy-compression.sh`** - Deployment script for compression service

### CI/CD

- **`.github/workflows/deploy-model.yml`** - GitHub Actions workflow for automated deployment

## 🔧 Configuration

### Environment Variables

Set these in your deployment scripts or GitHub secrets:

```bash
# Required
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export SERVICE_NAME="your-service-name"

# Optional
export MEMORY="2Gi"              # Memory allocation
export CPU="2"                   # CPU allocation
export TIMEOUT="300"             # Request timeout (seconds)
export MAX_INSTANCES="10"        # Maximum instances
export MIN_INSTANCES="0"         # Minimum instances
```

### Model-Specific Environment Variables

```bash
# For model serving
export MODEL_PATH="/app/models/final_compressed_model.pth"
export MODEL_VERSION="latest"

# For compression experiments
export EXPERIMENT_CONFIG="dynamic_quantization,magnitude_pruning"
export ENABLE_GPU="true"         # Enable GPU support if available
```

## 🏗️ Deployment Options

### Option 1: Manual Deployment

```bash
# Basic model serving
./deploy.sh

# Or compression experiments
./deploy-compression.sh
```

### Option 2: GitHub Actions

1. Add secrets to your GitHub repository:
   - `GCP_SA_KEY` - Service account JSON key
   - `GCP_PROJECT_ID` - Your project ID

2. Push to main branch to trigger deployment:
   ```bash
   git add .
   git commit -m "Deploy model service"
   git push origin main
   ```

## 🧪 Testing Your Deployment

### Health Check

```bash
SERVICE_URL="https://your-service-url"
curl $SERVICE_URL/health
```

### Model Inference (General Service)

```bash
# Test prediction endpoint
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "your_input_data"}'
```

### Compression Experiments (Compression Service)

```bash
# Start compression experiments
curl -X POST $SERVICE_URL/experiments/start

# Check status
curl $SERVICE_URL/experiments/status

# Get results
curl $SERVICE_URL/experiments/results
```

## 📊 Resource Configuration

### For Model Serving

- **Memory**: 2Gi (adjust based on model size)
- **CPU**: 2 cores
- **Timeout**: 300 seconds
- **Concurrency**: 10 requests per instance

### For Compression Experiments

- **Memory**: 8Gi (ML experiments need more memory)
- **CPU**: 4 cores
- **Timeout**: 1800 seconds (30 minutes for long experiments)
- **Concurrency**: 1 (compute-intensive, single experiment at a time)

## 🔍 Monitoring and Logs

### View Logs

```bash
# View service logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=your-service-name" --limit 50

# Follow logs in real-time
gcloud logs tail "resource.type=cloud_run_revision AND resource.labels.service_name=your-service-name"
```

### Monitor Performance

```bash
# Get service details
gcloud run services describe your-service-name --region=us-central1

# View revisions
gcloud run revisions list --service=your-service-name --region=us-central1
```

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs
   gcloud builds log --region=us-central1
   ```

2. **Memory Issues**
   ```bash
   # Increase memory allocation
   export MEMORY="4Gi"
   ./deploy.sh
   ```

3. **Timeout Issues**
   ```bash
   # Increase timeout for long-running experiments
   export TIMEOUT="1800"
   ./deploy-compression.sh
   ```

### Service Account Permissions

Ensure your service account has these roles:
- Cloud Run Developer
- Cloud Build Editor
- Artifact Registry Writer
- Storage Object Viewer (if using Cloud Storage)

## 📚 Advanced Configuration

### Custom Dockerfile

Modify the `Dockerfile` to add your specific dependencies:

```dockerfile
# Add custom dependencies
RUN pip install your-custom-package

# Copy your model files
COPY models/ ./models/
```

### Environment-Specific Deployments

```bash
# Development
export SERVICE_NAME="udaci-model-dev"
export MEMORY="1Gi"
./deploy.sh

# Production
export SERVICE_NAME="udaci-model-prod"
export MEMORY="4Gi"
export MIN_INSTANCES="1"
./deploy.sh
```

## 🔗 Integration with Your Project

To integrate with your existing Udacity project:

1. Copy model files to the deployment directory:
   ```bash
   cp ../../project/starter_kit/models/final_compressed_model.pth ./models/
   ```

2. Update the `main.py` file to load your specific model:
   ```python
   model_path = "/app/models/final_compressed_model.pth"
   model = torch.load(model_path, map_location='cpu')
   ```

3. Deploy with your configuration:
   ```bash
   export MODEL_PATH="/app/models/final_compressed_model.pth"
   ./deploy.sh
   ```

This setup provides a complete, production-ready deployment pipeline based on the proven Second Brain architecture!