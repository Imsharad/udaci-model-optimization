#!/bin/bash
# Setup and run compression experiments
# Works on any Linux/Mac environment (local, GCP, AWS, etc.)

echo "🚀 Setting up Model Compression Environment"
echo "==========================================="

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Check if running on GPU-enabled environment
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo "✅ Dependencies installed"

# Run compression experiments
echo ""
echo "🔄 Running compression experiments..."
python run_compression_standalone.py

echo ""
echo "🎉 Compression experiments completed!"
echo "Check the results/ and models/ directories for outputs"