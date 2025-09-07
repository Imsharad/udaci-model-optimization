#!/usr/bin/env python3
"""
Run the compression notebook step by step locally
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure we can import our modules
sys.path.append('.')

def main():
    print("🚀 Starting Compression Experiments")
    print("=" * 50)
    
    # Set device (MPS for M1/M2 Macs, CUDA for GPUs, CPU otherwise)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Using Metal Performance Shaders (MPS): {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA: {device}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU: {device}")
    
    # Import our custom modules
    try:
        from src.utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP, TARGET_MODEL_COMPRESSION
        print(f"✓ Constants loaded: {TARGET_MODEL_COMPRESSION=}, {TARGET_INFERENCE_SPEEDUP=}")
        
        from src.utils.data_loader import get_household_loaders, get_input_size
        from src.utils.model import MobileNetV3_Household, load_model, save_model
        print("✓ Custom modules imported successfully")
        
    except ImportError as e:
        print(f"✗ Error importing custom modules: {e}")
        return False
    
    # Load baseline model and metrics
    try:
        baseline_model_name = "baseline_mobilenet"
        baseline_model = load_model(f"models/{baseline_model_name}/checkpoints/model.pth", device)
        print("✓ Baseline model loaded successfully")
        
        with open(f"results/{baseline_model_name}/pretrained_metrics.json", 'r') as f:
            baseline_metrics = json.load(f)
        
        print("✓ Baseline metrics loaded:")
        print(f"  - Accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f}%")
        print(f"  - Model Size: {baseline_metrics['size']['model_size_mb']:.2f} MB")
        print(f"  - Inference Time (CPU): {baseline_metrics['timing']['cpu']['avg_time_ms']:.2f} ms")
        
    except Exception as e:
        print(f"✗ Error loading baseline: {e}")
        return False
    
    # Load dataset
    try:
        train_loader, test_loader = get_household_loaders(
            image_size="CIFAR", batch_size=128, num_workers=2,
        )
        input_size = get_input_size("CIFAR")
        class_names = train_loader.dataset.classes
        print(f"✓ Dataset loaded: {len(class_names)} classes, input size {input_size}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False
    
    print("\n🎯 Ready to run compression techniques!")
    print("All dependencies and baseline model are loaded successfully.")
    print("You can now run individual compression techniques or the full pipeline.")
    
    # Calculate target metrics
    target_model_size = baseline_metrics['size']['model_size_mb'] * (1 - TARGET_MODEL_COMPRESSION)
    target_inference_time_cpu = baseline_metrics['timing']['cpu']['avg_time_ms'] * (1 - TARGET_INFERENCE_SPEEDUP)
    min_acceptable_accuracy = baseline_metrics['accuracy']['top1_acc'] * (1 - MAX_ALLOWED_ACCURACY_DROP)
    
    print(f"\n📊 Optimization Targets:")
    print(f"  Target Model Size: {baseline_metrics['size']['model_size_mb']:.2f} → {target_model_size:.2f} MB ({TARGET_MODEL_COMPRESSION*100}% reduction)")
    print(f"  Target Inference Time: {baseline_metrics['timing']['cpu']['avg_time_ms']:.2f} → {target_inference_time_cpu:.2f} ms ({TARGET_INFERENCE_SPEEDUP*100}% reduction)")
    print(f"  Min Acceptable Accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f} → {min_acceptable_accuracy:.2f}% (within {MAX_ALLOWED_ACCURACY_DROP*100}% drop)")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Setup complete! Ready for compression experiments.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)