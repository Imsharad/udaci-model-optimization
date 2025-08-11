#!/usr/bin/env python
"""
Quick baseline with 10 epochs fine-tuning for realistic accuracy baseline
"""
import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random

sys.path.append('.')

from utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP, TARGET_MODEL_COMPRESSION
from utils.data_loader import get_household_loaders, get_input_size
from utils.model import MobileNetV3_Household, train_model
from utils.evaluation import evaluate_model_metrics

def main():
    print("Fine-tuning MobileNetV3 for 10 epochs to establish realistic baseline...")
    
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    models_dir = "models/baseline_mobilenet"
    results_dir = "results/baseline_mobilenet" 
    os.makedirs(f"{models_dir}/checkpoints", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=64, num_workers=2
    )
    
    # Initialize model
    model = MobileNetV3_Household().to(device)
    
    # Quick training configuration
    training_config = {
        'num_epochs': 10,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4),
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4),
            max_lr=0.005,
            steps_per_epoch=len(train_loader),
            epochs=10,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000
        ),
        'patience': 5,
        'device': device
    }
    
    print("Starting 10-epoch fine-tuning...")
    training_stats, best_accuracy, best_epoch = train_model(
        model, train_loader, test_loader, training_config,
        checkpoint_path=f"{models_dir}/checkpoints/model_10epochs.pth"
    )
    
    # Evaluate fine-tuned model
    class_names = test_loader.dataset.classes
    baseline_metrics = evaluate_model_metrics(
        model, test_loader, device, len(class_names), class_names, 
        get_input_size("CIFAR"), save_path=f"{results_dir}/baseline_metrics.json"
    )
    
    # Calculate targets
    target_size = baseline_metrics['size']['model_size_mb'] * (1 - TARGET_MODEL_COMPRESSION) 
    target_time = baseline_metrics['timing']['cpu']['avg_time_ms'] * (1 - TARGET_INFERENCE_SPEEDUP)
    min_accuracy = baseline_metrics['accuracy']['top1_acc'] * (1 - MAX_ALLOWED_ACCURACY_DROP)
    
    print(f"\n=== BASELINE RESULTS (10-epoch fine-tuned) ===")
    print(f"Model Size: {baseline_metrics['size']['model_size_mb']:.2f} MB")
    print(f"Parameters: {baseline_metrics['size']['total_params']:,}")
    print(f"Accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f}%")
    print(f"CPU Inference: {baseline_metrics['timing']['cpu']['avg_time_ms']:.2f} ms")
    
    print(f"\n=== OPTIMIZATION TARGETS ===")
    print(f"Target Size: {target_size:.2f} MB (30% reduction)")
    print(f"Target Speed: {target_time:.2f} ms (40% improvement)")  
    print(f"Min Accuracy: {min_accuracy:.2f}% (≤5% drop)")
    
    return baseline_metrics

if __name__ == "__main__":
    main()