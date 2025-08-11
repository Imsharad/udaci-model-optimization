#!/usr/bin/env python
"""
Baseline analysis starting with ImageNet pre-trained MobileNetV3
"""
import json
import os
import sys
import torch

# Add project root to path to import utils
sys.path.append('.')

from utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP, TARGET_MODEL_COMPRESSION
from utils.data_loader import get_household_loaders, get_input_size, print_dataloader_stats
from utils.model import MobileNetV3_Household, print_model_summary
from utils.evaluation import evaluate_model_metrics

def main():
    print("Starting UdaciSense Baseline Analysis with Pre-trained Weights...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    model_type = "baseline_mobilenet"
    models_dir = f"models/{model_type}"
    results_dir = f"results/{model_type}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=128, num_workers=2,
    )

    class_names = train_loader.dataset.classes
    print(f"Dataset classes: {class_names}")
    
    for dataset_type, data_loader in [('train', train_loader), ('test', test_loader)]:
        print_dataloader_stats(data_loader, dataset_type)
        
    # Initialize model with ImageNet pre-trained weights
    print("Initializing MobileNetV3 model with ImageNet pre-trained weights...")
    model = MobileNetV3_Household().to(device)  # This uses pre-trained weights by default
    print_model_summary(model)
    
    # Evaluate the pre-trained model (before fine-tuning)
    print("Evaluating pre-trained model performance...")
    n_classes = len(class_names)
    input_size = get_input_size("CIFAR")

    # Calculate baseline metrics on the pre-trained model
    baseline_metrics = evaluate_model_metrics(
        model, test_loader, device, n_classes, class_names, input_size, 
        save_path=f"{results_dir}/pretrained_metrics.json"
    )
    
    print(f"\nPre-trained Model Results (before fine-tuning):")
    print(f"Model Size: {baseline_metrics['size']['model_size_mb']:.2f} MB")
    print(f"Parameters: {baseline_metrics['size']['num_parameters']:,}")
    print(f"Top-1 Accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f}%")
    print(f"CPU Inference Time: {baseline_metrics['timing']['cpu']['avg_time_ms']:.2f} ms")
    if torch.cuda.is_available():
        print(f"GPU Inference Time: {baseline_metrics['timing']['cuda']['avg_time_ms']:.2f} ms")
    
    # Calculate target metrics based on CTO requirements
    target_model_size = baseline_metrics['size']['model_size_mb'] * (1 - TARGET_MODEL_COMPRESSION)
    target_inference_time_cpu = baseline_metrics['timing']['cpu']['avg_time_ms'] * (1 - TARGET_INFERENCE_SPEEDUP)
    min_acceptable_accuracy = baseline_metrics['accuracy']['top1_acc'] * (1 - MAX_ALLOWED_ACCURACY_DROP) 

    print(f"\nOptimization Targets (based on pre-trained model):")
    print(f"Target Model Size: {baseline_metrics['size']['model_size_mb']:.2f} → {target_model_size:.2f} MB ({TARGET_MODEL_COMPRESSION*100}% reduction)")
    print(f"Target CPU Inference Time: {baseline_metrics['timing']['cpu']['avg_time_ms']:.2f} → {target_inference_time_cpu:.2f} ms ({TARGET_INFERENCE_SPEEDUP*100}% reduction)")
    print(f"Min Acceptable Accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f} → {min_acceptable_accuracy:.2f}% (max {MAX_ALLOWED_ACCURACY_DROP*100}% drop)")
    
    # Note about fine-tuning
    print(f"\nNote: This is the pre-trained ImageNet model performance.")
    print(f"For production, you would fine-tune on the household objects dataset.")
    print(f"The pre-trained model gives us the baseline architecture metrics (size, speed).")
    print(f"Accuracy will improve significantly after fine-tuning on the target dataset.")
    
    print(f"\nPre-trained baseline analysis complete! Results saved to {results_dir}/")
    return baseline_metrics

if __name__ == "__main__":
    main()