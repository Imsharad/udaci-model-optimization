#!/usr/bin/env python
"""
Run baseline analysis for the UdaciSense object recognition model
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

# Add project root to path to import utils
sys.path.append('.')

# Import custom modules
from utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP, TARGET_MODEL_COMPRESSION
from utils.data_loader import get_household_loaders, get_input_size, print_dataloader_stats, visualize_batch
from utils.model import MobileNetV3_Household, load_model, print_model_summary, train_model
from utils.evaluation import calculate_confusion_matrix, evaluate_model_metrics
from utils.visualization import plot_confusion_matrix, plot_training_history, plot_weight_distribution

def main():
    print("Starting UdaciSense Baseline Analysis...")
    
    # Check if CUDA is available
    devices = ["cpu"]
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        devices.extend([f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(num_devices)])
    print(f"Devices available: {devices}")

    # Set device to cuda, if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    def set_deterministic_mode(seed):
        # Basic seed setting
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make cudnn deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For some PyTorch operations
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # For DataLoader workers
        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        return seed_worker

    set_deterministic_mode(42)
    g = torch.Generator()
    g.manual_seed(42)
    
    # Create directories
    model_type = "baseline_mobilenet"
    models_dir = f"models/{model_type}"
    models_ckp_dir = f"{models_dir}/checkpoints"
    results_dir = f"results/{model_type}"

    os.makedirs(models_ckp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load household objects dataset
    print("Loading dataset...")
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=128, num_workers=2,
    )

    # Get class names
    class_names = train_loader.dataset.classes
    print(f"Datasets have these classes: ")
    for i in range(len(class_names)):
        print(f"  {i}: {class_names[i]}")

    # Print dataset statistics
    for dataset_type, data_loader in [('train', train_loader), ('test', test_loader)]:
        print(f"\nInformation on {dataset_type} set")
        print_dataloader_stats(data_loader, dataset_type)
        
    # Initialize model
    print("Initializing MobileNetV3 model...")
    model = MobileNetV3_Household().to(device)
    print_model_summary(model)
    
    # Define training configuration
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,  # Note that MobileNet is sensitive to high LRs
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,  # Peak learning rate
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,  # Spend 30% of training time warming up
        div_factor=25,  # Initial LR is max_lr/25
        final_div_factor=1000  # Final LR is max_lr/1000
    )

    training_config = {
        'num_epochs': num_epochs,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'patience': 5,
        'device': device
    }
    
    # Train model given the training_config
    print("Starting training...")
    training_stats, best_accuracy, best_epoch = train_model(
        model,
        train_loader,
        test_loader,
        training_config,
        checkpoint_path=f"{models_ckp_dir}/model.pth",
    )

    # Save training statistics
    with open(f"{results_dir}/training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=4)
    
    # Load the best model
    print("Evaluating baseline model...")
    model = load_model(f"{models_ckp_dir}/model.pth", device)

    # Define evaluation input and output variables
    class_names = test_loader.dataset.classes
    n_classes = len(class_names)
    input_size = get_input_size("CIFAR")

    # Calculate and save model performance on all metrics
    print("Evaluating model's performance on all metrics...")
    baseline_metrics = evaluate_model_metrics(model, test_loader, device, n_classes, class_names, input_size, save_path=f"{results_dir}/metrics.json")

    # Calculate, plot, and save confusion matrix
    confusion_matrix = calculate_confusion_matrix(model, test_loader, device, n_classes)
    _ = plot_confusion_matrix(confusion_matrix, class_names, f"{results_dir}/confusion_matrix.png")

    # Plot and save training history
    _ = plot_training_history(training_stats, f"{results_dir}/training_history.png")

    # Plot weight distribution (can help guide optimization strategies)
    _ = plot_weight_distribution(model, output_path=f"{results_dir}/weight_distribution.png")
    
    print(f"\nAll artifacts saved to:")
    print(f" - Model: {models_ckp_dir}/model.pth")
    print(f" - Metrics: {results_dir}/metrics.json")
    print(f" - Confusion Matrix: {results_dir}/confusion_matrix.png")
    print(f" - Training History: {results_dir}/training_history.png")
    print(f" - Training Stats: {results_dir}/training_stats.json")
    print(f" - Weight Distribution: {results_dir}/weight_distribution.png")
    
    # Calculate target metrics based on CTO requirements
    target_model_size = baseline_metrics['size']['model_size_mb'] * (1 - TARGET_MODEL_COMPRESSION)
    target_inference_time_cpu = baseline_metrics['timing']['cpu']['avg_time_ms'] * (1 - TARGET_INFERENCE_SPEEDUP)
    if torch.cuda.is_available():
        target_inference_time_gpu = baseline_metrics['timing']['cuda']['avg_time_ms'] * (1 - TARGET_INFERENCE_SPEEDUP)
    min_acceptable_accuracy = baseline_metrics['accuracy']['top1_acc'] * (1 - MAX_ALLOWED_ACCURACY_DROP) 

    print("\nOptimization Targets:")
    print(f"Target Model Size: {baseline_metrics['size']['model_size_mb']:.2f} --> {target_model_size:.2f} MB ({TARGET_MODEL_COMPRESSION*100}% reduction)")
    print(f"Target Inference Time (CPU): {baseline_metrics['timing']['cpu']['avg_time_ms']:.2f} --> {target_inference_time_cpu:.2f} ms ({TARGET_INFERENCE_SPEEDUP*100}% reduction)")
    if torch.cuda.is_available():
        print(f"Target Inference Time (GPU): {baseline_metrics['timing']['cuda']['avg_time_ms']:.2f} --> {target_inference_time_gpu:.2f} ms ({TARGET_INFERENCE_SPEEDUP*100}% reduction)")
    print(f"Minimum Acceptable Accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f} --> {min_acceptable_accuracy:.2f} (within {MAX_ALLOWED_ACCURACY_DROP*100}% of baseline)")
    
    print("\nBaseline analysis complete!")
    return baseline_metrics

if __name__ == "__main__":
    main()