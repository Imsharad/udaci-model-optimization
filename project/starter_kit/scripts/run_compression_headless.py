#!/usr/bin/env python3
"""
Headless execution of compression notebook for cloud GPU environments.
Converts the interactive notebook into a script for automated execution.
"""

import os
import json
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compression_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_compression_experiments():
    """Run all compression experiments programmatically."""
    logger.info("Starting compression experiments...")
    
    # Import after installation
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    from src.compression.post_training.pruning import prune_model
    from src.compression.post_training.quantization import quantize_model
    from src.utils.data_loader import get_household_loaders
    from src.utils.model import load_model, save_model
    from src.utils.compression import evaluate_optimized_model, compare_optimized_model_to_baseline
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=128, num_workers=2,
    )
    class_names = train_loader.dataset.classes
    input_size = (1, 3, 32, 32)
    
    # Load baseline model
    logger.info("Loading baseline model...")
    baseline_model_name = "baseline_mobilenet"
    baseline_model = load_model(f"../models/{baseline_model_name}/checkpoints/model.pth", device)
    
    with open(f"../results/{baseline_model_name}/metrics.json", 'r') as f:
        baseline_metrics = json.load(f)
    
    experiments_results = {}
    
    # Experiment 1: Dynamic Quantization
    logger.info("Running dynamic quantization experiment...")
    try:
        experiment_name = "post_training/quantization/dynamic"
        os.makedirs(f"../models/{experiment_name}", exist_ok=True)
        os.makedirs(f"../results/{experiment_name}", exist_ok=True)
        
        orig_model = load_model(f"../models/{baseline_model_name}/checkpoints/model.pth").to(torch.device('cpu'))
        quantized_model = quantize_model(
            orig_model,
            quantization_type="dynamic",
            backend="fbgemm",
        )
        
        save_model(quantized_model, f"../models/{experiment_name}/model.pth")
        
        evaluate_optimized_model(
            quantized_model, test_loader, experiment_name, class_names, input_size, device=torch.device('cpu')
        )
        
        comparison_results = compare_optimized_model_to_baseline(
            baseline_model, quantized_model, experiment_name, test_loader, class_names, device=torch.device('cpu')
        )
        
        experiments_results['dynamic_quantization'] = comparison_results
        logger.info("Dynamic quantization completed successfully")
        
    except Exception as e:
        logger.error(f"Dynamic quantization failed: {str(e)}")
    
    # Experiment 2: Magnitude Pruning
    logger.info("Running magnitude pruning experiment...")
    try:
        experiment_name = "post_training/pruning/magnitude_0-3_cpu"
        os.makedirs(f"../models/{experiment_name}", exist_ok=True)
        os.makedirs(f"../results/{experiment_name}", exist_ok=True)
        
        orig_model = load_model(f"../models/{baseline_model_name}/checkpoints/model.pth").to(device)
        
        config = {
            'pruning_method': "magnitude",
            'amount': 0.3,
            'modules_to_prune': None,
            'n': None,
            'dim': None,
            'custom_pruning_fn': None,
            'device': device,
        }
        
        pruned_model = prune_model(
            orig_model, 
            config['pruning_method'], 
            config['amount'], 
            config["modules_to_prune"], 
            config["custom_pruning_fn"]
        )
        
        save_model(pruned_model, f"../models/{experiment_name}/model.pth")
        
        evaluate_optimized_model(
            pruned_model, test_loader, experiment_name, class_names, input_size, device=device
        )
        
        comparison_results = compare_optimized_model_to_baseline(
            baseline_model, pruned_model, experiment_name, test_loader, class_names, device=device
        )
        
        experiments_results['magnitude_pruning'] = comparison_results
        logger.info("Magnitude pruning completed successfully")
        
    except Exception as e:
        logger.error(f"Magnitude pruning failed: {str(e)}")
    
    # Save all results
    results_summary = {
        'experiments': experiments_results,
        'baseline_metrics': baseline_metrics,
        'device_used': str(device)
    }
    
    with open('compression_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    logger.info("All experiments completed. Results saved to compression_results_summary.json")
    return results_summary

def main():
    """Main execution function."""
    logger.info("Starting headless compression notebook execution...")
    
    # Create directories
    compression_types = [
        "post_training/pruning",
        "post_training/quantization",
        "post_training/graph_optimization",
        "in_training/distillation", 
        "in_training/quantization",
        "in_training/pruning",
    ]
    for comp_type in compression_types:
        models_dir = f"../models/{comp_type}"
        models_ckp_dir = f"{models_dir}/checkpoints"
        results_dir = f"../results/{comp_type}"
        
        os.makedirs(models_ckp_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    results = run_compression_experiments()
    
    logger.info("Execution completed successfully!")
    return results

if __name__ == "__main__":
    main()
