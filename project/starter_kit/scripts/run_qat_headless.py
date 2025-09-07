#!/usr/bin/env python3
"""
Headless execution of Quantization-Aware Training (QAT) for UdaciSense model.

This script runs the full QAT pipeline, including:
1.  Model preparation and calibration.
2.  Fine-tuning with simulated quantization.
3.  Conversion to a final, deployable INT8 model.
4.  Evaluation of the final model, including a stability check.
"""

import os
import json
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim

# --- Setup Project Path ---
# This allows the script to find the `src` module
# Assumes the script is run from the `project/starter_kit/scripts` directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project Root: {project_root}")

from src.compression.in_training.quantization import QuantizableMobileNetV3_Household, train_model_qat
from src.utils.data_loader import get_household_loaders
from src.utils.model import load_model
from src.utils.compression import evaluate_optimized_model, compare_optimized_model_to_baseline

# --- Setup Logging ---
log_file = 'qat_headless_run.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_qat_experiment():
    """Runs the full QAT experiment from configuration to evaluation."""
    logger.info("🚀 Starting Quantization-Aware Training (QAT) Experiment...")

    # --- 1. Configuration ---
    logger.info("Setting up configuration...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # This config should match the one in the notebook
    qat_config = {
        'num_epochs': 20,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer_class': optim.AdamW,
        'optimizer_kwargs': {'lr': 1e-4, 'weight_decay': 1e-4}, # Using a smaller LR for stability
        'scheduler_class': optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_kwargs': {'T_max': 20, 'eta_min': 1e-6},
        'patience': 5,
        'device': device,
        'grad_clip_norm': 1.0,
        'num_calibration_batches': 100,
    }
    logger.info(f"QAT Config: {qat_config}")

    # --- 2. Data Loading ---
    logger.info("Loading dataset...")
    # Using a larger batch size is fine for headless runs
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=256, num_workers=2
    )
    class_names = train_loader.dataset.classes
    input_size = (1, 3, 32, 32) # For evaluation metrics
    logger.info(f"Dataset loaded with {len(class_names)} classes.")

    # --- 3. Model Initialization ---
    logger.info("Initializing QAT-ready model...")
    # Important: `quantize` must be True to build a quantizable architecture
    qat_model = QuantizableMobileNetV3_Household(num_classes=len(class_names), quantize=True)
    qat_model.to(device)

    # --- 4. Run QAT Training ---
    # This calls the robust `train_model_qat` function we corrected
    experiment_name = "in_training/quantization/headless_qat_run"
    models_dir = os.path.join(project_root, f"models/{experiment_name}")
    results_dir = os.path.join(project_root, f"results/{experiment_name}")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(models_dir, "best_qat_model.pth")

    final_quantized_model, stats, best_acc, best_epoch = train_model_qat(
        model=qat_model,
        train_loader=train_loader,
        test_loader=test_loader,
        training_config=qat_config,
        checkpoint_path=checkpoint_path,
        backend="fbgemm" # Use fbgemm for x86 CPUs (like Colab)
    )
    logger.info(f"QAT training finished. Best accuracy: {best_acc:.2f}% at epoch {best_epoch}.")

    # --- 5. Evaluate Final Quantized Model ---
    logger.info("Evaluating final quantized model...")
    # The evaluation function now contains our stability check
    evaluate_optimized_model(
        optimized_model=final_quantized_model,
        data_loader=test_loader,
        technique_name=experiment_name,
        class_names=class_names,
        input_size=input_size,
        is_in_training_technique=True,
        training_stats=stats,
        device=torch.device('cpu') # Final evaluation is on CPU
    )

    # --- 6. Compare to Baseline ---
    logger.info("Comparing final model to baseline...")
    baseline_model_path = os.path.join(project_root, "models/baseline_mobilenet/checkpoints/model.pth")
    if os.path.exists(baseline_model_path):
        baseline_model = load_model(baseline_model_path, device)
        compare_optimized_model_to_baseline(
            baseline_model=baseline_model,
            optimized_model=final_quantized_model,
            technique_name=experiment_name,
            data_loader=test_loader,
            class_names=class_names,
            device=torch.device('cpu')
        )
    else:
        logger.warning("Baseline model not found. Skipping comparison.")

    logger.info("✅ QAT experiment script finished successfully!")


if __name__ == "__main__":
    run_qat_experiment()
