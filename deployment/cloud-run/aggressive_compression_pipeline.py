#!/usr/bin/env python3
"""
Aggressive compression pipeline targeting 70% size reduction.
Combines multiple techniques: Pruning + Quantization + Graph Optimization
"""

import os
import json
import sys
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to python path
sys.path.append('/app/project/starter_kit')

def run_aggressive_compression_pipeline() -> Dict[str, Any]:
    """
    Run aggressive compression pipeline targeting 70% size reduction.
    Combines: High-ratio pruning + Dynamic quantization + Graph optimization
    """
    logger.info("Starting aggressive compression pipeline for 70% target...")
    
    # Import modules
    from compression.post_training.pruning import prune_model
    from compression.post_training.quantization import quantize_model
    from compression.post_training.graph_optimization import optimize_graph
    from utils.data_loader import get_household_loaders
    from utils.model import load_model, save_model
    from utils.compression import evaluate_optimized_model, compare_optimized_model_to_baseline
    
    # Set device (prefer GPU for faster processing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=128, num_workers=2
    )
    class_names = train_loader.dataset.classes
    input_size = (1, 3, 32, 32)
    n_classes = len(class_names)
    
    # Load baseline model and metrics
    logger.info("Loading baseline model...")
    baseline_model_path = "/app/project/starter_kit/models/baseline_mobilenet/checkpoints/model.pth"
    baseline_metrics_path = "/app/project/starter_kit/results/baseline_mobilenet/metrics.json"
    
    # Check if baseline exists
    if not os.path.exists(baseline_model_path):
        logger.error(f"Baseline model not found at {baseline_model_path}")
        # Run baseline first
        logger.info("Running baseline first...")
        os.chdir('/app/project/starter_kit')
        from run_baseline import main as run_baseline
        run_baseline()
    
    baseline_model = load_model(baseline_model_path, device)
    
    with open(baseline_metrics_path, 'r') as f:
        baseline_metrics = json.load(f)
    
    logger.info(f"Baseline model size: {baseline_metrics['size']['model_size_mb']:.2f} MB")
    logger.info(f"Baseline accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f}%")
    
    # Create output directories
    pipeline_name = "aggressive_compression_pipeline"
    os.makedirs(f"/app/project/starter_kit/models/{pipeline_name}", exist_ok=True)
    os.makedirs(f"/app/project/starter_kit/results/{pipeline_name}", exist_ok=True)
    
    # Stage 1: Aggressive Magnitude Pruning (60% sparsity)
    logger.info("Stage 1: Aggressive magnitude pruning (60% sparsity)...")
    
    # Clone model for pruning
    pruned_model = load_model(baseline_model_path, device)
    
    # Apply aggressive pruning
    pruning_config = {
        'pruning_method': "magnitude",
        'amount': 0.60,  # 60% sparsity for aggressive compression
        'modules_to_prune': None,
        'custom_pruning_fn': None,
    }
    
    pruned_model = prune_model(
        pruned_model,
        pruning_config['pruning_method'],
        pruning_config['amount'],
        pruning_config['modules_to_prune'],
        pruning_config['custom_pruning_fn']
    )
    
    # Save intermediate pruned model
    save_model(pruned_model, f"/app/project/starter_kit/models/{pipeline_name}/pruned_model.pth")
    logger.info("Stage 1 completed: Pruning applied")
    
    # Stage 2: Dynamic Quantization (CPU-optimized)
    logger.info("Stage 2: Dynamic quantization...")
    
    # Move to CPU for quantization
    pruned_model_cpu = pruned_model.to('cpu')
    
    quantized_model = quantize_model(
        pruned_model_cpu,
        quantization_type="dynamic",
        backend="fbgemm",
        calibration_data_loader=None,
    )
    
    # Save intermediate quantized model
    save_model(quantized_model, f"/app/project/starter_kit/models/{pipeline_name}/quantized_model.pth")
    logger.info("Stage 2 completed: Quantization applied")
    
    # Stage 3: Graph Optimization
    logger.info("Stage 3: Graph optimization...")
    
    try:
        optimized_model = optimize_graph(
            quantized_model,
            input_size,
            optimization_level="aggressive"
        )
        save_model(optimized_model, f"/app/project/starter_kit/models/{pipeline_name}/final_optimized_model.pth")
        final_model = optimized_model
        logger.info("Stage 3 completed: Graph optimization applied")
    except Exception as e:
        logger.warning(f"Graph optimization failed: {e}. Using quantized model as final.")
        final_model = quantized_model
    
    # Final Evaluation
    logger.info("Evaluating final compressed model...")
    
    # Evaluate on CPU (quantized models typically run on CPU)
    eval_device = torch.device('cpu')
    final_model = final_model.to(eval_device)
    
    # Run comprehensive evaluation
    final_metrics = evaluate_optimized_model(
        final_model, 
        test_loader, 
        pipeline_name, 
        class_names, 
        input_size, 
        device=eval_device
    )
    
    # Compare with baseline
    comparison_results = compare_optimized_model_to_baseline(
        baseline_model.to(eval_device), 
        final_model, 
        pipeline_name, 
        test_loader, 
        class_names, 
        device=eval_device
    )
    
    # Calculate compression metrics
    baseline_size = baseline_metrics['size']['model_size_mb']
    final_size = final_metrics['size']['model_size_mb']
    size_reduction = (baseline_size - final_size) / baseline_size * 100
    
    baseline_time = baseline_metrics['timing']['cpu']['avg_time_ms']
    final_time = final_metrics['timing']['cpu']['avg_time_ms']
    speed_improvement = (baseline_time - final_time) / baseline_time * 100
    
    baseline_acc = baseline_metrics['accuracy']['top1_acc']
    final_acc = final_metrics['accuracy']['top1_acc']
    accuracy_drop = baseline_acc - final_acc
    
    # Results summary
    results_summary = {
        'pipeline_stages': [
            'Magnitude Pruning (60% sparsity)',
            'Dynamic Quantization (FBGEMM)',
            'Graph Optimization'
        ],
        'compression_results': {
            'size_reduction_percent': size_reduction,
            'speed_improvement_percent': speed_improvement,
            'accuracy_drop_percent': accuracy_drop,
            'baseline_size_mb': baseline_size,
            'final_size_mb': final_size,
            'baseline_accuracy': baseline_acc,
            'final_accuracy': final_acc,
            'baseline_time_ms': baseline_time,
            'final_time_ms': final_time
        },
        'target_achievement': {
            'size_target_70_percent': size_reduction >= 70.0,
            'speed_target_60_percent': speed_improvement >= 60.0,
            'accuracy_within_5_percent': accuracy_drop <= 5.0
        },
        'detailed_metrics': final_metrics,
        'comparison_results': comparison_results,
        'device_used': str(device)
    }
    
    # Save comprehensive results
    results_path = f"/app/project/starter_kit/results/{pipeline_name}/comprehensive_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # Log final results
    logger.info("=" * 60)
    logger.info("AGGRESSIVE COMPRESSION PIPELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Size Reduction: {size_reduction:.1f}% (Target: 70%)")
    logger.info(f"Speed Improvement: {speed_improvement:.1f}% (Target: 60%)")
    logger.info(f"Accuracy Drop: {accuracy_drop:.1f}% (Max allowed: 5%)")
    logger.info(f"Final Model Size: {final_size:.2f} MB (from {baseline_size:.2f} MB)")
    logger.info(f"Final Accuracy: {final_acc:.2f}% (from {baseline_acc:.2f}%)")
    logger.info("=" * 60)
    
    # Check if targets are met
    targets_met = all(results_summary['target_achievement'].values())
    logger.info(f"All targets met: {targets_met}")
    
    if not targets_met:
        logger.warning("Some targets not met. Consider:")
        if size_reduction < 70:
            logger.warning("- Increase pruning ratio to 70-80%")
        if speed_improvement < 60:
            logger.warning("- Apply more aggressive quantization (int8)")
        if accuracy_drop > 5:
            logger.warning("- Fine-tune after compression or use knowledge distillation")
    
    return results_summary

if __name__ == "__main__":
    results = run_aggressive_compression_pipeline()
    print(f"Pipeline completed. Results saved.")