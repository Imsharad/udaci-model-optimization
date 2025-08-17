"""
Multi-Stage Compression Pipeline for UdaciSense Model Optimization

This module implements a comprehensive pipeline that combines multiple compression
techniques in an optimal sequence to achieve maximum model optimization.
"""

import os
import sys
import time
import copy
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.post_training.quantization import quantize_model
from compression.post_training.pruning import prune_model
from utils.model import load_model, MobileNetV3_Household, get_model_size
from utils.evaluation import evaluate_accuracy, measure_inference_time
from utils.compression import calculate_sparsity


class MultiStageCompressionPipeline:
    """
    A pipeline that applies multiple compression techniques in sequence.
    
    The pipeline follows the optimal order:
    1. Pruning (removes parameters first)
    2. Quantization (quantizes remaining parameters)
    3. Optional: Knowledge distillation for accuracy recovery
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize the compression pipeline.
        
        Args:
            model: Base model to compress
            device: Device to run on ('cpu' or 'cuda')
        """
        self.original_model = model
        self.device = device
        self.pipeline_results = {}
        self.compressed_models = {}
        
    def run_pipeline(
        self,
        test_loader: DataLoader,
        pruning_config: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        save_dir: str = "models/multi_stage_pipeline",
        evaluate_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete multi-stage compression pipeline.
        
        Args:
            test_loader: DataLoader for evaluation
            pruning_config: Configuration for pruning step
            quantization_config: Configuration for quantization step
            save_dir: Directory to save compressed models
            evaluate_intermediate: Whether to evaluate after each step
            
        Returns:
            Dictionary containing results from each stage
        """
        print("=" * 60)
        print("MULTI-STAGE COMPRESSION PIPELINE")
        print("=" * 60)
        
        # Default configurations
        if pruning_config is None:
            pruning_config = {
                "pruning_method": "l1_unstructured",
                "amount": 0.3
            }
            
        if quantization_config is None:
            quantization_config = {
                "quantization_type": "dynamic",
                "backend": "qnnpack"  # ARM compatible
            }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Baseline evaluation
        print("\n1. BASELINE EVALUATION")
        print("-" * 30)
        baseline_results = self._evaluate_stage(
            self.original_model, test_loader, "Baseline"
        )
        self.pipeline_results['baseline'] = baseline_results
        
        # Stage 1: Pruning
        print("\n2. STAGE 1: PRUNING")
        print("-" * 30)
        pruned_model = self._apply_pruning(pruning_config)
        self.compressed_models['pruned'] = pruned_model
        
        # Save pruned model
        pruned_path = os.path.join(save_dir, "pruned_model.pth")
        torch.save(pruned_model.state_dict(), pruned_path)
        print(f"Pruned model saved to: {pruned_path}")
        
        if evaluate_intermediate:
            pruned_results = self._evaluate_stage(
                pruned_model, test_loader, "Pruned"
            )
            self.pipeline_results['pruned'] = pruned_results
            self._print_comparison(baseline_results, pruned_results, "Pruning")
        
        # Stage 2: Quantization on pruned model
        print("\n3. STAGE 2: QUANTIZATION (on pruned model)")
        print("-" * 30)
        final_model = self._apply_quantization(pruned_model, quantization_config)
        self.compressed_models['final'] = final_model
        
        # Save final model
        final_path = os.path.join(save_dir, "final_compressed_model.pth")
        torch.save(final_model.state_dict(), final_path)
        print(f"Final compressed model saved to: {final_path}")
        
        # Final evaluation
        print("\n4. FINAL EVALUATION")
        print("-" * 30)
        final_results = self._evaluate_stage(
            final_model, test_loader, "Final (Pruned + Quantized)"
        )
        self.pipeline_results['final'] = final_results
        
        # Overall comparison
        self._print_final_comparison()
        
        # Check CTO requirements
        self._check_cto_requirements(baseline_results, final_results)
        
        # Save results
        results_path = os.path.join(save_dir, "pipeline_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.pipeline_results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        return self.pipeline_results
    
    def _apply_pruning(self, config: Dict[str, Any]) -> nn.Module:
        """Apply pruning to the original model."""
        print(f"Applying pruning: {config}")
        
        # Create a copy for pruning
        model_to_prune = copy.deepcopy(self.original_model)
        model_to_prune.eval()
        
        # Apply pruning
        pruned_model = prune_model(
            model_to_prune,
            pruning_method=config.get("pruning_method", "l1_unstructured"),
            amount=config.get("amount", 0.3)
        )
        
        return pruned_model
    
    def _apply_quantization(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Apply quantization to the given model."""
        print(f"Applying quantization: {config}")
        
        # Apply quantization
        quantized_model = quantize_model(
            model,
            quantization_type=config.get("quantization_type", "dynamic"),
            backend=config.get("backend", "qnnpack")
        )
        
        return quantized_model
    
    def _evaluate_stage(self, model: nn.Module, test_loader: DataLoader, stage_name: str) -> Dict[str, Any]:
        """Evaluate a model at a given stage."""
        print(f"Evaluating {stage_name} model...")
        
        # Model size
        size_mb = get_model_size(model)
        
        # Accuracy
        device_obj = torch.device(self.device)
        accuracy_metrics = evaluate_accuracy(model, test_loader, device_obj)
        accuracy = accuracy_metrics['top1_acc']  # Get top-1 accuracy
        
        # Inference time  
        # Use IMAGENET input size for MobileNetV3
        input_size = (1, 3, 224, 224)
        timing_results = measure_inference_time(
            model, input_size=input_size, num_runs=100, num_warmup=10
        )
        # Get CPU inference time (in ms)
        inference_time = timing_results['cpu']['avg_time_ms'] / 1000  # Convert to seconds
        
        # Sparsity (for pruned models)
        sparsity = calculate_sparsity(model)
        
        results = {
            "accuracy": float(accuracy),
            "size_mb": float(size_mb),
            "inference_time_ms": float(inference_time * 1000),  # Convert to ms
            "sparsity_percent": float(sparsity)
        }
        
        print(f"{stage_name} Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Inference Time: {inference_time*1000:.1f} ms")
        print(f"  Sparsity: {sparsity:.1f}%")
        
        return results
    
    def _print_comparison(self, baseline: Dict[str, Any], current: Dict[str, Any], stage: str):
        """Print comparison between baseline and current stage."""
        print(f"\n{stage} Impact:")
        
        size_reduction = (1 - current['size_mb'] / baseline['size_mb']) * 100
        speed_improvement = (1 - current['inference_time_ms'] / baseline['inference_time_ms']) * 100
        accuracy_change = current['accuracy'] - baseline['accuracy']
        
        print(f"  Size reduction: {size_reduction:.1f}%")
        print(f"  Speed improvement: {speed_improvement:.1f}%")
        print(f"  Accuracy change: {accuracy_change:+.1f} percentage points")
    
    def _print_final_comparison(self):
        """Print final comparison showing overall pipeline impact."""
        baseline = self.pipeline_results['baseline']
        final = self.pipeline_results['final']
        
        print("\n" + "=" * 60)
        print("FINAL PIPELINE RESULTS")
        print("=" * 60)
        
        print(f"{'Metric':<20} {'Baseline':<12} {'Final':<12} {'Change':<15}")
        print("-" * 60)
        
        # Size
        size_reduction = (1 - final['size_mb'] / baseline['size_mb']) * 100
        print(f"{'Size (MB)':<20} {baseline['size_mb']:<12.2f} {final['size_mb']:<12.2f} {size_reduction:>+12.1f}%")
        
        # Speed
        speed_improvement = (1 - final['inference_time_ms'] / baseline['inference_time_ms']) * 100
        print(f"{'Inference (ms)':<20} {baseline['inference_time_ms']:<12.1f} {final['inference_time_ms']:<12.1f} {speed_improvement:>+12.1f}%")
        
        # Accuracy
        accuracy_change = final['accuracy'] - baseline['accuracy']
        print(f"{'Accuracy (%)':<20} {baseline['accuracy']:<12.2f} {final['accuracy']:<12.2f} {accuracy_change:>+12.1f}pp")
        
        # Sparsity
        print(f"{'Sparsity (%)':<20} {baseline['sparsity_percent']:<12.1f} {final['sparsity_percent']:<12.1f} {'+' if final['sparsity_percent'] > baseline['sparsity_percent'] else ''}{final['sparsity_percent'] - baseline['sparsity_percent']:>11.1f}pp")
        
    def _check_cto_requirements(self, baseline: Dict[str, Any], final: Dict[str, Any]):
        """Check if the final model meets CTO requirements."""
        print("\n" + "=" * 60)
        print("CTO REQUIREMENTS CHECK")
        print("=" * 60)
        
        # Size reduction requirement: 30%
        size_reduction = (1 - final['size_mb'] / baseline['size_mb']) * 100
        size_meets = size_reduction >= 30
        
        # Speed improvement requirement: 40%
        speed_improvement = (1 - final['inference_time_ms'] / baseline['inference_time_ms']) * 100
        speed_meets = speed_improvement >= 40
        
        # Accuracy tolerance: within 5% of baseline
        accuracy_change = abs(final['accuracy'] - baseline['accuracy'])
        accuracy_meets = accuracy_change <= 5
        
        print(f"1. Size Reduction >= 30%:     {size_reduction:6.1f}%  {'✅ PASS' if size_meets else '❌ FAIL'}")
        print(f"2. Speed Improvement >= 40%:  {speed_improvement:6.1f}%  {'✅ PASS' if speed_meets else '❌ FAIL'}")
        print(f"3. Accuracy within 5%:        {accuracy_change:6.1f}%  {'✅ PASS' if accuracy_meets else '❌ FAIL'}")
        
        overall_pass = size_meets and speed_meets and accuracy_meets
        print(f"\nOVERALL: {'✅ ALL REQUIREMENTS MET' if overall_pass else '❌ REQUIREMENTS NOT MET'}")
        
        if not overall_pass:
            print("\nRecommendations:")
            if not size_meets:
                print("- Consider higher pruning ratio or static quantization")
            if not speed_meets:
                print("- Consider structured pruning or model architecture changes")
            if not accuracy_meets:
                print("- Consider knowledge distillation or lower compression ratios")


def main():
    """Main function to run the pipeline validation."""
    print("Multi-Stage Compression Pipeline Validation")
    print("=" * 50)
    
    # Load model and data
    print("Loading baseline model and data...")
    
    # This would need to be implemented based on your data loading
    # For now, we'll create a placeholder
    try:
        from utils.data_loader import get_household_loaders
        from utils.model import load_model
        
        # Load baseline model
        model = load_model("models/baseline_mobilenet/checkpoints/model.pth", model_class=MobileNetV3_Household, num_classes=10)
        
        # Load test data
        _, test_loader = get_household_loaders(image_size="IMAGENET", batch_size=32)
        
        print(f"Model loaded: {type(model).__name__}")
        print(f"Test data loaded: {len(test_loader)} batches")
        
    except ImportError:
        print("Warning: Could not import data/model utilities.")
        print("This validation script requires the full project setup.")
        return
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = MultiStageCompressionPipeline(model, device=device)
    
    # Configure compression stages
    pruning_config = {
        "pruning_method": "l1_unstructured",
        "amount": 0.3  # 30% sparsity
    }
    
    quantization_config = {
        "quantization_type": "dynamic",
        "backend": "qnnpack"  # ARM compatible
    }
    
    # Run pipeline
    results = pipeline.run_pipeline(
        test_loader=test_loader,
        pruning_config=pruning_config,
        quantization_config=quantization_config,
        save_dir="models/multi_stage_pipeline"
    )
    
    print("\nPipeline validation completed successfully!")
    return results


if __name__ == "__main__":
    main()