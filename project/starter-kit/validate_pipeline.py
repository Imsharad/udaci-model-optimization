#!/usr/bin/env python3
"""
Pipeline Validation Script

This script validates the multi-stage compression pipeline by running it
with the existing baseline model and evaluating the results.
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add current directory to path for imports
sys.path.append(os.path.abspath('.'))

from compression.multi_stage_pipeline import MultiStageCompressionPipeline
from utils.model import MobileNetV3_Household
from utils.data_loader import get_household_loaders
from utils.evaluation import evaluate_accuracy


def load_existing_baseline_model():
    """Load the existing baseline model from the saved checkpoint."""
    model_path = "models/baseline_mobilenet/checkpoints/model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Baseline model not found at {model_path}")
    
    # Use the utils.model.load_model function
    from utils.model import load_model
    model = load_model(model_path, device='cpu', model_class=MobileNetV3_Household, num_classes=10)
    model.eval()
    
    print(f"Loaded baseline model from {model_path}")
    return model


def validate_individual_techniques():
    """Validate that individual compression techniques work properly."""
    print("\n" + "=" * 60)
    print("VALIDATING INDIVIDUAL TECHNIQUES")
    print("=" * 60)
    
    # Load model and data
    model = load_existing_baseline_model()
    _, test_loader = get_household_loaders(image_size="IMAGENET", batch_size=32)
    
    # Test pruning
    print("\n1. Testing Pruning...")
    from compression.post_training.pruning import prune_model
    from utils.compression import calculate_sparsity
    from utils.model import get_model_size
    
    # Create a copy for pruning
    import copy
    pruning_test_model = copy.deepcopy(model)
    
    # Apply pruning
    pruned_model = prune_model(
        pruning_test_model,
        pruning_method="l1_unstructured",
        amount=0.3
    )
    
    sparsity = calculate_sparsity(pruned_model)
    print(f"✅ Pruning successful. Sparsity: {sparsity:.1f}%")
    
    # Test quantization
    print("\n2. Testing Quantization...")
    from compression.post_training.quantization import quantize_model
    
    # Create a copy for quantization
    quantization_test_model = copy.deepcopy(model)
    
    # Apply quantization
    quantized_model = quantize_model(
        quantization_test_model,
        quantization_type="dynamic",
        backend="qnnpack"
    )
    
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    size_reduction = (1 - quantized_size / original_size) * 100
    
    print(f"✅ Quantization successful. Size reduction: {size_reduction:.1f}%")
    
    return True


def run_pipeline_validation():
    """Run the complete pipeline validation."""
    print("\n" + "=" * 60)
    print("RUNNING MULTI-STAGE PIPELINE")
    print("=" * 60)
    
    # Load model and data
    model = load_existing_baseline_model()
    _, test_loader = get_household_loaders(image_size="IMAGENET", batch_size=32)
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    print("Starting pipeline execution...")
    results = pipeline.run_pipeline(
        test_loader=test_loader,
        pruning_config=pruning_config,
        quantization_config=quantization_config,
        save_dir="models/multi_stage_pipeline",
        evaluate_intermediate=True
    )
    
    return results, pipeline


def analyze_results(results):
    """Analyze and summarize the pipeline results."""
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    baseline = results['baseline']
    final = results['final']
    
    # Calculate improvements
    size_reduction = (1 - final['size_mb'] / baseline['size_mb']) * 100
    speed_improvement = (1 - final['inference_time_ms'] / baseline['inference_time_ms']) * 100
    accuracy_change = final['accuracy'] - baseline['accuracy']
    
    print(f"Model Compression Summary:")
    print(f"  Original Size: {baseline['size_mb']:.2f} MB")
    print(f"  Compressed Size: {final['size_mb']:.2f} MB")
    print(f"  Size Reduction: {size_reduction:.1f}%")
    print(f"")
    print(f"  Original Inference Time: {baseline['inference_time_ms']:.1f} ms")
    print(f"  Compressed Inference Time: {final['inference_time_ms']:.1f} ms")
    print(f"  Speed Improvement: {speed_improvement:.1f}%")
    print(f"")
    print(f"  Original Accuracy: {baseline['accuracy']:.2f}%")
    print(f"  Compressed Accuracy: {final['accuracy']:.2f}%")
    print(f"  Accuracy Change: {accuracy_change:+.2f} percentage points")
    
    # Check CTO requirements
    print(f"\nCTO Requirements Check:")
    requirements_met = []
    
    # Size requirement (30%)
    size_ok = size_reduction >= 30
    requirements_met.append(size_ok)
    print(f"  Size Reduction >= 30%: {size_reduction:.1f}% {'✅' if size_ok else '❌'}")
    
    # Speed requirement (40%)
    speed_ok = speed_improvement >= 40
    requirements_met.append(speed_ok)
    print(f"  Speed Improvement >= 40%: {speed_improvement:.1f}% {'✅' if speed_ok else '❌'}")
    
    # Accuracy requirement (within 5%)
    accuracy_ok = abs(accuracy_change) <= 5
    requirements_met.append(accuracy_ok)
    print(f"  Accuracy within 5%: {abs(accuracy_change):.1f}% {'✅' if accuracy_ok else '❌'}")
    
    # Overall assessment
    all_met = all(requirements_met)
    print(f"\nOverall: {'✅ ALL REQUIREMENTS MET' if all_met else '❌ SOME REQUIREMENTS NOT MET'}")
    
    return all_met


def main():
    """Main validation function."""
    print("Multi-Stage Compression Pipeline Validation")
    print("=" * 60)
    
    try:
        # Step 1: Validate individual techniques
        validate_individual_techniques()
        
        # Step 2: Run pipeline validation
        results, pipeline = run_pipeline_validation()
        
        # Step 3: Analyze results
        success = analyze_results(results)
        
        # Step 4: Generate summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        if success:
            print("✅ Pipeline validation SUCCESSFUL!")
            print("   - All compression techniques work correctly")
            print("   - Multi-stage pipeline executes without errors")
            print("   - CTO requirements are met")
            print("   - Models are saved and ready for notebook integration")
        else:
            print("⚠️ Pipeline validation completed with WARNINGS:")
            print("   - Compression techniques work correctly")
            print("   - Pipeline executes without errors")
            print("   - Some CTO requirements may need optimization")
            print("   - Consider tuning hyperparameters or adding knowledge distillation")
        
        print(f"\nResults saved to: models/multi_stage_pipeline/pipeline_results.json")
        print("Ready to integrate into 03_pipeline.ipynb notebook!")
        
        return results
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()