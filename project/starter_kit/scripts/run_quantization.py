#!/usr/bin/env python3
"""
Run post-training quantization experiment
"""

import os
import sys
import json
import torch
import torch.nn as nn

# Ensure we can import our modules
sys.path.append('.')

def main():
    print("🔬 Running Post-Training Dynamic Quantization")
    print("=" * 50)
    
    # Set device (CPU for quantization)
    device = torch.device('cpu')  # Quantization typically runs on CPU
    print(f"Using device: {device}")
    
    # Import our custom modules
    from src.utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP, TARGET_MODEL_COMPRESSION
    from src.utils.data_loader import get_household_loaders, get_input_size
    from src.utils.model import MobileNetV3_Household, load_model, save_model
    from src.compression.post_training.quantization import quantize_model
    from src.utils.compression import evaluate_optimized_model, compare_optimized_model_to_baseline, is_quantized
    
    # Load baseline model and metrics
    baseline_model_name = "baseline_mobilenet"
    baseline_model = load_model(f"models/{baseline_model_name}/checkpoints/model.pth", device)
    baseline_model.eval()
    
    with open(f"results/{baseline_model_name}/pretrained_metrics.json", 'r') as f:
        baseline_metrics = json.load(f)
    
    print("✓ Baseline model and metrics loaded")
    
    # Load dataset
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=128, num_workers=2,
    )
    input_size = get_input_size("CIFAR")
    class_names = train_loader.dataset.classes
    
    print("✓ Dataset loaded")
    
    # Configuration for quantization
    quantization_type = "dynamic"  # Dynamic quantization
    backend = "fbgemm"  # For x86 CPUs
    experiment_name = f"post_training/quantization/{quantization_type}"
    
    # Create experiment directories
    os.makedirs(f"models/{experiment_name}", exist_ok=True)
    os.makedirs(f"results/{experiment_name}", exist_ok=True)
    
    print(f"🎯 Applying {quantization_type} quantization with {backend} backend...")
    
    # Apply quantization
    quantized_model = quantize_model(
        baseline_model,
        quantization_type=quantization_type,
        calibration_data_loader=None,  # Not needed for dynamic quantization
        calibration_num_batches=None,
        backend=backend,
    )
    
    print("✓ Model quantized successfully")
    
    # Save the quantized model
    save_model(quantized_model, f"models/{experiment_name}/model.pth")
    print("✓ Quantized model saved")
    
    # Verify the model is quantized
    is_quantized_result = is_quantized(quantized_model)
    print(f"✓ Model quantization verified: {is_quantized_result}")
    
    # Evaluate the quantized model
    print("📊 Evaluating quantized model...")
    metrics, confusion_matrix = evaluate_optimized_model(
        quantized_model, 
        test_loader, 
        experiment_name,
        class_names,
        input_size,
        device=device,
    )
    
    print("✓ Quantized model evaluated")
    
    # Compare with baseline
    print("📈 Comparing with baseline...")
    comparison_results = compare_optimized_model_to_baseline(
        baseline_model,
        quantized_model,
        experiment_name,
        test_loader,
        class_names,
        device=device,
    )
    
    print("✅ Post-training quantization experiment completed!")
    print("\n📊 Results Summary:")
    
    # Load and display results
    try:
        with open(f"results/{experiment_name}/metrics.json", 'r') as f:
            quantized_metrics = json.load(f)
        
        print(f"Baseline → Quantized:")
        print(f"  Accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f}% → {quantized_metrics['accuracy']['top1_acc']:.2f}%")
        print(f"  Model Size: {baseline_metrics['size']['model_size_mb']:.2f} MB → {quantized_metrics['size']['model_size_mb']:.2f} MB")
        print(f"  Inference Time: {baseline_metrics['timing']['cpu']['avg_time_ms']:.2f} ms → {quantized_metrics['timing']['cpu']['avg_time_ms']:.2f} ms")
        
        # Calculate improvements
        size_reduction = (1 - quantized_metrics['size']['model_size_mb'] / baseline_metrics['size']['model_size_mb']) * 100
        speed_improvement = (1 - quantized_metrics['timing']['cpu']['avg_time_ms'] / baseline_metrics['timing']['cpu']['avg_time_ms']) * 100
        accuracy_change = quantized_metrics['accuracy']['top1_acc'] - baseline_metrics['accuracy']['top1_acc']
        
        print(f"\n🎯 Performance Changes:")
        print(f"  Size Reduction: {size_reduction:.1f}% (target: 70%)")
        print(f"  Speed Improvement: {speed_improvement:.1f}% (target: 60%)")
        print(f"  Accuracy Change: {accuracy_change:+.2f}% (target: ≤ 5% drop)")
        
    except Exception as e:
        print(f"⚠️  Could not load detailed results: {e}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Quantization experiment completed successfully!")
        else:
            print("\n❌ Quantization experiment failed.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)