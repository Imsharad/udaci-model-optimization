#!/usr/bin/env python3
"""
Standalone GPU compression script for immediate execution.
No Cloud Run deployment needed - just run directly on GPU instance.
"""

import os
import json
import sys
import time
import torch
import base64
from datetime import datetime
from pathlib import Path

def download_compressed_models(models_dict, output_dir="./models"):
    """
    Save compressed models to local directory for manual download.
    
    Args:
        models_dict: Dictionary of model names and file paths
        output_dir: Local directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    download_info = {}
    
    print(f"\n📁 Saving compressed models to {output_dir}/")
    
    for model_name, model_path in models_dict.items():
        if os.path.exists(model_path):
            # Copy to output directory
            output_path = os.path.join(output_dir, f"{model_name}.pth")
            import shutil
            shutil.copy2(model_path, output_path)
            
            # Get file info
            file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
            download_info[model_name] = {
                "path": output_path,
                "size_mb": round(file_size, 2),
                "ready": True
            }
            
            print(f"✅ {model_name}: {output_path} ({file_size:.2f} MB)")
        else:
            print(f"❌ {model_name}: Not found at {model_path}")
            download_info[model_name] = {"ready": False, "error": "File not found"}
    
    # Save download manifest
    manifest_path = os.path.join(output_dir, "download_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "models": download_info,
            "total_models": len([m for m in download_info.values() if m.get("ready", False)])
        }, f, indent=4)
    
    print(f"📋 Download manifest: {manifest_path}")
    return download_info

def run_complete_compression_pipeline():
    """
    Complete compression pipeline that trains baseline and compresses to 70% target.
    """
    print("🚀 Starting Complete GPU Compression Pipeline")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Change to starter_kit directory
    os.chdir('project/starter_kit')
    
    # Add to Python path
    sys.path.insert(0, '.')
    
    # Step 1: Ensure baseline model exists
    print("\n📊 Step 1: Ensuring baseline model exists...")
    baseline_path = "models/baseline_mobilenet/checkpoints/model.pth"
    
    if not os.path.exists(baseline_path):
        print("Baseline model not found. Training baseline first...")
        from run_baseline import main as run_baseline
        baseline_metrics = run_baseline()
    else:
        print("✅ Baseline model found")
        # Check for metrics file
        metrics_path = "results/baseline_mobilenet/metrics.json"
        pretrained_metrics_path = "results/baseline_mobilenet/pretrained_metrics.json"
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                baseline_metrics = json.load(f)
        elif os.path.exists(pretrained_metrics_path):
            print("Using pretrained metrics file...")
            with open(pretrained_metrics_path, 'r') as f:
                baseline_metrics = json.load(f)
        else:
            print("No baseline metrics found. Running baseline evaluation...")
            from run_baseline import main as run_baseline
            baseline_metrics = run_baseline()
    
    print(f"Baseline size: {baseline_metrics['size']['model_size_mb']:.2f} MB")
    print(f"Baseline accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f}%")
    
    # Step 2: Run aggressive compression
    print("\n🗜️  Step 2: Running aggressive compression pipeline...")
    
    # Import compression modules
    from compression.post_training.pruning import prune_model
    from compression.post_training.quantization import quantize_model
    from utils.data_loader import get_household_loaders
    from utils.model import load_model, save_model
    from utils.compression import evaluate_optimized_model, compare_optimized_model_to_baseline
    
    # Load data
    train_loader, test_loader = get_household_loaders(
        image_size="CIFAR", batch_size=128, num_workers=2
    )
    class_names = train_loader.dataset.classes
    input_size = (1, 3, 32, 32)
    
    # Create output directories
    pipeline_name = "final_submission_pipeline"
    os.makedirs(f"models/{pipeline_name}", exist_ok=True)
    os.makedirs(f"results/{pipeline_name}", exist_ok=True)
    
    # Load baseline
    baseline_model = load_model(baseline_path, device)
    
    # Stage 1: Aggressive Pruning (65% sparsity for 70% size target)
    print("  🔪 Stage 1: Pruning (65% sparsity)...")
    pruned_model = load_model(baseline_path, device)
    pruned_model = prune_model(
        pruned_model,
        pruning_method="magnitude",
        amount=0.65,  # 65% sparsity
        modules_to_prune=None,
        custom_pruning_fn=None
    )
    
    pruned_path = f"models/{pipeline_name}/pruned_model.pth"
    save_model(pruned_model, pruned_path)
    
    # Stage 2: Dynamic Quantization
    print("  📊 Stage 2: Dynamic quantization...")
    pruned_cpu = pruned_model.to('cpu')
    quantized_model = quantize_model(
        pruned_cpu,
        quantization_type="dynamic",
        backend="fbgemm",
        calibration_data_loader=None
    )
    
    final_model_path = f"models/{pipeline_name}/final_compressed_model.pth"
    save_model(quantized_model, final_model_path)
    
    # Step 3: Evaluate final model
    print("\n📈 Step 3: Evaluating final compressed model...")
    
    final_metrics, final_confusion = evaluate_optimized_model(
        quantized_model, 
        test_loader, 
        pipeline_name, 
        class_names, 
        input_size, 
        device=torch.device('cpu')
    )
    
    comparison_results = compare_optimized_model_to_baseline(
        baseline_model.to('cpu'), 
        quantized_model, 
        pipeline_name, 
        test_loader, 
        class_names, 
        device=torch.device('cpu')
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
    
    # Final results
    results_summary = {
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
        'model_paths': {
            'baseline_model': baseline_path,
            'pruned_model': pruned_path,
            'final_compressed_model': final_model_path
        },
        'pipeline_info': {
            'stages': ['Magnitude Pruning (65%)', 'Dynamic Quantization'],
            'device_used': str(device),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Save results
    results_path = f"results/{pipeline_name}/submission_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # Step 4: Prepare download package
    print("\n📦 Step 4: Preparing download package...")
    
    models_for_download = {
        'final_compressed_model': final_model_path,
        'pruned_model': pruned_path,
        'baseline_model': baseline_path,
        'results': results_path
    }
    
    download_info = download_compressed_models(models_for_download)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 COMPRESSION PIPELINE RESULTS")
    print("=" * 60)
    print(f"Size Reduction: {size_reduction:.1f}% (Target: 70%)")
    print(f"Speed Improvement: {speed_improvement:.1f}% (Target: 60%)")
    print(f"Accuracy Drop: {accuracy_drop:.1f}% (Max: 5%)")
    print(f"Final Model Size: {final_size:.2f} MB (from {baseline_size:.2f} MB)")
    print(f"Final Accuracy: {final_acc:.2f}% (from {baseline_acc:.2f}%)")
    
    targets_met = all(results_summary['target_achievement'].values())
    print(f"\n✅ All targets met: {targets_met}")
    
    print(f"\n📁 Models ready for download in: ./models/")
    print("🎉 Pipeline complete! Ready for manual submission.")
    
    return results_summary

if __name__ == "__main__":
    try:
        results = run_complete_compression_pipeline()
        print("\n✅ Success! Check ./models/ for your submission files.")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()