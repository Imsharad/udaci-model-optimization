#!/usr/bin/env python3
"""
Quick results collector for the compression pipeline.
Gets the models that have been created so far.
"""

import os
import json
import torch
from datetime import datetime

def collect_compression_results():
    """Collect results from the compression pipeline."""
    print("📋 Collecting Compression Results")
    print("=" * 50)
    
    os.chdir('project/starter_kit')
    
    # Check what models we have
    models_found = {}
    model_paths = {
        'baseline': 'models/baseline_mobilenet/checkpoints/model.pth',
        'pruned': 'models/final_submission_pipeline/pruned_model.pth', 
        'final_compressed': 'models/final_submission_pipeline/final_compressed_model.pth'
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            models_found[name] = {
                'path': path,
                'size_mb': round(size_mb, 2),
                'exists': True
            }
            print(f"✅ {name}: {path} ({size_mb:.2f} MB)")
        else:
            models_found[name] = {'exists': False}
            print(f"❌ {name}: Not found")
    
    # Load baseline metrics
    baseline_metrics = None
    if os.path.exists('results/baseline_mobilenet/pretrained_metrics.json'):
        with open('results/baseline_mobilenet/pretrained_metrics.json', 'r') as f:
            baseline_metrics = json.load(f)
            print(f"📊 Baseline accuracy: {baseline_metrics['accuracy']['top1_acc']:.2f}%")
    
    # Calculate quick compression ratio if we have both models
    if models_found['baseline']['exists'] and models_found['final_compressed']['exists']:
        baseline_size = models_found['baseline']['size_mb']
        compressed_size = models_found['final_compressed']['size_mb']
        compression_ratio = (baseline_size - compressed_size) / baseline_size * 100
        
        print(f"\n🎯 Quick Compression Analysis:")
        print(f"   Baseline: {baseline_size} MB")
        print(f"   Compressed: {compressed_size} MB")
        print(f"   Size Reduction: {compression_ratio:.1f}%")
        print(f"   Target Met (70%): {'✅' if compression_ratio >= 70 else '❌'}")
    
    # Create download package
    download_dir = 'models/'
    os.makedirs(download_dir, exist_ok=True)
    
    import shutil
    for name, info in models_found.items():
        if info['exists']:
            dest_path = f"{download_dir}/{name}_model.pth"
            shutil.copy2(info['path'], dest_path)
            print(f"📁 Copied {name} to {dest_path}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_available': models_found,
        'baseline_metrics': baseline_metrics,
        'download_location': os.path.abspath(download_dir)
    }
    
    with open(f"{download_dir}/compression_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n📦 Download Package Ready:")
    print(f"   Location: {os.path.abspath(download_dir)}")
    print(f"   Files: {len([f for f in models_found.values() if f['exists']])} models + summary")
    
    return summary

if __name__ == "__main__":
    results = collect_compression_results()
    print("\n✅ Results collected! Check ./compressed_models/ directory.")