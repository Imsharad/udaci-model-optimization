"""
Fixed Inference Timing Measurement Utilities
Addresses critical timing measurement issues in the original pipeline
"""

import time
import torch
import numpy as np
from typing import Optional, Tuple


def measure_inference_time(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    num_batches: int = 100,
    warmup_batches: int = 5
) -> float:
    """
    Fixed timing measurement without model copying and with warmup.
    
    CRITICAL FIX: Eliminates the model copying bug that caused 11,886% slowdown
    Original issue: model_cpu = copy.deepcopy(model).to(self.cpu_device)
    
    Args:
        model: PyTorch model to benchmark
        data_loader: DataLoader with test data
        device: Device to run inference on (cuda/cpu)
        num_batches: Number of batches to benchmark
        warmup_batches: Number of warmup batches to discard
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    model = model.to(device)
    
    times = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches + warmup_batches:
                break
                
            data = data.to(device)
            
            # Warmup phase - discard these measurements
            if i < warmup_batches:
                _ = model(data)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                continue
            
            # Actual measurement
            start_time = time.perf_counter()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    if not times:
        raise ValueError("No valid timing measurements collected")
    
    return np.mean(times)


def measure_model_size(model: torch.nn.Module) -> float:
    """
    Calculate model size in MB accurately.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    total_params = 0
    total_size_bytes = 0
    
    for param in model.parameters():
        param_size = param.numel()
        param_bytes = param_size * param.element_size()
        total_params += param_size
        total_size_bytes += param_bytes
    
    # Convert bytes to MB
    size_mb = total_size_bytes / (1024 * 1024)
    
    return size_mb


def benchmark_model_comprehensive(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 100
) -> dict:
    """
    Comprehensive model benchmarking including accuracy, size, and timing.
    
    Args:
        model: PyTorch model to benchmark
        data_loader: DataLoader with test data
        device: Device to run inference on
        num_batches: Number of batches for timing measurement
        
    Returns:
        Dictionary with comprehensive metrics
    """
    # Accuracy measurement
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100.0 * correct / total
    
    # Size measurement
    size_mb = measure_model_size(model)
    
    # Timing measurement  
    avg_time_ms = measure_inference_time(
        model, data_loader, device, num_batches
    )
    
    return {
        'accuracy': {'top1_acc': accuracy},
        'size': {'model_size_mb': size_mb, 'total_params': sum(p.numel() for p in model.parameters())},
        'timing': {'cpu': {'avg_time_ms': avg_time_ms}} if device.type == 'cpu' else {'gpu': {'avg_time_ms': avg_time_ms}}
    }


def compare_models(
    baseline_metrics: dict,
    optimized_metrics: dict,
    targets: dict
) -> dict:
    """
    Compare baseline vs optimized model against CTO targets.
    
    Args:
        baseline_metrics: Baseline model metrics
        optimized_metrics: Optimized model metrics  
        targets: CTO targets (size_reduction, speed_improvement, max_accuracy_drop)
        
    Returns:
        Comparison results with CTO validation
    """
    baseline_acc = baseline_metrics['accuracy']['top1_acc']
    baseline_size = baseline_metrics['size']['model_size_mb']
    baseline_time = baseline_metrics['timing']['cpu']['avg_time_ms']
    
    final_acc = optimized_metrics['accuracy']['top1_acc']
    final_size = optimized_metrics['size']['model_size_mb']
    final_time = optimized_metrics['timing']['cpu']['avg_time_ms']
    
    # Calculate improvements
    size_reduction = (1 - final_size / baseline_size) * 100
    speed_improvement = (1 - final_time / baseline_time) * 100
    accuracy_drop = baseline_acc - final_acc
    
    # Check targets
    size_meets_target = size_reduction >= targets['size_reduction'] * 100
    speed_meets_target = speed_improvement >= targets['speed_improvement'] * 100
    accuracy_meets_target = accuracy_drop <= targets['max_accuracy_drop'] * 100
    
    return {
        'baseline': {
            'accuracy': baseline_acc,
            'size_mb': baseline_size,
            'time_ms': baseline_time
        },
        'optimized': {
            'accuracy': final_acc,
            'size_mb': final_size, 
            'time_ms': final_time
        },
        'improvements': {
            'size_reduction_pct': size_reduction,
            'speed_improvement_pct': speed_improvement,
            'accuracy_drop_pp': accuracy_drop
        },
        'cto_status': {
            'size_target_met': size_meets_target,
            'speed_target_met': speed_meets_target,
            'accuracy_target_met': accuracy_meets_target,
            'all_targets_met': size_meets_target and speed_meets_target and accuracy_meets_target
        }
    }