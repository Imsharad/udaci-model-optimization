"""
UdaciSense Project: Evaluation Metrics Module

This module provides functions for evaluating models and generating comprehensive metrics
including accuracy, inference time, model size, and memory usage.
"""

import copy
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional, Union


# ------ ACCURACY EVALUATION FUNCTIONS ------ #

def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    topk: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    """Evaluate model accuracy on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        Dictionary of accuracy metrics
    """
    model.eval()
    
    model = model.to(device)
    
    # Initialize counters
    corrects = {k: 0 for k in topk}
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating accuracy"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate top-k accuracy
            _, preds = outputs.topk(max(topk), 1, True, True)
            preds = preds.t()
            correct = preds.eq(labels.view(1, -1).expand_as(preds))
            
            # Update counters
            batch_size = inputs.size(0)
            total += batch_size
            
            for i, k in enumerate(topk):
                corrects[k] += correct[:k].reshape(-1).float().sum(0).item()
    
    # Calculate accuracy
    accuracy = {f'top{k}_acc': 100.0 * corrects[k] / total for k in topk}
    
    return accuracy


def evaluate_per_class_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate per-class accuracy on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes in the dataset
        class_names: List of class names (optional)
        
    Returns:
        Dictionary of per-class accuracy percentages
    """
    model.eval()
    class_correct = [0.0] * num_classes
    class_total = [0.0] * num_classes
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating per-class accuracy"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Ensure predicted and targets are on the same device
            if predicted.device != targets.device:
                predicted = predicted.to(targets.device)
                
            c = (predicted == targets).squeeze()
            
            # Handle single-element batches
            if c.dim() == 0:
                c = c.unsqueeze(0)
                
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100.0 * class_correct[i] / class_total[i]
            if class_names is not None and i < len(class_names):
                per_class_accuracy[class_names[i]] = accuracy
            else:
                per_class_accuracy[f'class_{i}'] = accuracy
    
    return per_class_accuracy


def calculate_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int
) -> np.ndarray:
    """Calculate confusion matrix for a model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes in the dataset
        
    Returns:
        numpy.ndarray: Confusion matrix of shape [num_classes, num_classes]
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Calculating confusion matrix"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Make sure both tensors are on CPU before converting to NumPy
            targets_cpu = targets.cpu()
            predicted_cpu = predicted.cpu()
            
            for t, p in zip(targets_cpu.numpy(), predicted_cpu.numpy()):
                confusion_matrix[t, p] += 1
    
    return confusion_matrix


# ------ PERFORMANCE MEASUREMENT FUNCTIONS ------ #

def measure_inference_time(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, Dict[str, float]]:
    """Measure model inference time on both CPU and CUDA (if available).
    
    Args:
        model: PyTorch model
        input_size: Shape of input tensor
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        
    Returns:
        Dictionary of timing metrics for CPU and CUDA (if available):
        {
            'cpu': {
                'total_time_ms': float,
                'avg_time_ms': float,
                'fps': float
            },
            'cuda': {  # Only present if CUDA is available
                'total_time_ms': float,
                'avg_time_ms': float,
                'fps': float
            }
        }
    """
    results = {}

    # Check if the model is quantized, as INT8 operations are typically CPU-only
    is_quantized = any(
        'quantized' in str(type(m)).lower()
        for m in model.modules()
    )
    
    # Determine the original device of the model
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        # Handle cases where the model has no parameters (e.g., a container)
        # or is a TorchScript model where .parameters() is empty.
        # We'll assume CPU if we can't determine it, or try to infer from buffers.
        try:
            original_device = next(model.buffers()).device
        except StopIteration:
            original_device = torch.device('cpu') # Fallback

    # --- Measure on CPU ---
    try:
        cpu_device = torch.device('cpu')
        model.to(cpu_device)
        dummy_input_cpu = torch.randn(input_size, device=cpu_device)

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(dummy_input_cpu)
        
        # Measurement
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input_cpu)
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / num_runs
        
        results['cpu'] = {
            'total_time_ms': total_time_ms,
            'avg_time_ms': avg_time_ms,
            'fps': 1000 / avg_time_ms if avg_time_ms > 0 else 0
        }
    except Exception as e:
        print(f"WARNING: Could not measure CPU inference time: {e}")
        results['cpu'] = {}
    
    # --- Measure on CUDA if available ---
    if torch.cuda.is_available() and not is_quantized:
        try:
            cuda_device = torch.device('cuda')
            model.to(cuda_device)
            dummy_input_cuda = torch.randn(input_size, device=cuda_device)

            # Warmup
            torch.cuda.synchronize()
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = model(dummy_input_cuda)
                    torch.cuda.synchronize()
            
            # Measurement
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(dummy_input_cuda)
                    torch.cuda.synchronize()
            end_time = time.time()

            total_time_ms = (end_time - start_time) * 1000
            avg_time_ms = total_time_ms / num_runs

            results['cuda'] = {
                'total_time_ms': total_time_ms,
                'avg_time_ms': avg_time_ms,
                'fps': 1000 / avg_time_ms if avg_time_ms > 0 else 0
            }
        except Exception as e:
            print(f"WARNING: Could not measure CUDA inference time: {e}")
            results['cuda'] = {}
    elif is_quantized and torch.cuda.is_available():
        print("ℹ️ Skipping GPU inference timing for quantized model (INT8 ops are CPU-only).")
        results['cuda'] = {}


    # --- Restore model to original device ---
    try:
        model.to(original_device)
    except Exception as e:
        print(f"WARNING: Could not move model back to original device '{original_device}': {e}")
    
    return results


def measure_inference_time_fixed(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: torch.device, 
    num_batches: int = 50,
    warmup_batches: int = 5
) -> float:
    """
    CORRECTED timing measurement function that fixes the 11,886% slowdown issue.
    
    CRITICAL FIX: Eliminates model copying bug that caused massive timing errors
    Original broken approach: model_cpu = copy.deepcopy(model).to(self.cpu_device)
    Corrected approach: Use model directly on target device with proper warmup
    
    Args:
        model: PyTorch model to benchmark
        data_loader: DataLoader with test data (uses actual batch data instead of dummy)
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
            
            # Actual measurement using high precision timer
            start_time = time.perf_counter()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    if not times:
        raise ValueError("No valid timing measurements collected")
    
    return np.mean(times)


def measure_model_size(
    model: nn.Module,
    save_path: Optional[str] = None,
) -> Dict[str, Union[int, float]]:
    """Measure model size metrics with support for various model types.
    
    Args:
        model: PyTorch model, TorchScript model, or FX GraphModule
        save_path: Path to save the model before measuring (if None, uses a temporary file)
        
    Returns:
        Dictionary of size metrics (total_params, trainable_params, model_size_bytes, model_size_mb)
    """
    import tempfile
    import os
    
    # Torchscript models require slighly different handling, so let's create a variable to identify them
    is_torchscript = isinstance(model, torch.jit.ScriptModule)
    
    # Default values
    total_params = 0
    trainable_params = 0
    
    # Set up save_path
    if save_path is None:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"temp_model{'_ts' if is_torchscript else ''}.pt")
        save_path = temp_file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model
    if is_torchscript:
        torch.jit.save(model, save_path)
    else:
        torch.save(model.state_dict(), save_path)

    # Get file size
    model_size_bytes = os.path.getsize(save_path)
    model_size_mb = model_size_bytes / (1024 * 1024)  # Convert to MB

    # Count parameters
    # For TorchScript models, (very roughly) estimate parameter count from file size since not availabe from model
    if not is_torchscript:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = model_size_bytes // 4
        trainable_params = total_params  # All params in TorchScript are "trainable" 

    # Cleanup temporary file
    if save_path == temp_file:
        try:
            os.remove(temp_file)
        except:
            pass
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_bytes': model_size_bytes,
        'model_size_mb': model_size_mb
    }

def measure_memory_usage(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cuda')
) -> Dict[str, Optional[float]]:
    """Measure peak memory usage during inference.
    
    Args:
        model: PyTorch model
        input_size: Shape of input tensor
        device: Device to run inference on (must be CUDA)
        
    Returns:
        Dictionary of memory metrics (peak_memory_mb, allocated_memory_mb)
    """
    # Only applicable for CUDA devices
    if device.type != 'cuda':
        return {'peak_memory_mb': None, 'allocated_memory_mb': None}
    
    model.eval()
    model = model.to(device)
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Run inference
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Measure memory usage
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
    
    return {
        'peak_memory_mb': peak_memory,
        'allocated_memory_mb': allocated_memory
    }


# ------ COMPREHENSIVE EVALUATION FUNCTIONS ------ #

def evaluate_model_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate comprehensive model metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes in the dataset
        class_names: List of class names (optional)
        input_size: Shape of input tensor for inference time measurement
        save_path: Path to save the evaluation results (optional)
        
    Returns:
        Dictionary of metrics including accuracy, timing, size, and memory usage
    """
    # Evaluate accuracy
    accuracy_metrics = evaluate_accuracy(model, dataloader, device)

    # --- Start of Debugging Code ---
    print("\n--- Starting model stability check... ---")
    try:
        # Get a single batch from the evaluation dataloader
        debug_inputs, _ = next(iter(dataloader))
        debug_inputs = debug_inputs.to(device)

        print("Running a single forward pass on the quantized model...")
        with torch.no_grad():
            model.eval()
            outputs = model(debug_inputs)

        print("Single forward pass completed. Checking outputs for instability...")
        # Check for NaN (Not a Number) or Inf (Infinity) values
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("🔴 INSTABILITY DETECTED: Model outputs contain NaN or Inf values.")
            print("🔴 This is the likely cause of the memory crash.")
        else:
            print("✅ Stability check passed: No NaN or Inf values detected in the output.")
            print("   Output stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
                outputs.min(), outputs.max(), outputs.mean()
            ))

    except Exception as e:
        print(f"🔴 An error occurred during the stability check: {e}")
    print("--- End of Debugging Code ---\n")
    # --- End of Debugging Code ---
    
    # Evaluate per-class accuracy
    per_class_accuracy = evaluate_per_class_accuracy(
        model, dataloader, device, num_classes, class_names
    )
    
    # Measure inference time on both CPU and CUDA (if available)
    timing_metrics = measure_inference_time(model, input_size)
    
    # Measure model size
    size_metrics = measure_model_size(model)
    
    # Measure memory usage (if CUDA)
    memory_metrics = measure_memory_usage(model, input_size, device)
    
    # Compile all metrics
    metrics = {
        'accuracy': accuracy_metrics,
        'per_class_accuracy': per_class_accuracy,
        'timing': timing_metrics,
        'size': size_metrics,
        'memory': memory_metrics
    }
    
    # Optionally save metrics
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Baseline metrics saved at {save_path}.")
    
    return metrics


def compare_models(
    baseline_model: nn.Module,
    optimized_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    input_size: Tuple[int, ...] = (1, 3, 32, 32)
) -> Dict[str, Dict[str, Any]]:
    """Compare baseline and optimized models across multiple metrics.
    
    Args:
        baseline_model: Baseline PyTorch model
        optimized_model: Optimized PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes in the dataset
        class_names: List of class names (optional)
        input_size: Shape of input tensor for inference time measurement
        
    Returns:
        Dictionary of comparison metrics containing baseline, optimized, and improvements
    """
    # Evaluate metrics for both models
    print("Get metrics of baseline model...")
    baseline_metrics = evaluate_model_metrics(
        baseline_model, dataloader, device, num_classes, class_names, input_size
    )
    
    print("Get metrics of optimized model...")
    optimized_metrics = evaluate_model_metrics(
        optimized_model, dataloader, device, num_classes, class_names, input_size
    )
    
    # Calculate improvements
    improvements = {}
    
    # Size improvements
    if baseline_metrics['size']['model_size_mb'] > 0:
        improvements['size_reduction'] = 1.0 - (
            optimized_metrics['size']['model_size_mb'] / 
            baseline_metrics['size']['model_size_mb']
        )
    else:
        improvements['size_reduction'] = 0.0
    
    # Parameter count improvements
    improvements['params_reduction'] = 1.0 - (
        optimized_metrics['size']['trainable_params'] / 
        baseline_metrics['size']['trainable_params']
    )
    
    # Inference time improvements for both CPU and CUDA
    improvements['inference'] = {}
    
    # CPU inference improvements (if available)
    if baseline_metrics['timing'].get('cpu') and optimized_metrics['timing'].get('cpu'):
        if baseline_metrics['timing']['cpu']['avg_time_ms'] > 0:
            improvements['inference']['cpu'] = {
                'speedup': (
                    baseline_metrics['timing']['cpu']['avg_time_ms'] / 
                    optimized_metrics['timing']['cpu']['avg_time_ms']
                ),
                'reduction': 1.0 - (
                    optimized_metrics['timing']['cpu']['avg_time_ms'] / 
                    baseline_metrics['timing']['cpu']['avg_time_ms']
                )
            }
        else:
            improvements['inference']['cpu'] = {
                'speedup': 1.0,
                'reduction': 0.0
            }
    
    # CUDA inference improvements (if available)
    if baseline_metrics['timing'].get('cuda') and optimized_metrics['timing'].get('cuda'):
        if baseline_metrics['timing']['cuda']['avg_time_ms'] > 0:
            improvements['inference']['cuda'] = {
                'speedup': (
                    baseline_metrics['timing']['cuda']['avg_time_ms'] / 
                    optimized_metrics['timing']['cuda']['avg_time_ms']
                ),
                'reduction': 1.0 - (
                    optimized_metrics['timing']['cuda']['avg_time_ms'] / 
                    baseline_metrics['timing']['cuda']['avg_time_ms']
                )
            }
        else:
            improvements['inference']['cuda'] = {
                'speedup': 1.0,
                'reduction': 0.0
            }
    
    # Accuracy change
    improvements['accuracy_change'] = (
        optimized_metrics['accuracy']['top1_acc'] - 
        baseline_metrics['accuracy']['top1_acc']
    ) / 100.0  # Convert from percentage to fraction
    
    # Memory usage improvements (if CUDA)
    if (baseline_metrics['memory']['peak_memory_mb'] is not None and 
            baseline_metrics['memory']['peak_memory_mb'] > 0):
        improvements['memory_reduction'] = 1.0 - (
            optimized_metrics['memory']['peak_memory_mb'] / 
            baseline_metrics['memory']['peak_memory_mb']
        )
    else:
        improvements['memory_reduction'] = None
    
    return {
        'baseline': baseline_metrics,
        'optimized': optimized_metrics,
        'improvements': improvements
    }


def evaluate_requirements_met(
    comparison_results: Dict[str, Dict[str, Any]],
    size_requirement: float = 0.6,  # 60% reduction
    speed_requirement: float = 0.5,  # 50% reduction
    accuracy_requirement: float = 0.05,  # Within 5% of baseline
    device_priority: str = 'cpu'  # Which device to prioritize for speed requirements
) -> Dict[str, bool]:
    """Check if the optimized model meets the technical requirements.
    
    Args:
        comparison_results: Results from compare_models function
        size_requirement: Required model size reduction (0.0 to 1.0)
        speed_requirement: Required inference time reduction (0.0 to 1.0)
        accuracy_requirement: Maximum allowed accuracy drop (0.0 to 1.0)
        device_priority: Which device to prioritize for speed requirements ('cpu' or 'cuda')
        
    Returns:
        Dictionary indicating which requirements are met
    """
    improvements = comparison_results['improvements']
    
    # Check size requirement
    size_met = improvements['size_reduction'] >= size_requirement
    
    # Check speed requirement based on prioritized device
    speed_requirements = {}
    
    # Check CPU speed requirement
    if improvements['inference'].get('cpu'):
        speed_requirements['cpu'] = improvements['inference']['cpu']['reduction'] >= speed_requirement
    else:
        speed_requirements['cpu'] = False
    
    # Check CUDA speed requirement (if available)
    if improvements['inference'].get('cuda'):
        speed_requirements['cuda'] = improvements['inference']['cuda']['reduction'] >= speed_requirement
    else:
        speed_requirements['cuda'] = False
    
    # Use the prioritized device for the overall speed requirement
    speed_met = speed_requirements.get(device_priority, False)
    
    # Check accuracy requirement
    # This checks if the accuracy drop is within the allowed range
    # Note: accuracy_change is negative for a drop
    accuracy_met = abs(improvements['accuracy_change']) <= accuracy_requirement
    
    # Check all requirements
    all_met = size_met and speed_met and accuracy_met
    
    result = {
        'size_requirement_met': size_met,
        'accuracy_requirement_met': accuracy_met,
        'all_requirements_met': all_met
    }
    
    # Add device-specific speed requirements
    for device, met in speed_requirements.items():
        result[f'speed_requirement_met_{device}'] = met
    
    return result


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary of evaluation results
        output_path: Path to save the results
    """
    # Create a deep copy to avoid modifying the original results
    results_copy = json.loads(json.dumps(results, default=lambda x: str(x) if isinstance(x, (torch.Tensor, np.ndarray)) else x))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results_copy, f, indent=4)
    
    print(f"Evaluation results saved to {output_path}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in a model, broken down by trainable and non-trainable.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of model architecture and parameter counts.
    
    Args:
        model: PyTorch model
    """
    param_counts = count_parameters(model)
    
    print(f"Model Summary:")
    print(f"=============")
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"Trainable parameters %: {param_counts['trainable']/max(1, param_counts['total'])*100:.2f}%")
    
    # Count parameters by layer type
    layer_counts = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            layer_type = module.__class__.__name__
            
            if layer_type not in layer_counts:
                layer_counts[layer_type] = {
                    'count': 0,
                    'params': 0
                }
            
            layer_counts[layer_type]['count'] += 1
            layer_counts[layer_type]['params'] += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print("\nParameters by layer type:")
    for layer_type, info in sorted(layer_counts.items(), key=lambda x: x[1]['params'], reverse=True):
        if info['params'] > 0:
            print(f"  {layer_type}: {info['count']} layers, {info['params']:,} parameters " +
                  f"({info['params']/param_counts['trainable']*100:.2f}% of trainable)")