"""
Unstructured Pruning for MobileNetV3 Architecture

CRITICAL CORRECTION: Replaces structured pruning which was architecturally incompatible
with MobileNetV3's inverted residual bottlenecks.

Unstructured pruning works by:
1. Removing individual weights (not entire channels/filters) 
2. Preserving architectural integrity of inverted bottlenecks
3. Creating sparsity that can be accelerated by sparse-aware kernels

Compatible with: MobileNetV3, EfficientNet, and other bottleneck architectures
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import List, Tuple, Dict, Optional


class UnstructuredPruner:
    """
    Unstructured magnitude-based pruning for MobileNetV3 models.
    
    Key advantages over structured pruning:
    - Preserves architectural integrity of inverted residual blocks
    - No information bottlenecks in narrow layers
    - Compatible with sparse-aware inference kernels
    - Can achieve high sparsity (50-70%) without catastrophic accuracy loss
    """
    
    def __init__(self, model: nn.Module, exclude_layers: Optional[List[str]] = None):
        """
        Initialize pruner for a model.
        
        Args:
            model: PyTorch model to prune
            exclude_layers: Layer names to exclude from pruning (e.g., first/last layers)
        """
        self.model = model
        self.exclude_layers = exclude_layers or []
        self.pruned_modules = []
        
    def identify_prunable_layers(self) -> List[Tuple[nn.Module, str]]:
        """
        Identify layers suitable for unstructured pruning.
        
        For MobileNetV3:
        - Conv2d layers (except first layer)
        - Linear layers in classifier (except output layer)
        - Avoid batch norm and activation layers
        
        Returns:
            List of (module, parameter_name) tuples to prune
        """
        modules_to_prune = []
        
        for name, module in self.model.named_modules():
            # Skip excluded layers
            if any(exclude in name for exclude in self.exclude_layers):
                continue
                
            # Target Conv2d and Linear layers with sufficient parameters
            if isinstance(module, nn.Conv2d):
                # Skip first convolutional layer (critical for feature extraction)
                if 'features.0.0' in name:  # MobileNetV3 first layer
                    continue
                # Skip layers with very few parameters
                if module.weight.numel() < 100:
                    continue
                modules_to_prune.append((module, 'weight'))
                
            elif isinstance(module, nn.Linear):
                # Skip final classification layer
                if name.endswith('classifier.6') or name.endswith('classifier.-1'):
                    continue
                # Skip very small layers
                if module.weight.numel() < 100:
                    continue
                modules_to_prune.append((module, 'weight'))
        
        return modules_to_prune
    
    def apply_magnitude_pruning(
        self, 
        sparsity: float,
        importance_scores: Optional[Dict] = None
    ) -> Dict:
        """
        Apply magnitude-based unstructured pruning.
        
        Args:
            sparsity: Target sparsity level (0.0-1.0)
            importance_scores: Optional per-layer importance scores for adaptive pruning
            
        Returns:
            Dictionary with pruning statistics
        """
        print(f"🔧 Applying unstructured magnitude pruning (target: {sparsity*100:.0f}% sparsity)")
        
        # Identify layers to prune
        modules_to_prune = self.identify_prunable_layers()
        print(f"   Targeting {len(modules_to_prune)} layers for pruning")
        
        # Apply global magnitude-based pruning
        if importance_scores is None:
            # Global pruning: prune lowest magnitude weights across all layers
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
        else:
            # Layer-wise adaptive pruning based on importance scores
            for (module, param_name), importance in zip(modules_to_prune, importance_scores.values()):
                # Adjust sparsity based on layer importance
                layer_sparsity = max(0.1, sparsity * (2.0 - importance))  # More important = less pruning
                layer_sparsity = min(0.9, layer_sparsity)  # Cap at 90%
                
                prune.l1_unstructured(module, name=param_name, amount=layer_sparsity)
        
        # Store pruned modules for later operations
        self.pruned_modules = modules_to_prune
        
        # Calculate actual sparsity achieved
        stats = self.calculate_sparsity_stats()
        
        print(f"   ✅ Achieved sparsity: {stats['global_sparsity']*100:.1f}%")
        print(f"   📊 Parameters: {stats['total_params']:,} total, {stats['sparse_params']:,} sparse")
        
        return stats
    
    def calculate_sparsity_stats(self) -> Dict:
        """Calculate detailed sparsity statistics."""
        total_params = 0
        sparse_params = 0
        layer_stats = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                layer_total = module.weight.numel()
                layer_sparse = torch.sum(module.weight == 0).item()
                
                total_params += layer_total
                sparse_params += layer_sparse
                
                layer_sparsity = layer_sparse / layer_total if layer_total > 0 else 0
                layer_stats[name] = {
                    'total_params': layer_total,
                    'sparse_params': layer_sparse,
                    'sparsity': layer_sparsity
                }
        
        global_sparsity = sparse_params / total_params if total_params > 0 else 0
        
        return {
            'global_sparsity': global_sparsity,
            'total_params': total_params,
            'sparse_params': sparse_params,
            'layer_stats': layer_stats
        }
    
    def make_pruning_permanent(self):
        """
        Remove pruning masks and make sparsity permanent.
        This converts masked weights to actual zeros.
        """
        print("🔧 Making pruning permanent...")
        
        for module, param_name in self.pruned_modules:
            if hasattr(module, f'{param_name}_mask'):
                prune.remove(module, param_name)
        
        print("   ✅ Pruning masks removed, sparsity is now permanent")
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio from sparsity."""
        stats = self.calculate_sparsity_stats()
        # Compression ratio = 1 / (1 - sparsity)
        # E.g., 60% sparsity = 1/(1-0.6) = 2.5x compression
        return 1.0 / (1.0 - stats['global_sparsity']) if stats['global_sparsity'] < 1.0 else float('inf')


def calculate_layer_importance_scores(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 50
) -> Dict[str, float]:
    """
    Calculate importance scores for each layer based on gradient magnitudes.
    Higher scores = more important layers that should be pruned less aggressively.
    
    Args:
        model: Model to analyze
        data_loader: DataLoader for importance calculation
        device: Device to run on
        num_batches: Number of batches to use for scoring
        
    Returns:
        Dictionary mapping layer names to importance scores (0.0-2.0)
    """
    print("📊 Calculating layer importance scores...")
    
    model.eval()
    model.to(device)
    
    # Accumulate gradients for importance scoring
    importance_scores = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
            importance_scores[name] = 0.0
    
    # Calculate gradients on representative data
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Accumulate gradient magnitudes
        for name, module in model.named_modules():
            if name in importance_scores and module.weight.grad is not None:
                importance_scores[name] += torch.mean(torch.abs(module.weight.grad)).item()
    
    # Normalize scores to 0.0-2.0 range (1.0 = average importance)
    if importance_scores:
        values = list(importance_scores.values())
        mean_importance = np.mean(values)
        std_importance = np.std(values) if len(values) > 1 else 1.0
        
        for name in importance_scores:
            # Normalize and clamp to [0.5, 2.0] range
            normalized = (importance_scores[name] - mean_importance) / (std_importance + 1e-8)
            importance_scores[name] = np.clip(1.0 + 0.5 * normalized, 0.5, 2.0)
    
    print(f"   ✅ Calculated importance for {len(importance_scores)} layers")
    
    return importance_scores


def apply_gradual_magnitude_pruning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    target_sparsity: float = 0.6,
    num_epochs: int = 10,
    pruning_frequency: int = 2
) -> Tuple[nn.Module, Dict]:
    """
    Apply gradual magnitude pruning during training.
    
    This approach slowly increases sparsity during training, allowing the model
    to adapt to the pruning and maintain better accuracy.
    
    Args:
        model: Model to prune and train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        target_sparsity: Final target sparsity
        num_epochs: Number of training epochs
        pruning_frequency: Apply pruning every N epochs
        
    Returns:
        Tuple of (pruned_model, training_stats)
    """
    print(f"🔄 Starting gradual magnitude pruning to {target_sparsity*100:.0f}% sparsity")
    
    model.to(device)
    pruner = UnstructuredPruner(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    training_stats = []
    current_sparsity = 0.0
    
    for epoch in range(num_epochs):
        # Gradually increase sparsity
        if epoch % pruning_frequency == 0 and epoch > 0:
            # Linear sparsity schedule
            current_sparsity = min(target_sparsity, (epoch / num_epochs) * target_sparsity)
            
            if current_sparsity > 0:
                pruner.apply_magnitude_pruning(current_sparsity)
                print(f"   📊 Epoch {epoch}: Applied {current_sparsity*100:.1f}% sparsity")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 100:  # Limit for efficiency
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        scheduler.step()
        
        # Get current sparsity stats
        sparsity_stats = pruner.calculate_sparsity_stats()
        actual_sparsity = sparsity_stats['global_sparsity']
        
        epoch_stats = {
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'sparsity': actual_sparsity,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_stats.append(epoch_stats)
        
        print(f"   Epoch {epoch+1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%, Sparsity {actual_sparsity*100:.1f}%")
    
    # Make pruning permanent
    pruner.make_pruning_permanent()
    
    final_stats = {
        'training_history': training_stats,
        'final_sparsity': pruner.calculate_sparsity_stats(),
        'compression_ratio': pruner.get_compression_ratio()
    }
    
    print(f"✅ Gradual pruning complete. Final sparsity: {final_stats['final_sparsity']['global_sparsity']*100:.1f}%")
    
    return model, final_stats