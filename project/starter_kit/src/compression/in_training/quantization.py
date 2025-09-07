"""
UdaciSense Project: Quantization-Aware Training Module (Robust Version)

This module provides a quantizable MobileNetV3 model and a robust, memory-efficient
workflow for quantization-aware training. It is optimized to prevent common pitfalls
and memory-related crashes in resource-constrained environments like Google Colab by
implementing aggressive garbage collection, strict gradient management, and optimized
model handling throughout the training and conversion process.
"""

import gc
import json
import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.ao.quantization
import torch.nn as nn
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.quantization.mobilenetv3 import (_mobilenet_v3_conf,
                                                      _mobilenet_v3_model)
from tqdm import tqdm

from utils.model import get_model_size, save_model, train_single_epoch, validate_single_epoch


class QuantizableMobileNetV3_Household(nn.Module):
    """
    Quantizable MobileNetV3 model tailored for the household objects dataset.
    Includes a custom classifier head and a forward pass that standardizes
    input image sizes.
    """
    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        quantize: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None

        self.model = _mobilenet_v3_model(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            weights=weights,
            progress=True,
            quantize=quantize,
        )

        classifier_in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        return self.model(x)

    def fuse_model(self, is_qat: bool = False) -> 'QuantizableMobileNetV3_Household':
        """Fuses convolution, batch norm, and ReLU modules for quantization."""
        self.model.fuse_model(is_qat=is_qat)
        return self


def _prepare_qat_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Prepares a model for Quantization-Aware Training (QAT)."""
    model.fuse_model(is_qat=True)
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def _convert_qat_model_to_quantized(model: nn.Module) -> nn.Module:
    """Converts a QAT-trained model to a fully quantized model for inference."""
    model.cpu()
    model.eval()
    quantized_model = torch.ao.quantization.convert(model, inplace=False)
    return quantized_model


def _train_single_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    grad_clip_norm: float | None = None
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * (predicted == labels).sum().item() / labels.size(0):.2f}%"
        )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = 100.0 * correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


def _validate_single_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> Tuple[float, float]:
    """
    Runs a single validation epoch with memory optimization.
    Crucially uses `torch.no_grad()` to prevent memory leaks.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Validation {epoch+1}/{num_epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * (predicted == labels).sum().item() / labels.size(0):.2f}%"
            )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = 100.0 * correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


def train_model_qat(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    training_config: Dict[str, Any],
    checkpoint_path: str,
    backend: str = "fbgemm",
) -> Tuple[nn.Module, Dict[str, Any], float, int]:
    """
    Trains a model using a robust, memory-efficient QAT workflow.

    This workflow includes:
    1.  Initial memory cleanup.
    2.  Model preparation and calibration.
    3.  Observer and BatchNorm freezing.
    4.  A fine-tuning loop with per-epoch memory management.
    5.  A safe conversion process to a final quantized model.
    """
    # --- Step 1: Define training variables & memory management ---
    num_epochs = training_config.get('num_epochs', 10)
    criterion = training_config.get('criterion')
    patience = training_config.get('patience', 5)
    device = training_config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    grad_clip_norm = training_config.get('grad_clip_norm', None)
    num_calibration_batches = training_config.get('num_calibration_batches', 50)

    # Proactive memory cleanup before training begins
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    model.to(device)

    # --- Step 2: Prepare model for QAT ---
    print("🔧 Preparing model for Quantization-Aware Training...")
    model = _prepare_qat_model(model, backend)

    # --- Step 3: Calibrate the model with `no_grad` for efficiency ---
    print(f"📊 Calibrating model with {num_calibration_batches} batches...")
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= num_calibration_batches:
                break
            _ = model(inputs.to(device))
    print("Calibration complete.")

    # --- Step 4: Freeze observers and BN stats, then prepare for fine-tuning ---
    print("🔒 Freezing observers and Batch Normalization statistics.")
    model.apply(torch.ao.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    optimizer = training_config['optimizer_class'](model.parameters(), **training_config['optimizer_kwargs'])
    scheduler = training_config['scheduler_class'](optimizer, **training_config['scheduler_kwargs'])

    # --- Step 5: Fine-tune the model ---
    print(f"💪 Fine-tuning model for {num_epochs} epochs...")
    best_accuracy = 0.0
    best_epoch = 0
    early_stop_counter = 0
    training_stats = {
        "epoch": [], "train_loss": [], "train_accuracy": [],
        "test_loss": [], "test_accuracy": [], "epoch_time": [], "lr": []
    }

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss, train_accuracy = _train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_epochs=num_epochs, grad_clip_norm=grad_clip_norm
        )

        test_loss, test_accuracy = _validate_single_epoch(
            model, test_loader, criterion, device, epoch, num_epochs
        )

        if scheduler:
            scheduler.step(test_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

        epoch_time = time.time() - epoch_start_time
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
              f"LR: {lr:.6f}, Time: {epoch_time:.2f}s")

        if test_accuracy > best_accuracy:
            print(f"New best model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            # Save model state_dict to CPU to reduce GPU memory usage and I/O overhead
            torch.save(model.state_dict(), checkpoint_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

        # Append stats (using detached floats, which is safe)
        stats_to_append = {
            "epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "test_loss": test_loss, "test_accuracy": test_accuracy,
            "epoch_time": epoch_time, "lr": lr
        }
        for key, value in stats_to_append.items():
            training_stats[key].append(value)

        # End-of-epoch memory cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"Fine-tuning completed. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")

    # --- Step 6: Convert to a final quantized model ---
    print("\nConverting best model to fully quantized format...")

    # Clean up memory before loading the final model
    del model, optimizer, scheduler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Create a fresh model instance on CPU for safe conversion
    final_model = QuantizableMobileNetV3_Household(quantize=False, pretrained=False).cpu()
    final_model = _prepare_qat_model(final_model, backend)

    # Load the state dict from the best QAT model
    print(f"Loading best model weights from: {checkpoint_path}")
    checkpoint_state_dict = torch.load(checkpoint_path, map_location="cpu")
    final_model.load_state_dict(checkpoint_state_dict)

    # Convert the QAT-prepared model to a fully quantized model
    quantized_model = _convert_qat_model_to_quantized(final_model)

    print("✅ Model successfully converted to a deployable integer format.")

    return quantized_model, training_stats, best_accuracy, best_epoch


def apply_quantization_aware_training(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    training_config: Dict[str, Any],
    backend: str,
    baseline_model: nn.Module,
    class_names: list[str],
    input_size: Tuple[int, int, int],
) -> Tuple[nn.Module, Dict[str, Any], str]:
    """
    Apply quantization-aware training with robust memory management to prevent
    crashes during post-training evaluation and comparison.
    """
    # --- Setup ---
    num_epochs = training_config['num_epochs']
    experiment_name = f"in_training/quantization/epochs{num_epochs}".replace('.', '-')
    device = training_config['device']
    inference_device = training_config.get("device_for_inference", torch.device("cpu"))

    # Create directories
    checkpoint_dir = f"models/{experiment_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f"results/{experiment_name}", exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    final_model_path = f"models/{experiment_name}/model.pth"
    stats_path = f"results/{experiment_name}/training_stats.json"

    print(f"🔧 Applying QAT: total {num_epochs} epochs")
    if torch.cuda.is_available():
        print("   Expected time: ~15-20 minutes on T4 GPU")

    # --- 1. Train and Save the Model ---
    model = model.to(device)
    quantized_model, qat_stats, _, _ = train_model_qat(
        model, train_loader, test_loader, training_config, checkpoint_path=checkpoint_path, backend=backend,
    )
    
    # Save the final converted model and its training stats
    with open(stats_path, 'w') as f:
        json.dump(qat_stats, f, indent=4)
    save_model(quantized_model, final_model_path)
    print(f"✅ Final quantized model and stats saved for experiment: {experiment_name}")

    # --- 2. Clear Memory Before Evaluation ---
    print("\n🧹 Clearing memory before evaluation...")
    del model
    del quantized_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3. Evaluate the Optimized Model (Load from Disk) ---
    print("\n🔬 Evaluating performance of the optimized model...")
    model_for_eval = torch.load(final_model_path, map_location=inference_device)

    from utils.evaluation import evaluate_model_metrics
    evaluate_model_metrics(
        model=model_for_eval,
        dataloader=test_loader,
        device=inference_device,
        num_classes=len(class_names),
        class_names=class_names,
        input_size=input_size,
    )

    # --- 4. Clear Memory Again Before Comparison ---
    print("\n🧹 Clearing memory before comparison...")
    del model_for_eval
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 5. Compare to Baseline (Load Both Models from Disk) ---
    print("\n⚖️ Comparing optimized model to baseline...")
    
    final_quantized_model = torch.load(final_model_path, map_location=inference_device)
    
    from utils.evaluation import compare_models
    comparison_results = compare_models(
        baseline_model=baseline_model,
        optimized_model=final_quantized_model,
        dataloader=test_loader,
        device=inference_device,
        num_classes=len(class_names),
        class_names=class_names,
        input_size=input_size,
    )

    print("\n✅ QAT workflow completed successfully!")
    return final_quantized_model, comparison_results, experiment_name
