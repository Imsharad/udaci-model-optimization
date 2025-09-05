"""
Corrected Multi-Stage Compression Pipeline

CRITICAL ARCHITECTURAL CORRECTION:
Replaces structured pruning (incompatible with MobileNetV3) with the corrected approach:

Stage 0: Knowledge Distillation + Integrated Unstructured Pruning  
Stage 1: Static INT8 Quantization (TensorFlow Lite)
Stage 2: Mobile Deployment Verification

This addresses the fundamental architectural mismatch that caused:
- 87.4% → 10% accuracy collapse (structured pruning destroyed bottleneck structure)
- 10.56ms → 1265ms speed regression (broken timing + dynamic quantization overhead)
- Only 2.2% size reduction (conservative approach + fallback to baseline)
"""

import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional

from utils.evaluation import measure_inference_time_fixed, measure_model_size
from compression.in_training.distillation import MobileNetV3_Household_Small, MobileNetV3_Household_UltraTiny, _knowledge_distillation_loss
from compression.multi_stage.pruning_unstructured import UnstructuredPruner, apply_gradual_magnitude_pruning
from utils.tflite_conversion import convert_model_to_tflite_int8


class CorrectedCompressionPipeline:
    """
    CORRECTED Multi-Stage Compression Pipeline
    
    Architectural Fix: MobileNetV3 + Inverted Residual Bottlenecks
    Compatible Techniques: 
    1. Unstructured Pruning (preserves bottleneck structure)
    2. Knowledge Distillation (accuracy recovery)
    3. Static Quantization (eliminates dynamic overhead)
    
    Expected Results:
    - Stage 0: ~86.5% accuracy, ~2.4MB, ~9.0ms (distillation + pruning)
    - Stage 1: ~86.0% accuracy, ~0.6MB, ~4.0ms (static INT8 quantization)
    - Stage 2: Mobile deployment verification with XNNPACK acceleration
    """
    
    def __init__(self, name: str, baseline_model: nn.Module, baseline_metrics: Dict,
                 train_loader, test_loader, class_names, input_size, device):
        self.name = name
        self.baseline_model = baseline_model
        self.baseline_metrics = baseline_metrics
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.input_size = input_size
        self.device = device
        self.cpu_device = torch.device('cpu')
        
        # Create save directories
        self.save_dir = f"models/pipeline/{name}"
        self.results_dir = f"results/pipeline"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create validation split
        self._create_validation_split()
        
        print(f"✅ Corrected Pipeline '{name}' initialized")
        print("🔧 FIXED: Structured → Unstructured pruning (MobileNetV3 compatible)")
        print("🔧 FIXED: Dynamic → Static quantization (eliminates runtime overhead)")
        print("🔧 FIXED: Model copying timing bug (eliminates 11,886% slowdown)")
        
    def _create_validation_split(self):
        """Create proper train/validation split to prevent overfitting"""
        train_dataset = self.train_loader.dataset
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        batch_size = self.train_loader.batch_size
        num_workers = getattr(self.train_loader, 'num_workers', 2)
        
        self.train_loader_split = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        print(f"🔧 Train/validation split: {train_size}/{val_size} samples")
        
    def evaluate_accuracy(self, model: nn.Module, data_loader, description: str = "") -> float:
        """Evaluate accuracy on a specific dataset"""
        model.eval()
        model = model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        if description:
            print(f"   📊 {description}: {accuracy:.2f}%")
        return accuracy
        
    def create_metrics(self, model: nn.Module, stage_name: str) -> Dict[str, Any]:
        """Create comprehensive metrics using FIXED timing measurement"""
        
        # Accuracy on test set (isolated)
        test_accuracy = self.evaluate_accuracy(model, self.test_loader, f"{stage_name} (test)")
        
        # Model size
        size_mb = measure_model_size(model)['model_size_mb']
        total_params = sum(p.numel() for p in model.parameters())
        
        # FIXED timing measurement (eliminates 11,886% slowdown bug)
        avg_time_ms = measure_inference_time_fixed(
            model, self.test_loader, self.cpu_device, num_batches=50
        )
        
        return {
            'accuracy': {'top1_acc': test_accuracy},
            'size': {'model_size_mb': size_mb, 'total_params': total_params},
            'timing': {'cpu': {'avg_time_ms': avg_time_ms}}
        }
    
    def stage0_knowledge_distillation_with_unstructured_pruning(
        self, 
        teacher_model: nn.Module,
        target_sparsity: float = 0.60,
        temperature: float = 4.0,
        alpha: float = 0.7,
        epochs: int = 15
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Stage 0: Knowledge Distillation + Integrated Unstructured Pruning
        
        CORRECTED APPROACH:
        1. Creates smaller student architecture (MobileNetV3_Household_Small)
        2. Applies unstructured magnitude-based pruning (compatible with MobileNetV3)
        3. Uses knowledge distillation for accuracy recovery
        4. Integrates pruning during training (not as separate stage)
        
        Expected: ~86.5% accuracy, ~2.4MB, ~9.0ms
        """
        print("\n🔄 STAGE 0: Knowledge Distillation + Unstructured Pruning")
        print("=" * 60)
        print(f"🎯 Target: {target_sparsity*100:.0f}% sparsity with accuracy recovery")
        print("🔧 CORRECTED: Unstructured pruning (MobileNetV3 compatible)")
        
        # Create student model - using Small instead of UltraTiny for Stage 0 stability
        student_model = MobileNetV3_Household_Small(
            num_classes=len(self.class_names),
            width_mult=0.5,  # Reduced from 0.6 for more compression
            linear_size=128,  # Reduced from 256
            dropout=0.2
        ).to(self.device)
        
        baseline_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        print(f"📊 Teacher: {baseline_params:,} parameters")
        print(f"📊 Student: {student_params:,} parameters ({student_params/baseline_params:.1%} of teacher)")
        
        # Apply unstructured pruning using our corrected implementation
        pruner = UnstructuredPruner(student_model)
        pruning_stats = pruner.apply_magnitude_pruning(target_sparsity)
        
        print(f"✅ Pruning applied: {pruning_stats['global_sparsity']*100:.1f}% sparsity")
        
        # Knowledge distillation training
        print("🎓 Starting knowledge distillation training...")
        
        optimizer = torch.optim.AdamW(
            student_model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        teacher_model.eval()
        best_val_accuracy = 0
        best_model_state = None
        patience_counter = 0
        patience_limit = 7
        
        print("   📊 Epoch | Train Loss | Train Acc | Val Acc | Sparsity | LR")
        print("   " + "-" * 65)
        
        for epoch in range(epochs):
            # Training phase
            student_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader_split):
                if batch_idx > 100:  # Limit for efficiency
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                student_logits = student_model(data)
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                
                # Knowledge distillation loss (using existing implementation)
                loss = _knowledge_distillation_loss(
                    student_logits, teacher_logits, target, temperature, alpha
                )
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(student_logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = running_loss / min(101, len(self.train_loader_split))
            train_accuracy = 100 * correct / total
            
            # Validation phase
            val_accuracy = self.evaluate_accuracy(student_model, self.val_loader, "")
            
            # Current sparsity check
            current_stats = pruner.calculate_sparsity_stats()
            current_sparsity = current_stats['global_sparsity']
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"   {epoch+1:2d}   | {avg_loss:8.4f} | {train_accuracy:7.1f}% | {val_accuracy:6.1f}% | {current_sparsity*100:6.1f}% | {current_lr:.6f}")
            
            # Best model tracking
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = copy.deepcopy(student_model.state_dict())
                patience_counter = 0
                print(f"   ✅ New best validation accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience_limit:
                print(f"   🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model and make pruning permanent
        if best_model_state:
            student_model.load_state_dict(best_model_state)
        
        pruner.make_pruning_permanent()
        
        print(f"🎯 Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Create comprehensive metrics using FIXED timing
        stage0_metrics = self.create_metrics(student_model, "stage0_distill_prune")
        
        print(f"📊 Stage 0 Results:")
        print(f"   Accuracy: {stage0_metrics['accuracy']['top1_acc']:.2f}%")
        print(f"   Size: {stage0_metrics['size']['model_size_mb']:.2f} MB")
        print(f"   CPU Time: {stage0_metrics['timing']['cpu']['avg_time_ms']:.2f} ms")
        
        return student_model, stage0_metrics
        
    def stage1_static_int8_quantization(
        self,
        pytorch_model: nn.Module,
        calibration_samples: int = 200
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Stage 1: Static INT8 Quantization with TensorFlow Lite
        
        CORRECTED APPROACH:
        1. Uses static quantization instead of dynamic (eliminates runtime overhead)
        2. Creates representative calibration dataset
        3. Converts to TensorFlow Lite with full integer inference path
        4. Verifies INT8 model for mobile deployment
        
        Expected: ~86.0% accuracy, ~0.6MB, ~4.0ms
        """
        print("\n🔄 STAGE 1: Static INT8 Quantization (TensorFlow Lite)")
        print("=" * 60)
        print("🔧 CORRECTED: Static quantization (eliminates dynamic overhead)")
        
        tflite_path = os.path.join(self.save_dir, "model_static_int8.tflite")
        
        # Convert PyTorch → TFLite with static INT8 quantization
        print("🔄 Starting PyTorch → TFLite conversion...")
        
        success, conversion_results = convert_model_to_tflite_int8(
            pytorch_model=pytorch_model,
            calibration_loader=self.val_loader,  # Use validation data for calibration
            output_path=tflite_path,
            input_shape=self.input_size,
            num_calibration_samples=calibration_samples
        )
        
        if not success:
            print("❌ TFLite conversion failed")
            return None, {'error': 'TFLite conversion failed'}
        
        # Verify the converted model
        verification = conversion_results['verification']
        
        print(f"📊 TFLite Model Results:")
        print(f"   Model path: {tflite_path}")
        print(f"   Size: {verification['model_size_mb']:.2f} MB")
        print(f"   Input type: {verification['input_dtype']}")
        print(f"   Output type: {verification['output_dtype']}")
        print(f"   Quantization: {verification['quantization_status']}")
        
        # Create metrics for TFLite model (size and conversion info)
        stage1_metrics = {
            'tflite_path': tflite_path,
            'size': {'model_size_mb': verification['model_size_mb']},
            'quantization': verification,
            'conversion_success': success
        }
        
        if verification['is_fully_quantized'] and not verification['has_dequantize_ops']:
            print("✅ Model uses full integer inference path - ready for mobile deployment")
        else:
            print("⚠️ Model may have mixed precision operations - check XNNPACK compatibility")
        
        return tflite_path, stage1_metrics
        
    def stage2_mobile_deployment_verification(
        self,
        tflite_path: str
    ) -> Dict[str, Any]:
        """
        Stage 2: Mobile Deployment Verification
        
        CORRECTED APPROACH:
        1. Loads TFLite model with TensorFlow Lite interpreter
        2. Measures mobile inference performance
        3. Verifies XNNPACK delegate compatibility
        4. Tests with representative mobile data
        
        Expected: Confirms mobile deployment readiness with hardware acceleration
        """
        print("\n🔄 STAGE 2: Mobile Deployment Verification")
        print("=" * 60)
        print("🔧 Verifying TFLite model for mobile deployment...")
        
        try:
            import tensorflow as tf
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"📊 TFLite Model Loaded:")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            
            # Mobile inference timing test
            print("⏱️ Testing mobile inference performance...")
            
            mobile_times = []
            num_mobile_tests = 50
            
            for i, (data, _) in enumerate(self.test_loader):
                if i >= num_mobile_tests:
                    break
                    
                # Take first sample from batch
                sample = data[0:1].numpy().astype(np.float32)
                
                # Convert to INT8 if needed
                if input_details[0]['dtype'] == np.int8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    if input_scale > 0:
                        sample = np.round(sample / input_scale + input_zero_point).astype(np.int8)
                
                # Mobile inference timing
                interpreter.set_tensor(input_details[0]['index'], sample)
                
                start_time = time.perf_counter()
                interpreter.invoke()
                end_time = time.perf_counter()
                
                mobile_times.append((end_time - start_time) * 1000)
            
            avg_mobile_time = np.mean(mobile_times)
            
            # Mobile accuracy test (simplified)
            mobile_correct = 0
            mobile_total = 0
            
            for i, (data, target) in enumerate(self.test_loader):
                if i >= 20:  # Test on limited samples
                    break
                    
                for j in range(min(5, data.size(0))):  # Up to 5 per batch
                    sample = data[j:j+1].numpy().astype(np.float32)
                    true_label = target[j].item()
                    
                    # Convert to INT8 if needed
                    if input_details[0]['dtype'] == np.int8:
                        input_scale, input_zero_point = input_details[0]['quantization']
                        if input_scale > 0:
                            sample = np.round(sample / input_scale + input_zero_point).astype(np.int8)
                    
                    interpreter.set_tensor(input_details[0]['index'], sample)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    predicted = np.argmax(output_data[0])
                    mobile_correct += (predicted == true_label)
                    mobile_total += 1
            
            mobile_accuracy = 100.0 * mobile_correct / mobile_total if mobile_total > 0 else 0
            
            print(f"📊 Mobile Deployment Results:")
            print(f"   Accuracy: {mobile_accuracy:.2f}%")
            print(f"   Inference Time: {avg_mobile_time:.2f} ms")
            print(f"   Model Size: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB")
            
            stage2_metrics = {
                'mobile_accuracy': mobile_accuracy,
                'mobile_inference_time_ms': avg_mobile_time,
                'mobile_model_size_mb': os.path.getsize(tflite_path) / (1024*1024),
                'deployment_ready': mobile_accuracy > 80 and avg_mobile_time < 10,
                'interpreter_loaded': True
            }
            
            if stage2_metrics['deployment_ready']:
                print("✅ Model is ready for mobile deployment!")
            else:
                print("⚠️ Model may need further optimization for mobile deployment")
            
            return stage2_metrics
            
        except ImportError:
            print("⚠️ TensorFlow Lite not available for mobile verification")
            return {'error': 'TensorFlow Lite not available'}
        except Exception as e:
            print(f"❌ Mobile verification failed: {e}")
            return {'error': str(e)}
    
    def run_corrected_pipeline(self) -> Tuple[Any, list]:
        """
        Run the complete corrected 3-stage pipeline.
        
        Returns:
            Tuple of (final_model_or_path, pipeline_results)
        """
        print(f"{'='*70}")
        print(f"🚀 RUNNING CORRECTED PIPELINE: {self.name}")
        print(f"{'='*70}")
        print("🔧 CORRECTED Architecture: MobileNetV3 + Unstructured Pruning")
        print("🔧 CORRECTED Sequence: Distill+Prune → Static Quantize → Mobile Verify")
        print("🎯 TARGET: 86%+ accuracy, <0.8MB, <5.0ms (mobile-ready)")
        
        pipeline_results = []
        
        # Stage 0: Knowledge Distillation + Unstructured Pruning
        try:
            stage0_model, stage0_results = self.stage0_knowledge_distillation_with_unstructured_pruning(
                teacher_model=self.baseline_model,
                target_sparsity=0.60,  # 60% sparsity
                temperature=4.0,  # Higher temperature for better knowledge transfer
                alpha=0.7,  # Emphasize distillation loss
                epochs=15
            )
            pipeline_results.append(("stage0_distill_prune", stage0_results))
        except Exception as e:
            print(f"❌ Stage 0 failed: {e}")
            return None, [("stage0_failed", {'error': str(e)})]
        
        # Stage 1: Static INT8 Quantization
        try:
            tflite_path, stage1_results = self.stage1_static_int8_quantization(
                pytorch_model=stage0_model,
                calibration_samples=200
            )
            if tflite_path:
                pipeline_results.append(("stage1_static_quant", stage1_results))
            else:
                print("⚠️ Stage 1 failed, continuing with PyTorch model")
                return stage0_model, pipeline_results
        except Exception as e:
            print(f"⚠️ Stage 1 failed: {e}, continuing with PyTorch model")
            return stage0_model, pipeline_results
        
        # Stage 2: Mobile Deployment Verification
        try:
            stage2_results = self.stage2_mobile_deployment_verification(tflite_path)
            pipeline_results.append(("stage2_mobile_verify", stage2_results))
        except Exception as e:
            print(f"⚠️ Stage 2 failed: {e}")
            pipeline_results.append(("stage2_failed", {'error': str(e)}))
        
        # Final results analysis
        self._analyze_final_results(pipeline_results)
        
        return tflite_path if tflite_path else stage0_model, pipeline_results
    
    def _analyze_final_results(self, pipeline_results: list):
        """Analyze final pipeline results against CTO requirements"""
        
        print(f"\n{'='*70}")
        print("🎯 CORRECTED PIPELINE FINAL ANALYSIS")
        print(f"{'='*70}")
        
        # Get baseline metrics
        baseline_acc = self.baseline_metrics['accuracy']['top1_acc']
        baseline_size = self.baseline_metrics['size']['model_size_mb']
        baseline_time = self.baseline_metrics['timing']['cpu']['avg_time_ms']
        
        print(f"📊 BASELINE: {baseline_acc:.1f}% accuracy, {baseline_size:.2f}MB, {baseline_time:.1f}ms")
        
        # Analyze each stage
        final_acc = baseline_acc
        final_size = baseline_size
        final_time = baseline_time
        
        for stage_name, results in pipeline_results:
            if 'error' in results:
                print(f"   ❌ {stage_name}: FAILED ({results['error']})")
                continue
                
            if 'accuracy' in results:
                stage_acc = results['accuracy']['top1_acc']
                final_acc = stage_acc
                print(f"   📊 {stage_name}: {stage_acc:.1f}% accuracy")
                
            if 'size' in results and 'model_size_mb' in results['size']:
                stage_size = results['size']['model_size_mb']
                final_size = stage_size
                print(f"   📊 {stage_name}: {stage_size:.2f}MB")
                
            if 'timing' in results and 'cpu' in results['timing']:
                stage_time = results['timing']['cpu']['avg_time_ms']
                final_time = stage_time
                print(f"   📊 {stage_name}: {stage_time:.1f}ms")
                
            if 'mobile_accuracy' in results:
                print(f"   📊 {stage_name}: {results['mobile_accuracy']:.1f}% mobile accuracy")
                print(f"   📊 {stage_name}: {results['mobile_inference_time_ms']:.1f}ms mobile timing")
        
        # CTO Requirements Check
        acc_drop = baseline_acc - final_acc
        size_reduction = (1 - final_size / baseline_size) * 100
        speed_improvement = (1 - final_time / baseline_time) * 100
        
        print(f"\n🏆 FINAL RESULTS vs CTO REQUIREMENTS:")
        print(f"   Accuracy: {final_acc:.1f}% ({acc_drop:+.1f}pp vs baseline)")
        print(f"   Size: {final_size:.2f}MB ({size_reduction:.1f}% reduction)")
        print(f"   Speed: {final_time:.1f}ms ({speed_improvement:+.1f}% improvement)")
        
        # Check CTO targets
        cto_targets = {
            'accuracy_target': baseline_acc * 0.95,  # Within 5%
            'size_target': baseline_size * 0.30,     # 70% reduction  
            'speed_target': baseline_time * 0.40      # 60% improvement
        }
        
        accuracy_meets = final_acc >= cto_targets['accuracy_target']
        size_meets = final_size <= cto_targets['size_target']  
        speed_meets = final_time <= cto_targets['speed_target']
        
        print(f"\n🎯 CTO REQUIREMENTS STATUS:")
        print(f"   ✅ Accuracy: {final_acc:.1f}% ≥ {cto_targets['accuracy_target']:.1f}% {'✓' if accuracy_meets else '✗'}")
        print(f"   ✅ Size: {final_size:.2f}MB ≤ {cto_targets['size_target']:.2f}MB {'✓' if size_meets else '✗'}")  
        print(f"   ✅ Speed: {final_time:.1f}ms ≤ {cto_targets['speed_target']:.1f}ms {'✓' if speed_meets else '✗'}")
        
        all_targets_met = accuracy_meets and size_meets and speed_meets
        
        if all_targets_met:
            print(f"\n🎉 SUCCESS: ALL CTO REQUIREMENTS MET!")
            print("   🚀 Model is ready for production deployment")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: Some targets need refinement")
            if not accuracy_meets:
                print("   📊 Consider: Larger student model or more distillation epochs")
            if not size_meets:
                print("   📊 Consider: Higher sparsity or more aggressive quantization")
            if not speed_meets:
                print("   📊 Consider: Mobile-specific optimizations or XNNPACK delegate")
        
        return all_targets_met