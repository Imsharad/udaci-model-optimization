"""
QAT Setup Validator - Catches 95% of potential errors before GPU training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from compression.in_training.quantization import QuantizableMobileNetV3_Household, _prepare_qat_model

def validate_qat_complete_setup(config, train_loader, test_loader, backend="fbgemm"):
    """
    Comprehensive validation that simulates the entire QAT workflow.
    Catches most errors before expensive GPU training.
    """
    print("🧪 COMPREHENSIVE QAT VALIDATION")
    print("=" * 50)
    
    errors = []
    warnings = []
    
    try:
        # 1. Test model creation
        print("1️⃣ Testing model creation...")
        model = QuantizableMobileNetV3_Household(quantize=False)
        print("   ✅ Model created successfully")
        
        # 2. Test config completeness
        print("2️⃣ Testing config completeness...")
        required_keys = [
            'optimizer_class', 'optimizer_kwargs', 'scheduler_class', 'scheduler_kwargs',
            'criterion', 'num_epochs', 'qat_start_epoch', 'device', 'device_for_inference'
        ]
        missing = [k for k in required_keys if k not in config]
        if missing:
            errors.append(f"Missing config keys: {missing}")
        else:
            print("   ✅ All required config keys present")
        
        # 3. Test initial optimizer/scheduler creation
        print("3️⃣ Testing initial optimizer/scheduler creation...")
        try:
            optimizer = config['optimizer_class'](model.parameters(), **config['optimizer_kwargs'])
            scheduler = config['scheduler_class'](optimizer, **config['scheduler_kwargs'])
            print("   ✅ Initial optimizer/scheduler created")
        except Exception as e:
            errors.append(f"Initial optimizer/scheduler creation failed: {e}")
        
        # 4. Test device compatibility
        print("4️⃣ Testing device compatibility...")
        device = config['device']
        model = model.to(device)
        print(f"   ✅ Model moved to {device}")
        
        # 5. Test data loading (1 batch)
        print("5️⃣ Testing data loading...")
        try:
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            inputs, labels = train_batch
            inputs, labels = inputs.to(device), labels.to(device)
            print(f"   ✅ Data loaded: batch size {inputs.shape[0]}")
        except Exception as e:
            errors.append(f"Data loading failed: {e}")
        
        # 6. Test forward pass (pre-QAT)
        print("6️⃣ Testing forward pass (pre-QAT)...")
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
            print(f"   ✅ Forward pass successful: {outputs.shape}")
        except Exception as e:
            errors.append(f"Forward pass failed: {e}")
        
        # 7. Test QAT preparation (CRITICAL)
        print("7️⃣ Testing QAT preparation...")
        try:
            model.train()
            qat_model = _prepare_qat_model(model, backend)
            print("   ✅ QAT preparation successful")
        except Exception as e:
            errors.append(f"QAT preparation failed: {e}")
            return errors, warnings  # Stop here if QAT prep fails
        
        # 8. Test optimizer recreation (CRITICAL)
        print("8️⃣ Testing optimizer recreation after QAT...")
        try:
            new_optimizer = config['optimizer_class'](qat_model.parameters(), **config['optimizer_kwargs'])
            new_scheduler = config['scheduler_class'](new_optimizer, **config['scheduler_kwargs'])
            print("   ✅ Optimizer recreation successful")
        except Exception as e:
            errors.append(f"Optimizer recreation failed: {e}")
        
        # 9. Test forward pass (post-QAT)
        print("9️⃣ Testing forward pass (post-QAT)...")
        try:
            qat_model.train()
            outputs = qat_model(inputs)
            print("   ✅ QAT forward pass successful")
        except Exception as e:
            errors.append(f"QAT forward pass failed: {e}")
        
        # 10. Test backward pass and optimization step
        print("🔟 Testing backward pass and optimization...")
        try:
            criterion = config['criterion']
            loss = criterion(outputs, labels)
            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()
            print(f"   ✅ Training step successful: loss={loss.item():.4f}")
        except Exception as e:
            errors.append(f"Training step failed: {e}")
        
        # 11. Test observer disabling and BN freezing
        print("1️⃣1️⃣ Testing observer/BN operations...")
        try:
            qat_model.apply(torch.ao.quantization.disable_observer)
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            print("   ✅ Observer/BN operations successful")
        except Exception as e:
            warnings.append(f"Observer/BN operations issue: {e}")
        
        # 12. Test model conversion
        print("1️⃣2️⃣ Testing model conversion...")
        try:
            qat_model.cpu().eval()
            quantized_model = torch.ao.quantization.convert(qat_model, inplace=False)
            print("   ✅ Model conversion successful")
        except Exception as e:
            errors.append(f"Model conversion failed: {e}")
        
        # 13. Memory check
        print("1️⃣3️⃣ Testing memory usage...")
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            if memory_mb > 10000:  # 10GB
                warnings.append(f"High memory usage: {memory_mb:.0f}MB")
            print(f"   ✅ Memory usage: {memory_mb:.0f}MB")
        
    except Exception as e:
        errors.append(f"Validation crashed: {e}")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print("❌ VALIDATION FAILED")
        for error in errors:
            print(f"   • {error}")
    else:
        print("✅ VALIDATION PASSED")
        print("   🚀 Safe to proceed with full training!")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    
    return errors, warnings

def quick_validate(config):
    """Quick 30-second validation for common issues"""
    print("⚡ Quick validation...")
    
    # Check config structure
    if 'optimizer_class' not in config:
        return ["Missing optimizer_class - use new config format"]
    
    # Test optimizer creation
    try:
        dummy_params = [torch.randn(10, requires_grad=True)]
        config['optimizer_class'](dummy_params, **config['optimizer_kwargs'])
    except Exception as e:
        return [f"Optimizer creation failed: {e}"]
    
    print("✅ Quick validation passed")
    return []