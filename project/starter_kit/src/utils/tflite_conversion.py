"""
TensorFlow Lite Conversion Pipeline
PyTorch → ONNX → TensorFlow → TFLite with Static INT8 Quantization

CRITICAL CORRECTION: Implements static quantization instead of dynamic
to eliminate the runtime overhead causing speed regression.
"""

import os
import tempfile
import torch
import torch.onnx
import numpy as np
from typing import Optional, Iterator, Tuple, Dict, Any
import warnings

# Suppress conversion warnings
warnings.filterwarnings('ignore')

try:
    import onnx
    import onnx_tf
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TensorFlow Lite dependencies not available: {e}")
    print("Install with: pip install onnx onnx-tf tensorflow")
    TFLITE_AVAILABLE = False


class TFLiteConverter:
    """
    Handles PyTorch to TensorFlow Lite conversion with static INT8 quantization.
    
    Conversion pipeline:
    PyTorch → ONNX → TensorFlow SavedModel → TFLite (with static quantization)
    
    Key features:
    - Static INT8 quantization with calibration dataset
    - Representative dataset generation from PyTorch DataLoader
    - Full integer inference path validation
    - Model size and accuracy verification
    """
    
    def __init__(self, input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)):
        """
        Initialize converter.
        
        Args:
            input_shape: Model input shape (batch, channels, height, width)
        """
        self.input_shape = input_shape
        self.temp_dir = tempfile.mkdtemp(prefix='tflite_conversion_')
        
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite dependencies not available")
            
    def pytorch_to_onnx(
        self, 
        pytorch_model: torch.nn.Module, 
        onnx_path: str,
        verify: bool = True
    ) -> bool:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            pytorch_model: PyTorch model to convert
            onnx_path: Output path for ONNX model
            verify: Whether to verify the ONNX model
            
        Returns:
            Success status
        """
        try:
            print("🔄 Converting PyTorch → ONNX...")
            
            # Set model to evaluation mode
            pytorch_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(self.input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,  # Compatible with onnx-tf
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            if verify:
                # Verify ONNX model
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("   ✅ ONNX model verified")
            
            print(f"   ✅ ONNX model saved to {onnx_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ PyTorch → ONNX conversion failed: {e}")
            return False
    
    def onnx_to_tensorflow(
        self, 
        onnx_path: str, 
        tf_savedmodel_dir: str
    ) -> bool:
        """
        Convert ONNX model to TensorFlow SavedModel format.
        
        Args:
            onnx_path: Path to ONNX model
            tf_savedmodel_dir: Output directory for TensorFlow SavedModel
            
        Returns:
            Success status
        """
        try:
            print("🔄 Converting ONNX → TensorFlow...")
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            
            # Export as SavedModel
            tf_rep.export_graph(tf_savedmodel_dir)
            
            print(f"   ✅ TensorFlow SavedModel saved to {tf_savedmodel_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ ONNX → TensorFlow conversion failed: {e}")
            return False
    
    def create_representative_dataset(
        self, 
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 200
    ) -> Iterator[list]:
        """
        Create representative dataset for static quantization calibration.
        
        CRITICAL: This dataset determines quantization accuracy.
        Must be representative of inference data distribution.
        
        Args:
            data_loader: PyTorch DataLoader with calibration data
            num_samples: Number of calibration samples
            
        Yields:
            Calibration samples as TensorFlow tensors
        """
        print(f"🔄 Creating representative dataset ({num_samples} samples)...")
        
        sample_count = 0
        
        for batch_data, _ in data_loader:
            if sample_count >= num_samples:
                break
                
            # Process each sample in the batch
            for i in range(batch_data.size(0)):
                if sample_count >= num_samples:
                    break
                    
                # Extract single sample and convert to numpy
                sample = batch_data[i:i+1].numpy().astype(np.float32)
                
                # Yield as list (required by TFLite converter)
                yield [sample]
                sample_count += 1
        
        print(f"   ✅ Generated {sample_count} calibration samples")
    
    def tensorflow_to_tflite_int8(
        self, 
        tf_savedmodel_dir: str,
        tflite_path: str,
        representative_dataset_fn: Iterator[list]
    ) -> bool:
        """
        Convert TensorFlow SavedModel to TFLite with static INT8 quantization.
        
        CRITICAL CORRECTION: Uses static quantization instead of dynamic
        to eliminate runtime overhead.
        
        Args:
            tf_savedmodel_dir: Path to TensorFlow SavedModel
            tflite_path: Output path for TFLite model
            representative_dataset_fn: Representative dataset generator
            
        Returns:
            Success status
        """
        try:
            print("🔄 Converting TensorFlow → TFLite (Static INT8)...")
            
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_savedmodel_dir)
            
            # Enable optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Set representative dataset for static quantization
            converter.representative_dataset = representative_dataset_fn
            
            # Force full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Additional optimizations
            converter.target_spec.supported_types = [tf.int8]
            
            # Convert to TFLite
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Verify the conversion
            model_size_mb = len(tflite_model) / (1024 * 1024)
            print(f"   ✅ TFLite INT8 model saved: {tflite_path}")
            print(f"   📊 Model size: {model_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"   ❌ TensorFlow → TFLite conversion failed: {e}")
            return False
    
    def verify_int8_model(self, tflite_path: str) -> Dict[str, Any]:
        """
        Verify that the TFLite model uses full integer inference path.
        
        Args:
            tflite_path: Path to TFLite model
            
        Returns:
            Verification results
        """
        try:
            print("🔍 Verifying INT8 model...")
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Check data types
            input_dtype = input_details[0]['dtype']
            output_dtype = output_details[0]['dtype']
            
            # Get model size
            model_size_bytes = os.path.getsize(tflite_path)
            model_size_mb = model_size_bytes / (1024 * 1024)
            
            # Check for DEQUANTIZE operations (indicates mixed precision)
            # This requires parsing the model graph (simplified check)
            with open(tflite_path, 'rb') as f:
                model_content = f.read()
                has_dequantize = b'DEQUANTIZE' in model_content
            
            verification_results = {
                'input_dtype': str(input_dtype),
                'output_dtype': str(output_dtype),
                'model_size_mb': model_size_mb,
                'model_size_bytes': model_size_bytes,
                'input_shape': input_details[0]['shape'].tolist(),
                'output_shape': output_details[0]['shape'].tolist(),
                'is_fully_quantized': input_dtype == np.int8 and output_dtype == np.int8,
                'has_dequantize_ops': has_dequantize,
                'quantization_status': 'FULL_INT8' if not has_dequantize else 'MIXED_PRECISION'
            }
            
            print(f"   📊 Input type: {input_dtype}, Output type: {output_dtype}")
            print(f"   📊 Model size: {model_size_mb:.2f} MB")
            print(f"   📊 Quantization: {verification_results['quantization_status']}")
            
            if verification_results['is_fully_quantized'] and not has_dequantize:
                print("   ✅ Model uses full integer inference path")
            else:
                print("   ⚠️ Model may have mixed precision operations")
            
            return verification_results
            
        except Exception as e:
            print(f"   ❌ Model verification failed: {e}")
            return {'error': str(e)}
    
    def convert_pytorch_to_tflite(
        self,
        pytorch_model: torch.nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        output_path: str,
        num_calibration_samples: int = 200
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Complete conversion pipeline: PyTorch → TFLite INT8.
        
        Args:
            pytorch_model: PyTorch model to convert
            calibration_loader: DataLoader for quantization calibration
            output_path: Output path for TFLite model
            num_calibration_samples: Number of samples for calibration
            
        Returns:
            Tuple of (success, conversion_results)
        """
        print(f"🚀 Starting complete PyTorch → TFLite INT8 conversion")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Calibration samples: {num_calibration_samples}")
        print(f"   Output path: {output_path}")
        
        try:
            # Step 1: PyTorch → ONNX
            onnx_path = os.path.join(self.temp_dir, "model.onnx")
            if not self.pytorch_to_onnx(pytorch_model, onnx_path):
                return False, {'error': 'ONNX conversion failed'}
            
            # Step 2: ONNX → TensorFlow
            tf_dir = os.path.join(self.temp_dir, "tf_model")
            if not self.onnx_to_tensorflow(onnx_path, tf_dir):
                return False, {'error': 'TensorFlow conversion failed'}
            
            # Step 3: Create representative dataset
            representative_dataset = self.create_representative_dataset(
                calibration_loader, num_calibration_samples
            )
            
            # Step 4: TensorFlow → TFLite INT8
            if not self.tensorflow_to_tflite_int8(tf_dir, output_path, representative_dataset):
                return False, {'error': 'TFLite conversion failed'}
            
            # Step 5: Verify the final model
            verification_results = self.verify_int8_model(output_path)
            
            conversion_results = {
                'success': True,
                'output_path': output_path,
                'temp_dir': self.temp_dir,
                'verification': verification_results
            }
            
            print("✅ Complete conversion pipeline successful!")
            
            return True, conversion_results
            
        except Exception as e:
            print(f"❌ Conversion pipeline failed: {e}")
            return False, {'error': str(e)}
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")


def convert_model_to_tflite_int8(
    pytorch_model: torch.nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    num_calibration_samples: int = 200
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function for PyTorch → TFLite INT8 conversion.
    
    Args:
        pytorch_model: PyTorch model to convert
        calibration_loader: DataLoader for quantization calibration
        output_path: Output path for TFLite model
        input_shape: Model input shape
        num_calibration_samples: Number of calibration samples
        
    Returns:
        Tuple of (success, conversion_results)
    """
    converter = TFLiteConverter(input_shape)
    
    try:
        success, results = converter.convert_pytorch_to_tflite(
            pytorch_model=pytorch_model,
            calibration_loader=calibration_loader,
            output_path=output_path,
            num_calibration_samples=num_calibration_samples
        )
        return success, results
    finally:
        converter.cleanup()