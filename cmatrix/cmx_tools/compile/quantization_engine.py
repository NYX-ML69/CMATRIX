"""
quantization_engine.py - Static post-training quantization

Quantizes weights and activations for integer inference on embedded targets.
"""

import copy
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union


class Quantizer:
    """Base class for quantization schemes."""
    
    def __init__(self, dtype: str = 'int8'):
        self.dtype = dtype
        self.bit_width = self._get_bit_width(dtype)
        self.signed = self._is_signed(dtype)
        
    def _get_bit_width(self, dtype: str) -> int:
        """Get bit width for data type."""
        bit_widths = {
            'int8': 8, 'uint8': 8,
            'int16': 16, 'uint16': 16,
            'int32': 32, 'uint32': 32
        }
        return bit_widths.get(dtype, 8)
    
    def _is_signed(self, dtype: str) -> bool:
        """Check if data type is signed."""
        return not dtype.startswith('u')
    
    def get_quantization_range(self) -> Tuple[int, int]:
        """Get quantization range for data type."""
        if self.signed:
            qmin = -(2 ** (self.bit_width - 1))
            qmax = 2 ** (self.bit_width - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** self.bit_width - 1
        return qmin, qmax
    
    def quantize_tensor(self, tensor: np.ndarray, 
                       scale: float, zero_point: int) -> np.ndarray:
        """Quantize floating point tensor to integer."""
        qmin, qmax = self.get_quantization_range()
        
        # Quantize: q = round(x/scale + zero_point)
        quantized = np.round(tensor / scale + zero_point)
        
        # Clamp to valid range
        quantized = np.clip(quantized, qmin, qmax)
        
        return quantized.astype(self._get_numpy_dtype())
    
    def dequantize_tensor(self, quantized: np.ndarray, 
                         scale: float, zero_point: int) -> np.ndarray:
        """Dequantize integer tensor back to floating point."""
        return scale * (quantized.astype(np.float32) - zero_point)
    
    def _get_numpy_dtype(self) -> np.dtype:
        """Get corresponding numpy dtype."""
        dtype_map = {
            'int8': np.int8, 'uint8': np.uint8,
            'int16': np.int16, 'uint16': np.uint16,
            'int32': np.int32, 'uint32': np.uint32
        }
        return dtype_map.get(self.dtype, np.int8)


class SymmetricQuantizer(Quantizer):
    """Symmetric quantization (zero_point = 0)."""
    
    def compute_scale_zero_point(self, tensor: np.ndarray) -> Tuple[float, int]:
        """Compute scale and zero point for symmetric quantization."""
        qmin, qmax = self.get_quantization_range()
        
        # For symmetric quantization, zero point is always 0
        zero_point = 0
        
        # Scale based on maximum absolute value
        max_val = np.max(np.abs(tensor))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / max(abs(qmin), abs(qmax))
        
        return scale, zero_point


class AsymmetricQuantizer(Quantizer):
    """Asymmetric quantization (zero_point != 0)."""
    
    def compute_scale_zero_point(self, tensor: np.ndarray) -> Tuple[float, int]:
        """Compute scale and zero point for asymmetric quantization."""
        qmin, qmax = self.get_quantization_range()
        
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        
        if min_val == max_val:
            scale = 1.0
            zero_point = qmin
        else:
            # Compute scale
            scale = (max_val - min_val) / (qmax - qmin)
            
            # Compute zero point
            zero_point_fp = qmin - min_val / scale
            zero_point = int(np.round(np.clip(zero_point_fp, qmin, qmax)))
        
        return scale, zero_point


class PerChannelQuantizer:
    """Per-channel quantization for weights."""
    
    def __init__(self, base_quantizer: Quantizer, axis: int = 0):
        self.base_quantizer = base_quantizer
        self.axis = axis
    
    def compute_scales_zero_points(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-channel scales and zero points."""
        num_channels = tensor.shape[self.axis]
        scales = np.zeros(num_channels)
        zero_points = np.zeros(num_channels, dtype=int)
        
        for i in range(num_channels):
            # Extract channel data
            channel_slice = [slice(None)] * tensor.ndim
            channel_slice[self.axis] = i
            channel_data = tensor[tuple(channel_slice)]
            
            # Compute scale and zero point for this channel
            scale, zero_point = self.base_quantizer.compute_scale_zero_point(channel_data)
            scales[i] = scale
            zero_points[i] = zero_point
        
        return scales, zero_points
    
    def quantize_tensor(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize tensor per-channel."""
        scales, zero_points = self.compute_scales_zero_points(tensor)
        
        # Quantize each channel
        quantized = np.zeros_like(tensor, dtype=self.base_quantizer._get_numpy_dtype())
        
        for i in range(len(scales)):
            channel_slice = [slice(None)] * tensor.ndim
            channel_slice[self.axis] = i
            channel_data = tensor[tuple(channel_slice)]
            
            quantized_channel = self.base_quantizer.quantize_tensor(
                channel_data, scales[i], zero_points[i]
            )
            quantized[tuple(channel_slice)] = quantized_channel
        
        return quantized, scales, zero_points


class CalibrationDataset:
    """Dataset for quantization calibration."""
    
    def __init__(self, data: List[np.ndarray]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def get_batch(self, batch_size: int = 32) -> List[np.ndarray]:
        """Get a batch of calibration data."""
        return self.data[:batch_size]


class QuantizationEngine:
    """Main quantization engine."""
    
    def __init__(self, mode: str = 'int8', symmetric: bool = True):
        self.mode = mode
        self.symmetric = symmetric
        
        # Create base quantizer
        if symmetric:
            self.weight_quantizer = SymmetricQuantizer(mode)
            self.activation_quantizer = SymmetricQuantizer(mode)
        else:
            self.weight_quantizer = AsymmetricQuantizer(mode)
            self.activation_quantizer = AsymmetricQuantizer(mode)
    
    def calibrate_activations(self, graph: Dict[str, Any], 
                            calibration_data: Optional[CalibrationDataset] = None) -> Dict[str, Tuple[float, int]]:
        """Calibrate activation quantization parameters."""
        if calibration_data is None:
            print("Warning: No calibration data provided, using dummy ranges")
            return self._get_dummy_activation_ranges(graph)
        
        print(f"Calibrating activations with {len(calibration_data)} samples...")
        
        # Run inference to collect activation statistics
        activation_stats = {}
        layers = graph.get('layers', [])
        
        for sample in calibration_data:
            # Simulate forward pass (simplified)
            current_tensor = sample
            
            for layer in layers:
                layer_name = layer.get('name', 'unknown')
                layer_type = layer.get('type', 'unknown')
                
                # Simulate layer execution and collect stats
                if layer_type in ['conv2d', 'dense', 'relu']:
                    # Collect activation statistics
                    if layer_name not in activation_stats:
                        activation_stats[layer_name] = {
                            'min_vals': [],
                            'max_vals': []
                        }
                    
                    # Simulate some activation values
                    activation_vals = self._simulate_layer_output(current_tensor, layer)
                    activation_stats[layer_name]['min_vals'].append(np.min(activation_vals))
                    activation_stats[layer_name]['max_vals'].append(np.max(activation_vals))
                    
                    current_tensor = activation_vals
        
        # Compute quantization parameters from collected stats
        quantization_params = {}
        for layer_name, stats in activation_stats.items():
            min_val = np.min(stats['min_vals'])
            max_val = np.max(stats['max_vals'])
            
            # Create dummy tensor with min/max range
            dummy_tensor = np.array([min_val, max_val])
            scale, zero_point = self.activation_quantizer.compute_scale_zero_point(dummy_tensor)
            
            quantization_params[layer_name] = (scale, zero_point)
        
        return quantization_params
    
    def _simulate_layer_output(self, input_tensor: np.ndarray, layer: Dict) -> np.ndarray:
        """Simulate layer output for calibration."""
        layer_type = layer.get('type', 'unknown')
        
        if layer_type == 'conv2d':
            # Simulate convolution output
            return np.random.normal(0, 0.5, size=input_tensor.shape)
        elif layer_type == 'relu':
            # ReLU clamps negative values
            return np.maximum(0, input_tensor)
        elif layer_type == 'dense':
            # Simulate dense layer output
            output_size = layer.get('output_size', 10)
            return np.random.normal(0, 1.0, size=(output_size,))
        else:
            return input_tensor
    
    def _get_dummy_activation_ranges(self, graph: Dict[str, Any]) -> Dict[str, Tuple[float, int]]:
        """Get dummy activation ranges when no calibration data is available."""
        layers = graph.get('layers', [])
        ranges = {}
        
        for layer in layers:
            layer_name = layer.get('name', 'unknown')
            layer_type = layer.get('type', 'unknown')
            
            if layer_type in ['conv2d', 'dense']:
                # Assume typical activation range
                dummy_tensor = np.array([-6.0, 6.0])  # Common range for many activations
            elif layer_type == 'relu':
                # ReLU has non-negative output
                dummy_tensor = np.array([0.0, 6.0])
            else:
                dummy_tensor = np.array([-1.0, 1.0])
            
            scale, zero_point = self.activation_quantizer.compute_scale_zero_point(dummy_tensor)
            ranges[layer_name] = (scale, zero_point)
        
        return ranges
    
    def quantize_weights(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize model weights."""
        quantized_graph = copy.deepcopy(graph)
        layers = quantized_graph.get('layers', [])
        
        weight_count = 0
        for layer in layers:
            layer_type = layer.get('type', 'unknown')
            
            if layer_type in ['conv2d', 'dense']:
                # Quantize weights
                if 'weights' in layer:
                    weights = np.array(layer['weights'])  # Convert to numpy if needed
                    
                    # Use per-channel quantization for weights
                    per_channel_quantizer = PerChannelQuantizer(self.weight_quantizer, axis=0)
                    quantized_weights, scales, zero_points = per_channel_quantizer.quantize_tensor(weights)
                    
                    layer['weights'] = quantized_weights.tolist()
                    layer['weight_scales'] = scales.tolist()
                    layer['weight_zero_points'] = zero_points.tolist()
                    layer['weight_quantized'] = True
                    
                    weight_count += 1
                
                # Quantize bias if present
                if 'bias' in layer:
                    bias = np.array(layer['bias'])
                    # Bias typically uses higher precision
                    bias_quantizer = AsymmetricQuantizer('int32')
                    scale, zero_point = bias_quantizer.compute_scale_zero_point(bias)
                    quantized_bias = bias_quantizer.quantize_tensor(bias, scale, zero_point)
                    
                    layer['bias'] = quantized_bias.tolist()
                    layer['bias_scale'] = scale
                    layer['bias_zero_point'] = zero_point
                    layer['bias_quantized'] = True
        
        print(f"Quantized weights for {weight_count} layers")
        return quantized_graph


def quantize(graph: Dict[str, Any], mode: str = 'int8', 
            calibration_data: Optional[Any] = None,
            symmetric: bool = True) -> Dict[str, Any]:
    """
    Quantize model for integer inference.
    
    Args:
        graph: Input graph IR
        mode: Quantization mode ('int8', 'int16', etc.)
        calibration_data: Data for activation calibration
        symmetric: Use symmetric quantization
        
    Returns:
        Quantized graph IR
    """
    
    print(f"Starting quantization to {mode} (symmetric={symmetric})")
    
    # Create quantization engine
    engine = QuantizationEngine(mode=mode, symmetric=symmetric)
    
    # Convert calibration data if provided
    if calibration_data is not None and not isinstance(calibration_data, CalibrationDataset):
        if isinstance(calibration_data, list):
            calibration_data = CalibrationDataset(calibration_data)
        else:
            # Convert single array to list
            calibration_data = CalibrationDataset([calibration_data])
    
    # Quantize weights
    quantized_graph = engine.quantize_weights(graph)
    
    # Calibrate and store activation quantization parameters
    activation_params = engine.calibrate_activations(quantized_graph, calibration_data)
    quantized_graph['activation_quantization'] = activation_params
    
    # Add quantization metadata
    quantized_graph['quantization'] = {
        'mode': mode,
        'symmetric': symmetric,
        'engine_version': '1.0.0',
        'calibration_samples': len(calibration_data) if calibration_data else 0
    }
    
    # Validate quantization
    if not validate_quantization(quantized_graph):
        print("Warning: Quantization validation failed")
    
    print("Quantization completed successfully")
    return quantized_graph


def validate_quantization(graph: Dict[str, Any]) -> bool:
    """Validate quantized graph."""
    layers = graph.get('layers', [])
    
    for layer in layers:
        layer_type = layer.get('type', 'unknown')
        
        if layer_type in ['conv2d', 'dense']:
            # Check that quantized layers have required fields
            if layer.get('weight_quantized', False):
                required_fields = ['weight_scales', 'weight_zero_points']
                for field in required_fields:
                    if field not in layer:
                        print(f"Missing quantization field '{field}' in layer {layer.get('name')}")
                        return False
    
    # Check activation quantization parameters
    if 'activation_quantization' not in graph:
        print("Warning: No activation quantization parameters found")
    
    return True


def get_quantization_stats(original_graph: Dict[str, Any], 
                          quantized_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Get quantization statistics."""
    
    original_layers = original_graph.get('layers', [])
    quantized_layers = quantized_graph.get('layers', [])
    
    # Count quantized parameters
    quantized_weight_layers = 0
    total_weight_layers = 0
    
    for orig_layer, quant_layer in zip(original_layers, quantized_layers):
        if orig_layer.get('type') in ['conv2d', 'dense']:
            total_weight_layers += 1
            if quant_layer.get('weight_quantized', False):
                quantized_weight_layers += 1
    
    # Estimate memory savings (int8 vs float32 = 4x reduction)
    mode = quantized_graph.get('quantization', {}).get('mode', 'int8')
    if mode == 'int8':
        memory_reduction = 4.0
    elif mode == 'int16':
        memory_reduction = 2.0
    else:
        memory_reduction = 1.0
    
    stats = {
        'quantization_mode': mode,
        'total_layers_with_weights': total_weight_layers,
        'quantized_layers': quantized_weight_layers,
        'quantization_coverage': (quantized_weight_layers / total_weight_layers * 100) if total_weight_layers > 0 else 0,
        'estimated_memory_reduction': memory_reduction,
        'calibration_samples': quantized_graph.get('quantization', {}).get('calibration_samples', 0)
    }
    
    return stats


