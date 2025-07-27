"""
Model inspection utilities for analyzing parsed models.

Provides functions to inspect model structure, collect statistics,
and extract information about layers, operations, and tensor dimensions.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Import placeholder for model format handlers
# In practice, these would import from specific parsers
try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def inspect_model(model_path: str) -> dict:
    """
    Inspect a model file and return comprehensive analysis.
    
    Args:
        model_path: Path to model file (.onnx, .pth, .pb, etc.)
    
    Returns:
        dict: Model inspection results containing:
            - basic_info: file size, format, version
            - structure: layers, operations, connections
            - tensors: input/output shapes, parameter counts
            - complexity: estimated MACs, memory usage
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model format is unsupported
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_path = Path(model_path)
    file_extension = model_path.suffix.lower()
    
    # Determine model format and dispatch to appropriate handler
    if file_extension == '.onnx' and HAS_ONNX:
        return _inspect_onnx_model(model_path)
    elif file_extension in ['.pth', '.pt'] and HAS_TORCH:
        return _inspect_torch_model(model_path)
    elif file_extension == '.json':
        return _inspect_json_model(model_path)
    else:
        # Fallback: basic file inspection
        return _inspect_generic_model(model_path)


def get_model_summary(model_info: dict) -> str:
    """
    Generate a human-readable summary from model inspection results.
    
    Args:
        model_info: Dictionary from inspect_model()
    
    Returns:
        str: Formatted summary string
    """
    basic = model_info.get('basic_info', {})
    structure = model_info.get('structure', {})
    tensors = model_info.get('tensors', {})
    
    lines = [
        "=== Model Summary ===",
        f"Format: {basic.get('format', 'unknown')}",
        f"File Size: {basic.get('file_size_mb', 0):.2f} MB",
        f"Total Layers: {structure.get('total_layers', 0)}",
        f"Total Operations: {structure.get('total_ops', 0)}",
        f"Parameters: {tensors.get('total_parameters', 0):,}",
        f"Input Shape: {tensors.get('input_shapes', [])}",
        f"Output Shape: {tensors.get('output_shapes', [])}",
    ]
    
    # Add layer type breakdown
    if 'layer_types' in structure:
        lines.append("\n=== Layer Types ===")
        for layer_type, count in structure['layer_types'].items():
            lines.append(f"{layer_type}: {count}")
    
    # Add complexity metrics if available
    if 'complexity' in model_info:
        complexity = model_info['complexity']
        lines.extend([
            "\n=== Complexity Metrics ===",
            f"Estimated MACs: {complexity.get('macs', 0):,}",
            f"Memory Usage: {complexity.get('memory_mb', 0):.2f} MB",
            f"Depth: {complexity.get('depth', 0)} layers"
        ])
    
    return "\n".join(lines)


def _inspect_onnx_model(model_path: Path) -> dict:
    """Inspect ONNX model format."""
    try:
        model = onnx.load(str(model_path))
        graph = model.graph
        
        # Basic info
        basic_info = {
            'format': 'ONNX',
            'version': getattr(model, 'ir_version', 'unknown'),
            'file_size_mb': model_path.stat().st_size / (1024 * 1024),
            'opset_version': [op.version for op in model.opset_import]
        }
        
        # Structure analysis
        nodes = graph.node
        layer_types = {}
        for node in nodes:
            op_type = node.op_type
            layer_types[op_type] = layer_types.get(op_type, 0) + 1
        
        structure = {
            'total_layers': len(nodes),
            'total_ops': len(nodes),
            'layer_types': layer_types,
            'connections': len(graph.value_info)
        }
        
        # Tensor analysis
        inputs = [(inp.name, _get_tensor_shape(inp)) for inp in graph.input]
        outputs = [(out.name, _get_tensor_shape(out)) for out in graph.output]
        
        # Count parameters
        total_params = 0
        for initializer in graph.initializer:
            param_size = 1
            for dim in initializer.dims:
                param_size *= dim
            total_params += param_size
        
        tensors = {
            'input_shapes': [shape for _, shape in inputs],
            'output_shapes': [shape for _, shape in outputs],
            'input_names': [name for name, _ in inputs],
            'output_names': [name for name, _ in outputs],
            'total_parameters': total_params,
            'initializers': len(graph.initializer)
        }
        
        return {
            'basic_info': basic_info,
            'structure': structure,
            'tensors': tensors,
            'raw_info': {
                'producer_name': model.producer_name,
                'producer_version': model.producer_version,
                'domain': model.domain
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to inspect ONNX model: {e}")
        return _inspect_generic_model(model_path)


def _inspect_torch_model(model_path: Path) -> dict:
    """Inspect PyTorch model format."""
    try:
        # Load model checkpoint
        checkpoint = torch.load(str(model_path), map_location='cpu')
        
        # Basic info
        basic_info = {
            'format': 'PyTorch',
            'file_size_mb': model_path.stat().st_size / (1024 * 1024),
        }
        
        # Analyze state dict if available
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint if isinstance(checkpoint, dict) else {}
        
        # Count parameters and analyze structure
        total_params = 0
        layer_info = {}
        
        for name, tensor in state_dict.items():
            if hasattr(tensor, 'numel'):
                total_params += tensor.numel()
            
            # Extract layer type from parameter name
            layer_name = name.split('.')[0] if '.' in name else name
            if layer_name not in layer_info:
                layer_info[layer_name] = {'params': 0, 'tensors': 0}
            
            layer_info[layer_name]['tensors'] += 1
            if hasattr(tensor, 'numel'):
                layer_info[layer_name]['params'] += tensor.numel()
        
        structure = {
            'total_layers': len(layer_info),
            'total_ops': len(layer_info),  # Approximation
            'layer_info': layer_info
        }
        
        tensors = {
            'total_parameters': total_params,
            'parameter_tensors': len(state_dict),
            'input_shapes': [],  # Not easily extractable from checkpoint
            'output_shapes': []
        }
        
        # Add additional info if present in checkpoint
        if isinstance(checkpoint, dict):
            basic_info.update({
                'epoch': checkpoint.get('epoch', 'unknown'),
                'optimizer': 'present' if 'optimizer' in checkpoint else 'absent',
                'scheduler': 'present' if 'scheduler' in checkpoint else 'absent'
            })
        
        return {
            'basic_info': basic_info,
            'structure': structure,
            'tensors': tensors
        }
        
    except Exception as e:
        logging.error(f"Failed to inspect PyTorch model: {e}")
        return _inspect_generic_model(model_path)


def _inspect_json_model(model_path: Path) -> dict:
    """Inspect JSON model description."""
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        basic_info = {
            'format': 'JSON',
            'file_size_mb': model_path.stat().st_size / (1024 * 1024)
        }
        
        # Try to extract structure information
        structure = {}
        tensors = {}
        
        if 'layers' in model_data:
            layers = model_data['layers']
            structure['total_layers'] = len(layers)
            
            layer_types = {}
            for layer in layers:
                layer_type = layer.get('type', 'unknown')
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
            
            structure['layer_types'] = layer_types
        
        if 'inputs' in model_data:
            tensors['input_shapes'] = [inp.get('shape', []) for inp in model_data['inputs']]
        
        if 'outputs' in model_data:
            tensors['output_shapes'] = [out.get('shape', []) for out in model_data['outputs']]
        
        return {
            'basic_info': basic_info,
            'structure': structure,
            'tensors': tensors,
            'raw_data': model_data
        }
        
    except Exception as e:
        logging.error(f"Failed to inspect JSON model: {e}")
        return _inspect_generic_model(model_path)


def _inspect_generic_model(model_path: Path) -> dict:
    """Fallback inspection for unknown formats."""
    stat_info = model_path.stat()
    
    return {
        'basic_info': {
            'format': 'unknown',
            'file_size_mb': stat_info.st_size / (1024 * 1024),
            'extension': model_path.suffix,
            'modified_time': stat_info.st_mtime
        },
        'structure': {
            'total_layers': 0,
            'total_ops': 0
        },
        'tensors': {
            'total_parameters': 0,
            'input_shapes': [],
            'output_shapes': []
        },
        'error': 'Unsupported model format - basic file info only'
    }


def _get_tensor_shape(tensor_proto) -> List[int]:
    """Extract shape from ONNX tensor prototype."""
    if hasattr(tensor_proto, 'type') and hasattr(tensor_proto.type, 'tensor_type'):
        shape = tensor_proto.type.tensor_type.shape
        return [dim.dim_value if dim.dim_value > 0 else -1 for dim in shape.dim]
    return []


def extract_layer_details(model_info: dict, layer_name: str) -> Optional[dict]:
    """
    Extract detailed information about a specific layer.
    
    Args:
        model_info: Model inspection results
        layer_name: Name of the layer to analyze
    
    Returns:
        dict: Layer details or None if not found
    """
    structure = model_info.get('structure', {})
    
    # Search in layer types
    if 'layer_types' in structure:
        if layer_name in structure['layer_types']:
            return {
                'name': layer_name,
                'count': structure['layer_types'][layer_name],
                'type': 'operation_type'
            }
    
    # Search in layer info (PyTorch style)
    if 'layer_info' in structure:
        if layer_name in structure['layer_info']:
            return {
                'name': layer_name,
                'details': structure['layer_info'][layer_name],
                'type': 'layer_module'
            }
    
    return None


def compare_models(model1_info: dict, model2_info: dict) -> dict:
    """
    Compare two model inspection results.
    
    Args:
        model1_info: First model inspection results
        model2_info: Second model inspection results
    
    Returns:
        dict: Comparison results
    """
    comparison = {
        'size_difference_mb': (
            model2_info.get('basic_info', {}).get('file_size_mb', 0) - 
            model1_info.get('basic_info', {}).get('file_size_mb', 0)
        ),
        'parameter_difference': (
            model2_info.get('tensors', {}).get('total_parameters', 0) - 
            model1_info.get('tensors', {}).get('total_parameters', 0)
        ),
        'layer_difference': (
            model2_info.get('structure', {}).get('total_layers', 0) - 
            model1_info.get('structure', {}).get('total_layers', 0)
        )
    }
    
    # Compare formats
    format1 = model1_info.get('basic_info', {}).get('format', 'unknown')
    format2 = model2_info.get('basic_info', {}).get('format', 'unknown')
    comparison['same_format'] = format1 == format2
    comparison['formats'] = [format1, format2]
    
    return comparison


