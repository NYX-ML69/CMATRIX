"""
CMatrix Unified Export Controller

Wraps all format-specific converters and provides a unified interface
for converting models from different frameworks to CMatrix format.
"""

import os
from typing import Union, Dict, Any, Optional, Tuple
from .torch_converter import convert_from_torch
from .tf_converter import convert_from_tf
from .onnx_converter import convert_from_onnx

class CMXGraph:
    """CMatrix internal graph representation"""
    def __init__(self):
        self.nodes = {}
        self.weights = {}
        self.inputs = []
        self.outputs = []
        self.metadata = {}

class ExportError(Exception):
    """Custom exception for export-related errors"""
    pass

def _detect_model_format(model) -> str:
    """Automatically detect the model format"""
    
    # Check if it's a file path
    if isinstance(model, str):
        if os.path.exists(model):
            if model.endswith('.onnx'):
                return 'onnx'
            elif model.endswith('.pb') or 'saved_model' in model:
                return 'tf'
            elif model.endswith('.pth') or model.endswith('.pt'):
                return 'torch'
        else:
            raise FileNotFoundError(f"Model file not found: {model}")
    
    # Check framework-specific types
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'torch'
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        if isinstance(model, (tf.keras.Model, tf.Module)):
            return 'tf'
    except ImportError:
        pass
    
    raise ValueError(f"Could not detect model format for: {type(model)}")

def _validate_export_params(model, format_type: str, **kwargs) -> Dict[str, Any]:
    """Validate export parameters for different formats"""
    
    if format_type == 'torch':
        valid_params = {'input_shape', 'use_onnx_fallback'}
        # Set default input shape if not provided
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = (1, 3, 224, 224)
    
    elif format_type == 'tf':
        valid_params = {'use_concrete_function'}
    
    elif format_type == 'onnx':
        valid_params = {'optimize'}
        # Set default optimization
        if 'optimize' not in kwargs:
            kwargs['optimize'] = True
    
    else:
        raise ExportError(f"Unsupported format: {format_type}")
    
    # Remove invalid parameters
    filtered_params = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return filtered_params

def _sanitize_graph(cmx_graph: CMXGraph) -> CMXGraph:
    """Perform post-processing and sanitization of the CMatrix graph"""
    
    # Validate node connections
    _validate_node_connections(cmx_graph)
    
    # Normalize weight names
    _normalize_weight_names(cmx_graph)
    
    # Add missing metadata
    _add_default_metadata(cmx_graph)
    
    return cmx_graph

def _validate_node_connections(cmx_graph: CMXGraph):
    """Validate that all node inputs/outputs are properly connected"""
    
    all_outputs = set()
    all_inputs = set()
    
    # Collect all inputs and outputs
    for node in cmx_graph.nodes.values():
        all_inputs.update(node.inputs)
        all_outputs.update(node.outputs)
    
    # Check for dangling inputs (not connected to any output)
    weight_names = set(cmx_graph.weights.keys())
    graph_inputs = {inp['name'] if isinstance(inp, dict) else inp for inp in cmx_graph.inputs}
    
    valid_inputs = all_outputs | weight_names | graph_inputs
    
    dangling_inputs = all_inputs - valid_inputs
    if dangling_inputs:
        print(f"Warning: Found dangling inputs: {dangling_inputs}")

def _normalize_weight_names(cmx_graph: CMXGraph):
    """Normalize weight names to follow consistent naming convention"""
    
    normalized_weights = {}
    
    for weight_name, weight_data in cmx_graph.weights.items():
        # Remove framework-specific prefixes/suffixes
        clean_name = weight_name.replace(':', '_').replace('/', '_').replace('.', '_')
        
        # Ensure unique names
        counter = 0
        final_name = clean_name
        while final_name in normalized_weights:
            counter += 1
            final_name = f"{clean_name}_{counter}"
        
        normalized_weights[final_name] = weight_data
    
    cmx_graph.weights = normalized_weights

def _add_default_metadata(cmx_graph: CMXGraph):
    """Add default metadata if missing"""
    
    if 'export_version' not in cmx_graph.metadata:
        cmx_graph.metadata['export_version'] = '1.0.0'
    
    if 'num_nodes' not in cmx_graph.metadata:
        cmx_graph.metadata['num_nodes'] = len(cmx_graph.nodes)
    
    if 'num_weights' not in cmx_graph.metadata:
        cmx_graph.metadata['num_weights'] = len(cmx_graph.weights)
    
    if 'total_weight_size_mb' not in cmx_graph.metadata:
        total_size = sum(w.nbytes for w in cmx_graph.weights.values() if hasattr(w, 'nbytes'))
        cmx_graph.metadata['total_weight_size_mb'] = total_size / (1024 * 1024)

def export_model(model, 
                format_type: Optional[str] = None,
                sanitize: bool = True,
                **kwargs) -> CMXGraph:
    """
    Convert model from any supported format to CMatrix internal format
    
    Args:
        model: Model to convert (PyTorch model, TF model, or path to ONNX file)
        format_type: Format type ('torch', 'tf', 'onnx'). Auto-detected if None
        sanitize: Whether to perform post-processing sanitization
        **kwargs: Format-specific parameters
        
    Returns:
        CMXGraph: CMatrix internal graph representation
        
    Raises:
        ExportError: If conversion fails
    """
    
    try:
        # Auto-detect format if not specified
        if format_type is None:
            format_type = _detect_model_format(model)
        
        # Validate and filter parameters
        filtered_params = _validate_export_params(model, format_type, **kwargs)
        
        # Dispatch to appropriate converter
        if format_type == 'torch':
            cmx_graph = convert_from_torch(model, **filtered_params)
            
        elif format_type == 'tf':
            cmx_graph = convert_from_tf(model, **filtered_params)
            
        elif format_type == 'onnx':
            cmx_graph = convert_from_onnx(model, **filtered_params)
            
        else:
            raise ExportError(f"Unsupported format: {format_type}")
        
        # Post-processing and sanitization
        if sanitize:
            cmx_graph = _sanitize_graph(cmx_graph)
        
        # Add export metadata
        cmx_graph.metadata['export_format'] = format_type
        cmx_graph.metadata['sanitized'] = sanitize
        
        return cmx_graph
        
    except Exception as e:
        raise ExportError(f"Model export failed: {str(e)}") from e

def batch_export(models: Dict[str, Union[str, Any]], 
                output_dir: str = None,
                **kwargs) -> Dict[str, CMXGraph]:
    """
    Export multiple models in batch
    
    Args:
        models: Dictionary of {name: model} pairs
        output_dir: Directory to save exported models (optional)
        **kwargs: Export parameters
        
    Returns:
        Dictionary of {name: CMXGraph} pairs
    """
    
    results = {}
    errors = {}
    
    for name, model in models.items():
        try:
            print(f"Exporting model: {name}")
            cmx_graph = export_model(model, **kwargs)
            results[name] = cmx_graph
            
            # Save if output directory specified
            if output_dir:
                from .model_serializer import serialize_model
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{name}.cmx")
                serialize_model(cmx_graph, output_path)
                print(f"Saved: {output_path}")
                
        except Exception as e:
            errors[name] = str(e)
            print(f"Failed to export {name}: {str(e)}")
    
    if errors:
        print(f"\nExport completed with {len(errors)} errors:")
        for name, error in errors.items():
            print(f"  {name}: {error}")
    
    return results

def get_supported_formats() -> Dict[str, Dict[str, Any]]:
    """Get information about supported model formats"""
    
    formats = {
        'torch': {
            'description': 'PyTorch models (nn.Module)',
            'file_extensions': ['.pth', '.pt'],
            'parameters': {
                'input_shape': 'Input tensor shape for tracing (default: (1,3,224,224))',
                'use_onnx_fallback': 'Use ONNX export if direct conversion fails'
            },
            'supported_ops': ['conv2d', 'linear', 'relu', 'max_pool2d', 'batch_norm', 'dropout']
        },
        'tf': {
            'description': 'TensorFlow/Keras models',
            'file_extensions': ['.pb', '.h5'],
            'parameters': {
                'use_concrete_function': 'Use concrete function tracing instead of layer extraction'
            },
            'supported_ops': ['conv2d', 'dense', 'relu', 'max_pool2d', 'batch_norm', 'dropout']
        },
        'onnx': {
            'description': 'ONNX models',
            'file_extensions': ['.onnx'],
            'parameters': {
                'optimize': 'Apply basic graph optimizations (default: True)'
            },
            'supported_ops': ['Conv', 'Relu', 'MaxPool', 'MatMul', 'Add', 'BatchNormalization']
        }
    }
    
    return formats

def compare_models(models: Dict[str, CMXGraph]) -> Dict[str, Any]:
    """Compare multiple exported models"""
    
    if len(models) < 2:
        raise ValueError("At least 2 models required for comparison")
    
    comparison = {
        'models': list(models.keys()),
        'metrics': {}
    }
    
    # Compare basic metrics
    for metric in ['num_nodes', 'num_weights', 'total_weight_size_mb']:
        values = []
        for name, graph in models.items():
            value = graph.metadata.get(metric, 0)
            values.append((name, value))
        
        comparison['metrics'][metric] = {
            'values': dict(values),
            'min': min(values, key=lambda x: x[1]),
            'max': max(values, key=lambda x: x[1])
        }
    
    # Compare architectures
    op_types = {}
    for name, graph in models.items():
        ops = [node.op_type for node in graph.nodes.values()]
        op_types[name] = set(ops)
    
    # Find common and unique operations
    all_ops = set()
    for ops in op_types.values():
        all_ops.update(ops)
    
    common_ops = all_ops.copy()
    for ops in op_types.values():
        common_ops &= ops
    
    comparison['operations'] = {
        'common': list(common_ops),
        'by_model': {name: list(ops) for name, ops in op_types.items()}
    }
    
    return comparison


