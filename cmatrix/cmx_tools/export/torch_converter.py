"""
PyTorch to CMatrix Converter

Converts PyTorch models to CMatrix internal graph representation.
"""

import torch
import torch.onnx
import tempfile
import os
from typing import Dict, List, Any, Tuple
import numpy as np

class CMXGraph:
    """CMatrix internal graph representation"""
    def __init__(self):
        self.nodes = {}
        self.weights = {}
        self.inputs = []
        self.outputs = []
        self.metadata = {}

class CMXOp:
    """CMatrix operation representation"""
    def __init__(self, op_type: str, inputs: List[str], outputs: List[str], 
                 attributes: Dict[str, Any] = None):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}

def _trace_model(model: torch.nn.Module, input_shape: Tuple) -> torch.jit.ScriptModule:
    """Trace PyTorch model using TorchScript"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    try:
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        return traced_model
    except Exception as e:
        raise RuntimeError(f"Failed to trace PyTorch model: {str(e)}")

def _extract_graph_structure(traced_model: torch.jit.ScriptModule) -> CMXGraph:
    """Extract graph structure from traced PyTorch model"""
    cmx_graph = CMXGraph()
    
    # Get the graph from traced model
    graph = traced_model.graph
    
    # Extract nodes
    node_counter = 0
    for node in graph.nodes():
        node_id = f"node_{node_counter}"
        
        # Map PyTorch ops to CMatrix ops
        op_type = _map_torch_op_to_cmx(node.kind())
        
        # Get inputs and outputs
        inputs = [str(inp) for inp in node.inputs()]
        outputs = [str(out) for out in node.outputs()]
        
        # Extract attributes
        attributes = {}
        for attr_name in node.attributeNames():
            attributes[attr_name] = node[attr_name]
        
        cmx_op = CMXOp(op_type, inputs, outputs, attributes)
        cmx_graph.nodes[node_id] = cmx_op
        node_counter += 1
    
    return cmx_graph

def _map_torch_op_to_cmx(torch_op: str) -> str:
    """Map PyTorch operation types to CMatrix operation types"""
    op_mapping = {
        'aten::conv2d': 'conv2d',
        'aten::relu': 'relu',
        'aten::max_pool2d': 'max_pool2d',
        'aten::adaptive_avg_pool2d': 'adaptive_avg_pool2d',
        'aten::linear': 'linear',
        'aten::add': 'add',
        'aten::mul': 'mul',
        'aten::flatten': 'flatten',
        'aten::dropout': 'dropout',
        'aten::batch_norm': 'batch_norm',
        'aten::softmax': 'softmax',
        'aten::sigmoid': 'sigmoid',
        'aten::tanh': 'tanh'
    }
    
    return op_mapping.get(torch_op, torch_op.replace('aten::', ''))

def _extract_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Extract weights and parameters from PyTorch model"""
    weights = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights[name] = param.detach().cpu().numpy()
    
    # Also extract buffers (like BatchNorm running stats)
    for name, buffer in model.named_buffers():
        weights[name] = buffer.detach().cpu().numpy()
    
    return weights

def _via_onnx_export(model: torch.nn.Module, input_shape: Tuple) -> CMXGraph:
    """Convert PyTorch model via ONNX export as fallback"""
    import onnx
    from .onnx_converter import convert_from_onnx
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Create temporary ONNX file
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            tmp_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        # Convert using ONNX converter
        cmx_graph = convert_from_onnx(tmp_path)
        return cmx_graph
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def convert_from_torch(torch_model: torch.nn.Module, 
                      input_shape: Tuple = (1, 3, 224, 224),
                      use_onnx_fallback: bool = False) -> CMXGraph:
    """
    Convert PyTorch model to CMatrix internal format
    
    Args:
        torch_model: PyTorch model to convert
        input_shape: Input tensor shape for tracing
        use_onnx_fallback: Use ONNX export as intermediary if direct conversion fails
        
    Returns:
        CMXGraph: CMatrix internal graph representation
    """
    
    if not isinstance(torch_model, torch.nn.Module):
        raise TypeError("Input must be a PyTorch nn.Module")
    
    try:
        if use_onnx_fallback:
            return _via_onnx_export(torch_model, input_shape)
        
        # Direct conversion via TorchScript tracing
        traced_model = _trace_model(torch_model, input_shape)
        cmx_graph = _extract_graph_structure(traced_model)
        
        # Extract weights
        cmx_graph.weights = _extract_weights(torch_model)
        
        # Set metadata
        cmx_graph.metadata = {
            'framework': 'pytorch',
            'input_shape': input_shape,
            'num_parameters': sum(p.numel() for p in torch_model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in torch_model.parameters()) / (1024 * 1024)
        }
        
        return cmx_graph
        
    except Exception as e:
        if not use_onnx_fallback:
            print(f"Direct conversion failed: {str(e)}")
            print("Falling back to ONNX export...")
            return _via_onnx_export(torch_model, input_shape)
        else:
            raise RuntimeError(f"PyTorch to CMatrix conversion failed: {str(e)}")

def get_model_info(torch_model: torch.nn.Module) -> Dict[str, Any]:
    """Get information about a PyTorch model"""
    total_params = sum(p.numel() for p in torch_model.parameters())
    trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': sum(p.numel() * p.element_size() for p in torch_model.parameters()) / (1024 * 1024),
        'layers': len(list(torch_model.modules())),
        'framework': 'pytorch'
    }
