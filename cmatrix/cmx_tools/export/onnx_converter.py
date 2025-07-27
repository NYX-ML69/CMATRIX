"""
ONNX to CMatrix Converter

Parses ONNX models directly into CMatrix graph IR.
"""

import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Any, Optional
import os

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

def _load_onnx_model(model_path: str) -> onnx.ModelProto:
    """Load ONNX model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

def _extract_initializers(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """Extract weight tensors from ONNX model initializers"""
    weights = {}
    
    for initializer in model.graph.initializer:
        # Convert ONNX tensor to numpy array
        weight_array = onnx.numpy_helper.to_array(initializer)
        weights[initializer.name] = weight_array
    
    return weights

def _extract_graph_nodes(model: onnx.ModelProto) -> Dict[str, CMXOp]:
    """Extract computational nodes from ONNX graph"""
    nodes = {}
    
    for i, node in enumerate(model.graph.node):
        node_id = f"node_{i}_{node.name}" if node.name else f"node_{i}"
        
        # Map ONNX op to CMatrix op
        op_type = _map_onnx_op_to_cmx(node.op_type)
        
        # Get inputs and outputs
        inputs = list(node.input)
        outputs = list(node.output)
        
        # Extract attributes
        attributes = _extract_node_attributes(node)
        
        cmx_op = CMXOp(op_type, inputs, outputs, attributes)
        nodes[node_id] = cmx_op
    
    return nodes

def _map_onnx_op_to_cmx(onnx_op: str) -> str:
    """Map ONNX operation types to CMatrix operation types"""
    op_mapping = {
        'Conv': 'conv2d',
        'Relu': 'relu',
        'MaxPool': 'max_pool2d',
        'AveragePool': 'avg_pool2d',
        'GlobalAveragePool': 'global_avg_pool2d',
        'MatMul': 'matmul',
        'Gemm': 'linear',  # General Matrix Multiplication (used for linear/dense layers)
        'Add': 'add',
        'Mul': 'mul',
        'Sub': 'sub',
        'Div': 'div',
        'Softmax': 'softmax',
        'Sigmoid': 'sigmoid',
        'Tanh': 'tanh',
        'Reshape': 'reshape',
        'Transpose': 'transpose',
        'Concat': 'concat',
        'Split': 'split',
        'Flatten': 'flatten',
        'Dropout': 'dropout',
        'BatchNormalization': 'batch_norm',
        'LayerNormalization': 'layer_norm',
        'InstanceNormalization': 'instance_norm',
        'LeakyRelu': 'leaky_relu',
        'Elu': 'elu',
        'Selu': 'selu',
        'PRelu': 'prelu',
        'Clip': 'clip',
        'Pad': 'pad',
        'Upsample': 'upsample',
        'Resize': 'resize',
        'ReduceMean': 'mean',
        'ReduceSum': 'sum',
        'ReduceMax': 'max',
        'ReduceMin': 'min',
        'Slice': 'slice',
        'Gather': 'gather',
        'Constant': 'constant',
        'Identity': 'identity',
        'Cast': 'cast'
    }
    
    return op_mapping.get(onnx_op, onnx_op.lower())

def _extract_node_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    """Extract attributes from ONNX node"""
    attributes = {}
    
    for attr in node.attribute:
        attr_name = attr.name
        
        # Parse different attribute types
        if attr.type == onnx.AttributeProto.INT:
            attributes[attr_name] = attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            attributes[attr_name] = attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            attributes[attr_name] = attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.INTS:
            attributes[attr_name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attributes[attr_name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRINGS:
            attributes[attr_name] = [s.decode('utf-8') for s in attr.strings]
        elif attr.type == onnx.AttributeProto.TENSOR:
            attributes[attr_name] = onnx.numpy_helper.to_array(attr.t)
        else:
            # Fallback for other types
            attributes[attr_name] = str(attr)
    
    return attributes

def _extract_input_output_info(model: onnx.ModelProto) -> tuple:
    """Extract input and output information from ONNX model"""
    inputs = []
    outputs = []
    
    # Extract input info
    for input_info in model.graph.input:
        input_name = input_info.name
        input_shape = []
        
        if input_info.type.tensor_type.shape:
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value:
                    input_shape.append(dim.dim_value)
                elif dim.dim_param:
                    input_shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    input_shape.append(-1)  # Unknown dimension
        
        inputs.append({
            'name': input_name,
            'shape': input_shape,
            'dtype': _get_onnx_dtype(input_info.type.tensor_type.elem_type)
        })
    
    # Extract output info
    for output_info in model.graph.output:
        output_name = output_info.name
        output_shape = []
        
        if output_info.type.tensor_type.shape:
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.dim_value:
                    output_shape.append(dim.dim_value)
                elif dim.dim_param:
                    output_shape.append(dim.dim_param)
                else:
                    output_shape.append(-1)
        
        outputs.append({
            'name': output_name,
            'shape': output_shape,
            'dtype': _get_onnx_dtype(output_info.type.tensor_type.elem_type)
        })
    
    return inputs, outputs

def _get_onnx_dtype(onnx_type: int) -> str:
    """Convert ONNX data type to string representation"""
    type_mapping = {
        1: 'float32',
        2: 'uint8',
        3: 'int8',
        4: 'uint16',
        5: 'int16',
        6: 'int32',
        7: 'int64',
        8: 'string',
        9: 'bool',
        10: 'float16',
        11: 'float64',
        12: 'uint32',
        13: 'uint64'
    }
    
    return type_mapping.get(onnx_type, f'unknown_type_{onnx_type}')

def _validate_onnx_model(model: onnx.ModelProto) -> bool:
    """Validate ONNX model structure"""
    try:
        # Check if model has required components
        if not model.graph:
            raise ValueError("ONNX model has no graph")
        
        if not model.graph.node:
            raise ValueError("ONNX model has no nodes")
        
        # Check for inputs and outputs
        if not model.graph.input:
            raise ValueError("ONNX model has no inputs defined")
        
        if not model.graph.output:
            raise ValueError("ONNX model has no outputs defined")
        
        return True
    except Exception as e:
        raise RuntimeError(f"ONNX model validation failed: {str(e)}")

def _get_model_metadata(model: onnx.ModelProto, model_path: str) -> Dict[str, Any]:
    """Extract metadata from ONNX model"""
    metadata = {
        'framework': 'onnx',
        'model_path': model_path,
        'producer_name': model.producer_name,
        'producer_version': model.producer_version,
        'model_version': model.model_version,
        'doc_string': model.doc_string,
        'ir_version': model.ir_version,
        'opset_imports': []
    }
    
    # Extract opset information
    for opset in model.opset_import:
        metadata['opset_imports'].append({
            'domain': opset.domain,
            'version': opset.version
        })
    
    # Calculate model size
    try:
        model_size = os.path.getsize(model_path)
        metadata['model_size_bytes'] = model_size
        metadata['model_size_mb'] = model_size / (1024 * 1024)
    except:
        metadata['model_size_bytes'] = 0
        metadata['model_size_mb'] = 0
    
    return metadata

def _optimize_graph(cmx_graph: CMXGraph) -> CMXGraph:
    """Perform basic graph optimizations"""
    # Remove identity operations
    nodes_to_remove = []
    for node_id, node in cmx_graph.nodes.items():
        if node.op_type == 'identity':
            # Connect inputs directly to outputs of identity nodes
            nodes_to_remove.append(node_id)
    
    # Remove identified nodes
    for node_id in nodes_to_remove:
        del cmx_graph.nodes[node_id]
    
    return cmx_graph

def convert_from_onnx(model_path: str, optimize: bool = True) -> CMXGraph:
    """
    Convert ONNX model to CMatrix internal format
    
    Args:
        model_path: Path to ONNX model file
        optimize: Whether to apply basic graph optimizations
        
    Returns:
        CMXGraph: CMatrix internal graph representation
    """
    
    # Load ONNX model
    model = _load_onnx_model(model_path)
    
    # Validate model
    _validate_onnx_model(model)
    
    # Create CMatrix graph
    cmx_graph = CMXGraph()
    
    # Extract components
    cmx_graph.weights = _extract_initializers(model)
    cmx_graph.nodes = _extract_graph_nodes(model)
    
    # Extract input/output information
    inputs, outputs = _extract_input_output_info(model)
    cmx_graph.inputs = inputs
    cmx_graph.outputs = outputs
    
    # Set metadata
    cmx_graph.metadata = _get_model_metadata(model, model_path)
    
    # Add graph statistics
    cmx_graph.metadata.update({
        'num_nodes': len(cmx_graph.nodes),
        'num_weights': len(cmx_graph.weights),
        'num_inputs': len(cmx_graph.inputs),
        'num_outputs': len(cmx_graph.outputs)
    })
    
    # Apply optimizations if requested
    if optimize:
        cmx_graph = _optimize_graph(cmx_graph)
    
    return cmx_graph

def get_onnx_model_info(model_path: str) -> Dict[str, Any]:
    """Get information about an ONNX model without full conversion"""
    model = _load_onnx_model(model_path)
    
    inputs, outputs = _extract_input_output_info(model)
    metadata = _get_model_metadata(model, model_path)
    
    return {
        'inputs': inputs,
        'outputs': outputs,
        'num_nodes': len(model.graph.node),
        'num_initializers': len(model.graph.initializer),
        'metadata': metadata
    }

def validate_onnx_runtime(model_path: str) -> Dict[str, Any]:
    """Validate ONNX model can be loaded by ONNX Runtime"""
    try:
        # Create inference session
        session = ort.InferenceSession(model_path)
        
        # Get input/output info from runtime
        input_info = []
        for inp in session.get_inputs():
            input_info.append({
                'name': inp.name,
                'shape': inp.shape,
                'type': inp.type
            })
        
        output_info = []
        for out in session.get_outputs():
            output_info.append({
                'name': out.name,
                'shape': out.shape,
                'type': out.type
            })
        
        return {
            'valid': True,
            'providers': session.get_providers(),
            'inputs': input_info,
            'outputs': output_info
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def extract_subgraph(model_path: str, start_node: str, end_node: str) -> CMXGraph:
    """Extract a subgraph from ONNX model between specified nodes"""
    model = _load_onnx_model(model_path)
    
    # Find start and end node indices
    start_idx = None
    end_idx = None
    
    for i, node in enumerate(model.graph.node):
        if node.name == start_node or f"node_{i}" == start_node:
            start_idx = i
        if node.name == end_node or f"node_{i}" == end_node:
            end_idx = i
    
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not find start node '{start_node}' or end node '{end_node}'")
    
    if start_idx >= end_idx:
        raise ValueError("Start node must come before end node in the graph")
    
    # Create subgraph
    cmx_graph = CMXGraph()
    
    # Extract nodes in range
    for i in range(start_idx, end_idx + 1):
        node = model.graph.node[i]
        node_id = f"node_{i}_{node.name}" if node.name else f"node_{i}"
        
        op_type = _map_onnx_op_to_cmx(node.op_type)
        inputs = list(node.input)
        outputs = list(node.output)
        attributes = _extract_node_attributes(node)
        
        cmx_op = CMXOp(op_type, inputs, outputs, attributes)
        cmx_graph.nodes[node_id] = cmx_op
    
    # Extract relevant weights
    node_inputs = set()
    for i in range(start_idx, end_idx + 1):
        node_inputs.update(model.graph.node[i].input)
    
    for initializer in model.graph.initializer:
        if initializer.name in node_inputs:
            weight_array = onnx.numpy_helper.to_array(initializer)
            cmx_graph.weights[initializer.name] = weight_array
    
    # Set metadata
    cmx_graph.metadata = {
        'framework': 'onnx_subgraph',
        'original_model': model_path,
        'start_node': start_node,
        'end_node': end_node,
        'num_nodes': end_idx - start_idx + 1
    }
    
    return cmx_graph


