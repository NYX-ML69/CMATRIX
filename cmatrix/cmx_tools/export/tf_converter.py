"""
TensorFlow/Keras to CMatrix Converter

Converts TensorFlow and Keras models to CMatrix internal graph representation.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Union, Tuple
import json

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

def _extract_keras_layers(model: tf.keras.Model) -> CMXGraph:
    """Extract layers from Keras model"""
    cmx_graph = CMXGraph()
    
    # Get model configuration
    config = model.get_config()
    
    # Process each layer
    for i, layer in enumerate(model.layers):
        node_id = f"layer_{i}_{layer.name}"
        
        # Map layer type to CMatrix op
        op_type = _map_tf_layer_to_cmx(layer.__class__.__name__)
        
        # Get layer configuration
        layer_config = layer.get_config()
        
        # Extract attributes
        attributes = _extract_layer_attributes(layer, layer_config)
        
        # Determine inputs and outputs
        inputs = [f"input_{i}"] if i == 0 else [f"layer_{i-1}_output"]
        outputs = [f"layer_{i}_output"]
        
        cmx_op = CMXOp(op_type, inputs, outputs, attributes)
        cmx_graph.nodes[node_id] = cmx_op
    
    return cmx_graph

def _extract_concrete_function(model: Union[tf.keras.Model, tf.Module]) -> CMXGraph:
    """Extract graph from TensorFlow concrete function"""
    cmx_graph = CMXGraph()
    
    # Get concrete function
    if isinstance(model, tf.keras.Model):
        # Create dummy input to get concrete function
        dummy_input = tf.random.normal((1,) + model.input_shape[1:])
        concrete_func = model(dummy_input)
        func = model.__call__.get_concrete_function(dummy_input)
    else:
        # For tf.Module or saved model
        func = model.signatures['serving_default']
    
    # Extract graph def
    graph_def = func.graph.as_graph_def()
    
    # Process nodes
    for i, node in enumerate(graph_def.node):
        node_id = f"node_{i}_{node.name}"
        
        # Map TF op to CMatrix op
        op_type = _map_tf_op_to_cmx(node.op)
        
        # Get inputs and outputs
        inputs = list(node.input)
        outputs = [node.name]
        
        # Extract attributes
        attributes = {}
        for attr_name, attr_value in node.attr.items():
            attributes[attr_name] = _parse_tf_attr(attr_value)
        
        cmx_op = CMXOp(op_type, inputs, outputs, attributes)
        cmx_graph.nodes[node_id] = cmx_op
    
    return cmx_graph

def _map_tf_layer_to_cmx(layer_type: str) -> str:
    """Map TensorFlow/Keras layer types to CMatrix operation types"""
    layer_mapping = {
        'Dense': 'linear',
        'Conv2D': 'conv2d',
        'Conv1D': 'conv1d',
        'MaxPooling2D': 'max_pool2d',
        'AveragePooling2D': 'avg_pool2d',
        'GlobalAveragePooling2D': 'global_avg_pool2d',
        'Flatten': 'flatten',
        'Reshape': 'reshape',
        'Dropout': 'dropout',
        'BatchNormalization': 'batch_norm',
        'LayerNormalization': 'layer_norm',
        'ReLU': 'relu',
        'Softmax': 'softmax',
        'Sigmoid': 'sigmoid',
        'Tanh': 'tanh',
        'Add': 'add',
        'Multiply': 'mul',
        'Concatenate': 'concat',
        'LSTM': 'lstm',
        'GRU': 'gru',
        'Embedding': 'embedding'
    }
    
    return layer_mapping.get(layer_type, layer_type.lower())

def _map_tf_op_to_cmx(tf_op: str) -> str:
    """Map TensorFlow operation types to CMatrix operation types"""
    op_mapping = {
        'MatMul': 'matmul',
        'Conv2D': 'conv2d',
        'Relu': 'relu',
        'MaxPool': 'max_pool2d',
        'AvgPool': 'avg_pool2d',
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
        'Pad': 'pad',
        'Mean': 'mean',
        'Sum': 'sum'
    }
    
    return op_mapping.get(tf_op, tf_op.lower())

def _extract_layer_attributes(layer: tf.keras.layers.Layer, config: Dict) -> Dict[str, Any]:
    """Extract attributes from Keras layer"""
    attributes = {}
    
    # Common attributes
    if hasattr(layer, 'kernel_size') and layer.kernel_size is not None:
        attributes['kernel_size'] = layer.kernel_size
    if hasattr(layer, 'strides') and layer.strides is not None:
        attributes['strides'] = layer.strides
    if hasattr(layer, 'padding') and layer.padding is not None:
        attributes['padding'] = layer.padding
    if hasattr(layer, 'activation') and layer.activation is not None:
        attributes['activation'] = layer.activation.__name__
    if hasattr(layer, 'units') and layer.units is not None:
        attributes['units'] = layer.units
    if hasattr(layer, 'filters') and layer.filters is not None:
        attributes['filters'] = layer.filters
    if hasattr(layer, 'rate') and layer.rate is not None:
        attributes['dropout_rate'] = layer.rate
    
    # Add any additional config parameters
    for key, value in config.items():
        if key not in attributes and not key.startswith('_'):
            attributes[key] = value
    
    return attributes

def _parse_tf_attr(attr_value) -> Any:
    """Parse TensorFlow attribute value"""
    if attr_value.HasField('i'):
        return attr_value.i
    elif attr_value.HasField('f'):
        return attr_value.f
    elif attr_value.HasField('s'):
        return attr_value.s.decode('utf-8')
    elif attr_value.HasField('b'):
        return attr_value.b
    elif attr_value.list.i:
        return list(attr_value.list.i)
    elif attr_value.list.f:
        return list(attr_value.list.f)
    elif attr_value.list.s:
        return [s.decode('utf-8') for s in attr_value.list.s]
    else:
        return str(attr_value)

def _extract_weights(model: tf.keras.Model) -> Dict[str, np.ndarray]:
    """Extract weights from TensorFlow/Keras model"""
    weights = {}
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            for i, weight in enumerate(layer_weights):
                weight_name = f"{layer.name}_weight_{i}"
                weights[weight_name] = weight
    
    return weights

def _extract_saved_model_weights(saved_model_path: str) -> Dict[str, np.ndarray]:
    """Extract weights from saved model"""
    weights = {}
    
    # Load the saved model
    loaded_model = tf.saved_model.load(saved_model_path)
    
    # Extract variables
    for variable in loaded_model.variables:
        weights[variable.name] = variable.numpy()
    
    return weights

def convert_from_tf(tf_model: Union[tf.keras.Model, str], 
                   use_concrete_function: bool = False) -> CMXGraph:
    """
    Convert TensorFlow/Keras model to CMatrix internal format
    
    Args:
        tf_model: TensorFlow model (keras.Model) or path to saved model
        use_concrete_function: Use concrete function tracing instead of layer extraction
        
    Returns:
        CMXGraph: CMatrix internal graph representation
    """
    
    if isinstance(tf_model, str):
        # Load saved model
        model = tf.saved_model.load(tf_model)
        cmx_graph = _extract_concrete_function(model)
        cmx_graph.weights = _extract_saved_model_weights(tf_model)
        framework_info = 'tensorflow_saved_model'
    elif isinstance(tf_model, tf.keras.Model):
        if use_concrete_function:
            cmx_graph = _extract_concrete_function(tf_model)
        else:
            cmx_graph = _extract_keras_layers(tf_model)
        
        cmx_graph.weights = _extract_weights(tf_model)
        framework_info = 'tensorflow_keras'
    else:
        raise TypeError("Input must be a tf.keras.Model or path to saved model")
    
    # Set inputs and outputs
    if isinstance(tf_model, tf.keras.Model):
        cmx_graph.inputs = [f"input_shape_{tf_model.input_shape}"]
        cmx_graph.outputs = [f"output_shape_{tf_model.output_shape}"]
        
        # Set metadata
        cmx_graph.metadata = {
            'framework': framework_info,
            'input_shape': tf_model.input_shape,
            'output_shape': tf_model.output_shape,
            'num_parameters': tf_model.count_params(),
            'num_layers': len(tf_model.layers)
        }
    else:
        cmx_graph.metadata = {
            'framework': framework_info
        }
    
    return cmx_graph

def get_model_info(tf_model: Union[tf.keras.Model, str]) -> Dict[str, Any]:
    """Get information about a TensorFlow model"""
    if isinstance(tf_model, str):
        # For saved models, load and inspect
        model = tf.saved_model.load(tf_model)
        return {
            'framework': 'tensorflow_saved_model',
            'signatures': list(model.signatures.keys()) if hasattr(model, 'signatures') else []
        }
    elif isinstance(tf_model, tf.keras.Model):
        return {
            'total_parameters': tf_model.count_params(),
            'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in tf_model.trainable_weights]),
            'input_shape': tf_model.input_shape,
            'output_shape': tf_model.output_shape,
            'num_layers': len(tf_model.layers),
            'framework': 'tensorflow_keras',
            'layer_types': [layer.__class__.__name__ for layer in tf_model.layers]
        }
    else:
        raise TypeError("Input must be a tf.keras.Model or path to saved model")

def convert_from_keras(keras_model: tf.keras.Model) -> CMXGraph:
    """Alias for convert_from_tf for Keras models"""
    return convert_from_tf(keras_model, use_concrete_function=False)

