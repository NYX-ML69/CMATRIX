"""
CMatrix Model Serializer

Serialize and deserialize CMatrix graphs to/from binary or JSON formats
for storage, transmission, and loading in embedded environments.
"""

import json
import pickle
import gzip
import os
import numpy as np
from typing import Dict, Any, Union, Optional
import struct
import hashlib
from datetime import datetime

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
    def __init__(self, op_type: str, inputs: list, outputs: list, attributes: dict = None):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}

class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass

def _numpy_to_dict(arr: np.ndarray) -> Dict[str, Any]:
    """Convert numpy array to serializable dictionary"""
    return {
        'data': arr.tobytes(),
        'dtype': str(arr.dtype),
        'shape': arr.shape
    }

def _dict_to_numpy(data_dict: Dict[str, Any]) -> np.ndarray:
    """Convert dictionary back to numpy array"""
    arr = np.frombuffer(data_dict['data'], dtype=data_dict['dtype'])
    return arr.reshape(data_dict['shape'])

def _graph_to_dict(cmx_graph: CMXGraph) -> Dict[str, Any]:
    """Convert CMXGraph to serializable dictionary"""
    
    # Convert nodes
    nodes_dict = {}
    for node_id, node in cmx_graph.nodes.items():
        nodes_dict[node_id] = {
            'op_type': node.op_type,
            'inputs': node.inputs,
            'outputs': node.outputs,
            'attributes': node.attributes
        }
    
    # Convert weights
    weights_dict = {}
    for weight_name, weight_data in cmx_graph.weights.items():
        if isinstance(weight_data, np.ndarray):
            weights_dict[weight_name] = _numpy_to_dict(weight_data)
        else:
            weights_dict[weight_name] = weight_data
    
    return {
        'nodes': nodes_dict,
        'weights': weights_dict,
        'inputs': cmx_graph.inputs,
        'outputs': cmx_graph.outputs,
        'metadata': cmx_graph.metadata,
        'version': '1.0.0',
        'serialization_timestamp': datetime.now().isoformat()
    }

def _dict_to_graph(data_dict: Dict[str, Any]) -> CMXGraph:
    """Convert dictionary back to CMXGraph"""
    
    cmx_graph = CMXGraph()
    
    # Restore nodes
    for node_id, node_data in data_dict['nodes'].items():
        node = CMXOp(
            op_type=node_data['op_type'],
            inputs=node_data['inputs'],
            outputs=node_data['outputs'],
            attributes=node_data['attributes']
        )
        cmx_graph.nodes[node_id] = node
    
    # Restore weights
    for weight_name, weight_data in data_dict['weights'].items():
        if isinstance(weight_data, dict) and 'data' in weight_data:
            cmx_graph.weights[weight_name] = _dict_to_numpy(weight_data)
        else:
            cmx_graph.weights[weight_name] = weight_data
    
    cmx_graph.inputs = data_dict['inputs']
    cmx_graph.outputs = data_dict['outputs']
    cmx_graph.metadata = data_dict['metadata']
    
    return cmx_graph

def _calculate_checksum(data: bytes) -> str:
    """Calculate MD5 checksum of data"""
    return hashlib.md5(data).hexdigest()

def _create_binary_header(metadata: Dict[str, Any]) -> bytes:
    """Create binary header for CMX format"""
    
    # Magic number for CMX format
    magic = b'CMX\x01'
    
    # Version
    version = struct.pack('<H', 100)  # Version 1.00
    
    # Metadata as JSON
    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_size = struct.pack('<I', len(metadata_json))
    
    return magic + version + metadata_size + metadata_json

def _parse_binary_header(data: bytes) -> tuple:
    """Parse binary header and return metadata and offset"""
    
    # Check magic number
    if data[:4] != b'CMX\x01':
        raise SerializationError("Invalid CMX file format")
    
    # Parse version
    version = struct.unpack('<H', data[4:6])[0]
    
    # Parse metadata size
    metadata_size = struct.unpack('<I', data[6:10])[0]
    
    # Parse metadata
    metadata_json = data[10:10+metadata_size].decode('utf-8')
    metadata = json.loads(metadata_json)
    
    return metadata, 10 + metadata_size

def serialize_model(cmx_graph: CMXGraph, output_path: str, 
                   format_type: str = 'binary', compress: bool = True) -> Dict[str, Any]:
    """
    Serialize CMatrix graph to file
    
    Args:
        cmx_graph: CMatrix graph to serialize
        output_path: Output file path
        format_type: 'binary' or 'json'
        compress: Whether to compress the output (gzip)
        
    Returns:
        Dictionary with serialization info
    """
    
    try:
        # Convert graph to dictionary
        graph_dict = _graph_to_dict(cmx_graph)
        
        # Add serialization metadata
        serialization_info = {
            'format': format_type,
            'compressed': compress,
            'file_size_bytes': 0,
            'checksum': '',
            'serialization_time': datetime.now().isoformat()
        }
        
        if format_type == 'json':
            # JSON serialization
            json_data = json.dumps(graph_dict, indent=2, default=str)
            data_bytes = json_data.encode('utf-8')
            
        elif format_type == 'binary':
            # Binary serialization
            # Create header
            header = _create_binary_header(graph_dict['metadata'])
            
            # Serialize graph data
            graph_data = pickle.dumps(graph_dict)
            
            # Combine header and data
            data_bytes = header + graph_data
            
        else:
            raise SerializationError(f"Unsupported format: {format_type}")
        
        # Calculate checksum
        serialization_info['checksum'] = _calculate_checksum(data_bytes)
        
        # Compress if requested
        if compress:
            data_bytes = gzip.compress(data_bytes)
            output_path += '.gz' if not output_path.endswith('.gz') else ''
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(data_bytes)
        
        # Update serialization info
        serialization_info['file_size_bytes'] = len(data_bytes)
        serialization_info['output_path'] = output_path
        
        return serialization_info
        
    except Exception as e:
        raise SerializationError(f"Serialization failed: {str(e)}") from e

def deserialize_model(file_path: str) -> CMXGraph:
    """
    Deserialize CMatrix graph from file
    
    Args:
        file_path: Path to serialized model file
        
    Returns:
        CMXGraph: Deserialized CMatrix graph
    """
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Read file
        with open(file_path, 'rb') as f:
            data_bytes = f.read()
        
        # Check if compressed
        is_compressed = file_path.endswith('.gz')
        if is_compressed:
            data_bytes = gzip.decompress(data_bytes)
        
        # Determine format
        if data_bytes.startswith(b'CMX\x01'):
            # Binary format
            metadata, offset = _parse_binary_header(data_bytes)
            graph_data = data_bytes[offset:]
            graph_dict = pickle.loads(graph_data)
            
        elif data_bytes.startswith(b'{'):
            # JSON format
            json_data = data_bytes.decode('utf-8')
            graph_dict = json.loads(json_data)
            
        else:
            raise SerializationError("Unknown file format")
        
        # Convert back to CMXGraph
        cmx_graph = _dict_to_graph(graph_dict)
        
        return cmx_graph
        
    except Exception as e:
        raise SerializationError(f"Deserialization failed: {str(e)}") from e

def get_model_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about serialized model without full deserialization
    
    Args:
        file_path: Path to serialized model file
        
    Returns:
        Dictionary with model information
    """
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Get file stats
        file_stats = os.stat(file_path)
        info = {
            'file_path': file_path,
            'file_size_bytes': file_stats.st_size,
            'file_size_mb': file_stats.st_size / (1024 * 1024),
            'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        }
        
        # Read file header
        with open(file_path, 'rb') as f:
            data_bytes = f.read(1024)  # Read first 1KB for header
        
        # Check if compressed
        is_compressed = file_path.endswith('.gz')
        info['compressed'] = is_compressed
        
        if is_compressed:
            try:
                data_bytes = gzip.decompress(data_bytes)
            except:
                # If partial decompression fails, read full file
                with open(file_path, 'rb') as f:
                    data_bytes = gzip.decompress(f.read())
        
        # Parse format and metadata
        if data_bytes.startswith(b'CMX\x01'):
            info['format'] = 'binary'
            try:
                metadata, _ = _parse_binary_header(data_bytes)
                info['metadata'] = metadata
            except:
                info['metadata'] = {}
                
        elif data_bytes.startswith(b'{'):
            info['format'] = 'json'
            try:
                # Try to parse JSON metadata
                json_str = data_bytes.decode('utf-8')
                partial_data = json.loads(json_str)
                info['metadata'] = partial_data.get('metadata', {})
            except:
                info['metadata'] = {}
        else:
            info['format'] = 'unknown'
            info['metadata'] = {}
        
        return info
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'valid': False
        }

def convert_format(input_path: str, output_path: str, 
                  target_format: str, compress: bool = None) -> Dict[str, Any]:
    """
    Convert serialized model between formats
    
    Args:
        input_path: Input model file path
        output_path: Output model file path
        target_format: Target format ('binary' or 'json')
        compress: Whether to compress output (None = auto-detect from extension)
        
    Returns:
        Dictionary with conversion info
    """
    
    try:
        # Load model
        cmx_graph = deserialize_model(input_path)
        
        # Auto-detect compression if not specified
        if compress is None:
            compress = output_path.endswith('.gz')
        
        # Serialize in target format
        result = serialize_model(cmx_graph, output_path, target_format, compress)
        
        # Add conversion info
        input_info = get_model_info(input_path)
        result['conversion'] = {
            'input_format': input_info.get('format', 'unknown'),
            'input_size_mb': input_info.get('file_size_mb', 0),
            'output_format': target_format,
            'output_size_mb': result['file_size_bytes'] / (1024 * 1024),
            'compression_ratio': input_info.get('file_size_bytes', 1) / result['file_size_bytes']
        }
        
        return result
        
    except Exception as e:
        raise SerializationError(f"Format conversion failed: {str(e)}") from e

def validate_serialized_model(file_path: str) -> Dict[str, Any]:
    """
    Validate serialized model file integrity
    
    Args:
        file_path: Path to serialized model file
        
    Returns:
        Dictionary with validation results
    """
    
    validation_result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'file_path': file_path
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            validation_result['errors'].append("File does not exist")
            return validation_result
        
        # Get basic file info
        info = get_model_info(file_path)
        if 'error' in info:
            validation_result['errors'].append(f"Failed to read file info: {info['error']}")
            return validation_result
        
        # Try to deserialize
        try:
            cmx_graph = deserialize_model(file_path)
            validation_result['deserialization_success'] = True
        except Exception as e:
            validation_result['errors'].append(f"Deserialization failed: {str(e)}")
            return validation_result
        
        # Validate graph structure
        if not cmx_graph.nodes:
            validation_result['warnings'].append("Model has no nodes")
        
        if not cmx_graph.weights:
            validation_result['warnings'].append("Model has no weights")
        
        if not cmx_graph.inputs:
            validation_result['warnings'].append("Model has no defined inputs")
        
        if not cmx_graph.outputs:
            validation_result['warnings'].append("Model has no defined outputs")
        
        # Check for common issues
        node_outputs = set()
        node_inputs = set()
        
        for node in cmx_graph.nodes.values():
            if not node.op_type:
                validation_result['errors'].append("Found node with empty op_type")
            
            node_outputs.update(node.outputs)
            node_inputs.update(node.inputs)
        
        # Check for dangling references
        weight_names = set(cmx_graph.weights.keys())
        valid_inputs = node_outputs | weight_names
        
        dangling = node_inputs - valid_inputs
        if dangling:
            validation_result['warnings'].append(f"Found dangling inputs: {list(dangling)[:5]}")
        
        # Validate weights
        for weight_name, weight_data in cmx_graph.weights.items():
            if not isinstance(weight_data, np.ndarray):
                validation_result['warnings'].append(f"Weight '{weight_name}' is not a numpy array")
            elif weight_data.size == 0:
                validation_result['warnings'].append(f"Weight '{weight_name}' is empty")
        
        # If we got here without errors, the model is valid
        if not validation_result['errors']:
            validation_result['valid'] = True
            validation_result['summary'] = {
                'num_nodes': len(cmx_graph.nodes),
                'num_weights': len(cmx_graph.weights),
                'total_parameters': sum(w.size for w in cmx_graph.weights.values() if isinstance(w, np.ndarray)),
                'framework': cmx_graph.metadata.get('framework', 'unknown')
            }
        
        return validation_result
        
    except Exception as e:
        validation_result['errors'].append(f"Validation failed: {str(e)}")
        return validation_result

def batch_serialize(models: Dict[str, CMXGraph], output_dir: str, 
                   format_type: str = 'binary', compress: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Serialize multiple models in batch
    
    Args:
        models: Dictionary of {name: CMXGraph} pairs
        output_dir: Output directory
        format_type: Serialization format
        compress: Whether to compress files
        
    Returns:
        Dictionary of {name: serialization_info} pairs
    """
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for name, cmx_graph in models.items():
        try:
            # Determine file extension
            ext = '.cmx' if format_type == 'binary' else '.json'
            if compress:
                ext += '.gz'
            
            output_path = os.path.join(output_dir, f"{name}{ext}")
            
            # Serialize model
            result = serialize_model(cmx_graph, output_path, format_type, compress)
            results[name] = result
            
            print(f"Serialized {name} -> {output_path}")
            
        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"Failed to serialize {name}: {str(e)}")
    
    return results

def create_model_archive(models: Dict[str, CMXGraph], archive_path: str, 
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a compressed archive containing multiple models
    
    Args:
        models: Dictionary of {name: CMXGraph} pairs
        archive_path: Output archive path
        metadata: Optional archive metadata
        
    Returns:
        Dictionary with archive info
    """
    
    import tempfile
    import shutil
    import tarfile
    
    try:
        archive_info = {
            'archive_path': archive_path,
            'num_models': len(models),
            'created_time': datetime.now().isoformat(),
            'models': {}
        }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Serialize each model to temp directory
            for name, cmx_graph in models.items():
                model_path = os.path.join(temp_dir, f"{name}.cmx")
                result = serialize_model(cmx_graph, model_path, 'binary', False)
                archive_info['models'][name] = {
                    'size_bytes': result['file_size_bytes'],
                    'checksum': result['checksum']
                }
            
            # Add metadata file if provided
            if metadata:
                metadata_path = os.path.join(temp_dir, 'archive_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Create archive
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(temp_dir, arcname='models')
        
        # Get final archive size
        archive_info['archive_size_bytes'] = os.path.getsize(archive_path)
        archive_info['archive_size_mb'] = archive_info['archive_size_bytes'] / (1024 * 1024)
        
        return archive_info
        
    except Exception as e:
        raise SerializationError(f"Failed to create archive: {str(e)}") from e

def extract_model_archive(archive_path: str, output_dir: str) -> Dict[str, CMXGraph]:
    """
    Extract models from compressed archive
    
    Args:
        archive_path: Path to model archive
        output_dir: Directory to extract models
        
    Returns:
        Dictionary of {name: CMXGraph} pairs
    """
    
    import tarfile
    
    try:
        models = {}
        
        # Extract archive
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        
        # Load each model
        models_dir = os.path.join(output_dir, 'models')
        for filename in os.listdir(models_dir):
            if filename.endswith('.cmx'):
                model_name = filename[:-4]  # Remove .cmx extension
                model_path = os.path.join(models_dir, filename)
                
                try:
                    cmx_graph = deserialize_model(model_path)
                    models[model_name] = cmx_graph
                except Exception as e:
                    print(f"Failed to load model {model_name}: {str(e)}")
        
        return models
        
    except Exception as e:
        raise SerializationError(f"Failed to extract archive: {str(e)}") from e

