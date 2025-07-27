"""
CMatrix Format Validator

Ensures exported graphs conform to valid schema and contain all required fields.
Validates model structure, operations, and compatibility for embedded deployment.
"""

import numpy as np
from typing import Dict, List, Any, Set, Tuple, Optional
import re

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

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class ValidationWarning(Exception):
    """Custom exception for validation warnings"""
    pass

# Supported operations and their requirements
SUPPORTED_OPERATIONS = {
    'conv2d': {
        'required_attributes': ['kernel_size', 'strides'],
        'optional_attributes': ['padding', 'dilation', 'groups'],
        'min_inputs': 2,  # input, weight
        'max_inputs': 3,  # input, weight, bias
        'outputs': 1
    },
    'conv1d': {
        'required_attributes': ['kernel_size', 'strides'],
        'optional_attributes': ['padding', 'dilation', 'groups'],
        'min_inputs': 2,
        'max_inputs': 3,
        'outputs': 1
    },
    'linear': {
        'required_attributes': [],
        'optional_attributes': ['bias'],
        'min_inputs': 2,
        'max_inputs': 3,
        'outputs': 1
    },
    'matmul': {
        'required_attributes': [],
        'optional_attributes': ['transpose_a', 'transpose_b'],
        'min_inputs': 2,
        'max_inputs': 2,
        'outputs': 1
    },
    'relu': {
        'required_attributes': [],
        'optional_attributes': [],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'leaky_relu': {
        'required_attributes': ['alpha'],
        'optional_attributes': [],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'sigmoid': {
        'required_attributes': [],
        'optional_attributes': [],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'tanh': {
        'required_attributes': [],
        'optional_attributes': [],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'softmax': {
        'required_attributes': [],
        'optional_attributes': ['axis'],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'max_pool2d': {
        'required_attributes': ['kernel_size'],
        'optional_attributes': ['strides', 'padding'],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'avg_pool2d': {
        'required_attributes': ['kernel_size'],
        'optional_attributes': ['strides', 'padding'],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'global_avg_pool2d': {
        'required_attributes': [],
        'optional_attributes': [],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'batch_norm': {
        'required_attributes': [],
        'optional_attributes': ['epsilon', 'momentum'],
        'min_inputs': 3,  # input, scale, bias
        'max_inputs': 5,  # input, scale, bias, mean, variance
        'outputs': 1
    },
    'layer_norm': {
        'required_attributes': [],
        'optional_attributes': ['epsilon', 'axis'],
        'min_inputs': 1,
        'max_inputs': 3,
        'outputs': 1
    },
    'dropout': {
        'required_attributes': [],
        'optional_attributes': ['rate', 'training'],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'add': {
        'required_attributes': [],
        'optional_attributes': ['broadcast'],
        'min_inputs': 2,
        'max_inputs': 10,  # Allow multiple inputs for add
        'outputs': 1
    },
    'mul': {
        'required_attributes': [],
        'optional_attributes': ['broadcast'],
        'min_inputs': 2,
        'max_inputs': 2,
        'outputs': 1
    },
    'sub': {
        'required_attributes': [],
        'optional_attributes': ['broadcast'],
        'min_inputs': 2,
        'max_inputs': 2,
        'outputs': 1
    },
    'div': {
        'required_attributes': [],
        'optional_attributes': ['broadcast'],
        'min_inputs': 2,
        'max_inputs': 2,
        'outputs': 1
    },
    'concat': {
        'required_attributes': ['axis'],
        'optional_attributes': [],
        'min_inputs': 2,
        'max_inputs': 20,
        'outputs': 1
    },
    'reshape': {
        'required_attributes': ['shape'],
        'optional_attributes': [],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'transpose': {
        'required_attributes': [],
        'optional_attributes': ['perm'],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    },
    'flatten': {
        'required_attributes': [],
        'optional_attributes': ['axis'],
        'min_inputs': 1,
        'max_inputs': 1,
        'outputs': 1
    }
}

def _validate_basic_structure(cmx_graph: CMXGraph) -> List[str]:
    """Validate basic graph structure"""
    errors = []
    
    # Check if graph has nodes
    if not cmx_graph.nodes:
        errors.append("Graph has no computational nodes")
    
    # Check if graph has inputs
    if not cmx_graph.inputs:
        errors.append("Graph has no defined inputs")
    
    # Check if graph has outputs
    if not cmx_graph.outputs:
        errors.append("Graph has no defined outputs")
    
    # Check metadata
    if not isinstance(cmx_graph.metadata, dict):
        errors.append("Graph metadata must be a dictionary")
    
    return errors

def _validate_node_structure(node_id: str, node: CMXOp) -> List[str]:
    """Validate individual node structure"""
    errors = []
    
    # Check node ID
    if not node_id or not isinstance(node_id, str):
        errors.append(f"Invalid node ID: {node_id}")
    
    # Check op_type
    if not node.op_type or not isinstance(node.op_type, str):
        errors.append(f"Node {node_id}: Missing or invalid op_type")
    
    # Check inputs
    if not isinstance(node.inputs, list):
        errors.append(f"Node {node_id}: Inputs must be a list")
    
    # Check outputs
    if not isinstance(node.outputs, list):
        errors.append(f"Node {node_id}: Outputs must be a list")
    
    # Check attributes
    if not isinstance(node.attributes, dict):
        errors.append(f"Node {node_id}: Attributes must be a dictionary")
    
    return errors

def _validate_supported_operations(cmx_graph: CMXGraph) -> List[str]:
    """Validate that all operations are supported"""
    errors = []
    warnings = []
    
    for node_id, node in cmx_graph.nodes.items():
        op_type = node.op_type
        
        if op_type not in SUPPORTED_OPERATIONS:
            errors.append(f"Node {node_id}: Unsupported operation '{op_type}'")
            continue
        
        op_spec = SUPPORTED_OPERATIONS[op_type]
        
        # Check number of inputs
        num_inputs = len(node.inputs)
        if num_inputs < op_spec['min_inputs']:
            errors.append(f"Node {node_id}: Too few inputs ({num_inputs} < {op_spec['min_inputs']})")
        elif num_inputs > op_spec['max_inputs']:
            errors.append(f"Node {node_id}: Too many inputs ({num_inputs} > {op_spec['max_inputs']})")
        
        # Check number of outputs
        num_outputs = len(node.outputs)
        if num_outputs != op_spec['outputs']:
            errors.append(f"Node {node_id}: Expected {op_spec['outputs']} outputs, got {num_outputs}")
        
        # Check required attributes
        for required_attr in op_spec['required_attributes']:
            if required_attr not in node.attributes:
                errors.append(f"Node {node_id}: Missing required attribute '{required_attr}'")
        
        # Check for unknown attributes
        known_attrs = set(op_spec['required_attributes'] + op_spec['optional_attributes'])
        unknown_attrs = set(node.attributes.keys()) - known_attrs
        if unknown_attrs:
            warnings.extend([f"Node {node_id}: Unknown attribute '{attr}'" for attr in unknown_attrs])
    
    return errors

def _validate_weight_references(cmx_graph: CMXGraph) -> List[str]:
    """Validate that all weight references are valid"""
    errors = []
    
    # Collect all weight references from nodes
    referenced_weights = set()
    for node in cmx_graph.nodes.values():
        for input_name in node.inputs:
            if input_name in cmx_graph.weights:
                referenced_weights.add(input_name)
    
    # Check for missing weights
    available_weights = set(cmx_graph.weights.keys())
    missing_weights = referenced_weights - available_weights
    if missing_weights:
        errors.extend([f"Missing weight: '{weight}'" for weight in missing_weights])
    
    # Check for unused weights (warning, not error)
    unused_weights = available_weights - referenced_weights
    if unused_weights:
        print(f"Warning: Unused weights found: {list(unused_weights)[:5]}")
    
    return errors

def _validate_data_flow(cmx_graph: CMXGraph) -> List[str]:
    """Validate data flow connectivity"""
    errors = []
    
    # Collect all tensor names
    all_outputs = set()
    all_inputs = set()
    
    for node in cmx_graph.nodes.values():
        all_outputs.update(node.outputs)
        all_inputs.update(node.inputs)
    
    # Add graph inputs and weights as valid sources
    graph_input_names = set()
    for inp in cmx_graph.inputs:
        if isinstance(inp, dict):
            graph_input_names.add(inp['name'])
        else:
            graph_input_names.add(str(inp))
    
    weight_names = set(cmx_graph.weights.keys())
    valid_sources = all_outputs | graph_input_names | weight_names
    
    # Check for dangling inputs
    dangling_inputs = all_inputs - valid_sources
    if dangling_inputs:
        errors.extend([f"Dangling input: '{inp}'" for inp in dangling_inputs])
    
    # Check graph outputs are available
    graph_output_names = set()
    for out in cmx_graph.outputs:
        if isinstance(out, dict):
            graph_output_names.add(out['name'])
        else:
            graph_output_names.add(str(out))
    
    missing_outputs = graph_output_names - valid_sources
    if missing_outputs:
        errors.extend([f"Graph output not produced: '{out}'" for out in missing_outputs])
    
    return errors

def _validate_weight_data(cmx_graph: CMXGraph) -> List[str]:
    """Validate weight data integrity"""
    errors = []
    
    for weight_name, weight_data in cmx_graph.weights.items():
        # Check if weight is numpy array
        if not isinstance(weight_data, np.ndarray):
            errors.append(f"Weight '{weight_name}': Must be numpy array, got {type(weight_data)}")
            continue
        
        # Check for empty weights
        if weight_data.size == 0:
            errors.append(f"Weight '{weight_name}': Empty weight tensor")
        
        # Check for invalid values
        if not np.isfinite(weight_data).all():
            errors.append(f"Weight '{weight_name}': Contains non-finite values (NaN/Inf)")
        
        # Check data type
        if weight_data.dtype not in [np.float32, np.float16, np.int32, np.int8, np.uint8]:
            print(f"Warning: Weight '{weight_name}' has unusual dtype: {weight_data.dtype}")
    
    return errors

def _validate_input_output_shapes(cmx_graph: CMXGraph) -> List[str]:
    """Validate input and output shape specifications"""
    errors = []
    
    # Validate inputs
    for i, inp in enumerate(cmx_graph.inputs):
        if isinstance(inp, dict):
            if 'name' not in inp:
                errors.append(f"Input {i}: Missing 'name' field")
            if 'shape' not in inp:
                errors.append(f"Input {i}: Missing 'shape' field")
            elif not isinstance(inp['shape'], (list, tuple)):
                errors.append(f"Input {i}: Shape must be list or tuple")
        elif not isinstance(inp, str):
            errors.append(f"Input {i}: Must be string or dictionary")
    
    # Validate outputs
    for i, out in enumerate(cmx_graph.outputs):
        if isinstance(out, dict):
            if 'name' not in out:
                errors.append(f"Output {i}: Missing 'name' field")
            if 'shape' not in out:
                errors.append(f"Output {i}: Missing 'shape' field")
            elif not isinstance(out['shape'], (list, tuple)):
                errors.append(f"Output {i}: Shape must be list or tuple")
        elif not isinstance(out, str):
            errors.append(f"Output {i}: Must be string or dictionary")
    
    return errors

def _validate_naming_conventions(cmx_graph: CMXGraph) -> List[str]:
    """Validate naming conventions"""
    warnings = []
    
    # Check node naming
    for node_id in cmx_graph.nodes.keys():
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', node_id):
            warnings.append(f"Node ID '{node_id}' doesn't follow naming convention")
    
    # Check weight naming
    for weight_name in cmx_graph.weights.keys():
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_./]*$', weight_name):
            warnings.append(f"Weight name '{weight_name}' doesn't follow naming convention")
    
    return warnings

def _check_embedded_compatibility(cmx_graph: CMXGraph) -> List[str]:
    """Check compatibility for embedded deployment"""
    warnings = []
    
    # Check model size
    total_params = sum(w.size for w in cmx_graph.weights.values() if isinstance(w, np.ndarray))
    total_size_mb = sum(w.nbytes for w in cmx_graph.weights.values() if hasattr(w, 'nbytes')) / (1024 * 1024)
    
    if total_params > 10_000_000:  # 10M parameters
        warnings.append(f"Large model: {total_params:,} parameters may be challenging for embedded deployment")
    
    if total_size_mb > 100:  # 100MB
        warnings.append(f"Large model size: {total_size_mb:.1f}MB may exceed embedded memory constraints")
    
    # Check for dynamic shapes
    for inp in cmx_graph.inputs:
        if isinstance(inp, dict) and 'shape' in inp:
            if any(dim == -1 or isinstance(dim, str) for dim in inp['shape']):
                warnings.append("Dynamic input shapes may not be supported in embedded deployment")
    
    # Check for unsupported operations in embedded context
    embedded_unsupported = {'lstm', 'gru', 'attention', 'transformer'}
    used_ops = {node.op_type for node in cmx_graph.nodes.values()}
    unsupported = used_ops & embedded_unsupported
    if unsupported:
        warnings.extend([f"Operation '{op}' may not be optimized for embedded deployment" for op in unsupported])
    
    return warnings

def validate_model_format(cmx_graph: CMXGraph, 
                         strict: bool = True,
                         check_embedded: bool = False) -> bool:
    """
    Validate CMatrix graph format
    
    Args:
        cmx_graph: CMatrix graph to validate
        strict: Whether to treat warnings as errors
        check_embedded: Whether to check embedded deployment compatibility
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValidationError: If validation fails with errors
        ValidationWarning: If validation fails with warnings in strict mode
    """
    
    all_errors = []
    all_warnings = []
    
    try:
        # Basic structure validation
        all_errors.extend(_validate_basic_structure(cmx_graph))
        
        # Node structure validation
        for node_id, node in cmx_graph.nodes.items():
            all_errors.extend(_validate_node_structure(node_id, node))
        
        # Operation support validation
        all_errors.extend(_validate_supported_operations(cmx_graph))
        
        # Weight reference validation
        all_errors.extend(_validate_weight_references(cmx_graph))
        
        # Data flow validation
        all_errors.extend(_validate_data_flow(cmx_graph))
        
        # Weight data validation
        all_errors.extend(_validate_weight_data(cmx_graph))
        
        # Input/output shape validation
        all_errors.extend(_validate_input_output_shapes(cmx_graph))
        
        # Naming convention validation (warnings only)
        all_warnings.extend(_validate_naming_conventions(cmx_graph))
        
        # Embedded compatibility check
        if check_embedded:
            all_warnings.extend(_check_embedded_compatibility(cmx_graph))
        
        # Report results
        if all_errors:
            error_msg = f"Validation failed with {len(all_errors)} errors:\n" + "\n".join(f"  - {err}" for err in all_errors[:10])
            if len(all_errors) > 10:
                error_msg += f"\n  ... and {len(all_errors) - 10} more errors"
            raise ValidationError(error_msg)
        
        if all_warnings:
            warning_msg = f"Validation completed with {len(all_warnings)} warnings:\n" + "\n".join(f"  - {warn}" for warn in all_warnings[:10])
            if len(all_warnings) > 10:
                warning_msg += f"\n  ... and {len(all_warnings) - 10} more warnings"
            
            if strict:
                raise ValidationWarning(warning_msg)
            else:
                print(f"Warning: {warning_msg}")
        
        return True
        
    except (ValidationError, ValidationWarning):
        raise
    except Exception as e:
        raise ValidationError(f"Validation failed with unexpected error: {str(e)}") from e

def get_validation_report(cmx_graph: CMXGraph, check_embedded: bool = False) -> Dict[str, Any]:
    """
    Get detailed validation report without raising exceptions
    
    Args:
        cmx_graph: CMatrix graph to validate
        check_embedded: Whether to check embedded deployment compatibility
        
    Returns:
        Dictionary with detailed validation results
    """
    
    report = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'summary': {},
        'checks_performed': []
    }
    
    try:
        # Basic structure validation
        report['checks_performed'].append('basic_structure')
        report['errors'].extend(_validate_basic_structure(cmx_graph))
        
        # Node structure validation
        report['checks_performed'].append('node_structure')
        for node_id, node in cmx_graph.nodes.items():
            report['errors'].extend(_validate_node_structure(node_id, node))
        
        # Operation support validation
        report['checks_performed'].append('operation_support')
        report['errors'].extend(_validate_supported_operations(cmx_graph))
        
        # Weight reference validation
        report['checks_performed'].append('weight_references')
        report['errors'].extend(_validate_weight_references(cmx_graph))
        
        # Data flow validation
        report['checks_performed'].append('data_flow')
        report['errors'].extend(_validate_data_flow(cmx_graph))
        
        # Weight data validation
        report['checks_performed'].append('weight_data')
        report['errors'].extend(_validate_weight_data(cmx_graph))
        
        # Input/output shape validation
        report['checks_performed'].append('input_output_shapes')
        report['errors'].extend(_validate_input_output_shapes(cmx_graph))
        
        # Naming convention validation
        report['checks_performed'].append('naming_conventions')
        report['warnings'].extend(_validate_naming_conventions(cmx_graph))
        
        # Embedded compatibility check
        if check_embedded:
            report['checks_performed'].append('embedded_compatibility')
            report['warnings'].extend(_check_embedded_compatibility(cmx_graph))
        
        # Generate summary
        report['summary'] = {
            'num_nodes': len(cmx_graph.nodes),
            'num_weights': len(cmx_graph.weights),
            'num_inputs': len(cmx_graph.inputs),
            'num_outputs': len(cmx_graph.outputs),
            'total_parameters': sum(w.size for w in cmx_graph.weights.values() if isinstance(w, np.ndarray)),
            'total_size_mb': sum(w.nbytes for w in cmx_graph.weights.values() if hasattr(w, 'nbytes')) / (1024 * 1024),
            'operation_types': list(set(node.op_type for node in cmx_graph.nodes.values())),
            'framework': cmx_graph.metadata.get('framework', 'unknown')
        }
        
        # Set validation status
        report['valid'] = len(report['errors']) == 0
        
    except Exception as e:
        report['errors'].append(f"Validation failed with unexpected error: {str(e)}")
    
    return report

def validate_operation_compatibility(op_type: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate compatibility of a specific operation
    
    Args:
        op_type: Operation type to validate
        attributes: Operation attributes
        
    Returns:
        Dictionary with validation results
    """
    
    result = {
        'supported': False,
        'errors': [],
        'warnings': [],
        'missing_attributes': [],
        'unknown_attributes': []
    }
    
    if op_type not in SUPPORTED_OPERATIONS:
        result['errors'].append(f"Unsupported operation: {op_type}")
        return result
    
    result['supported'] = True
    op_spec = SUPPORTED_OPERATIONS[op_type]
    
    # Check required attributes
    for required_attr in op_spec['required_attributes']:
        if required_attr not in attributes:
            result['missing_attributes'].append(required_attr)
            result['errors'].append(f"Missing required attribute: {required_attr}")
    
    # Check for unknown attributes
    known_attrs = set(op_spec['required_attributes'] + op_spec['optional_attributes'])
    for attr_name in attributes.keys():
        if attr_name not in known_attrs:
            result['unknown_attributes'].append(attr_name)
            result['warnings'].append(f"Unknown attribute: {attr_name}")
    
    return result

def get_supported_operations() -> Dict[str, Dict[str, Any]]:
    """Get information about all supported operations"""
    return SUPPORTED_OPERATIONS.copy()

def suggest_fixes(cmx_graph: CMXGraph) -> List[Dict[str, Any]]:
    """
    Suggest fixes for common validation issues
    
    Args:
        cmx_graph: CMatrix graph to analyze
        
    Returns:
        List of suggested fixes
    """
    
    suggestions = []
    
    try:
        # Get validation report
        report = get_validation_report(cmx_graph)
        
        # Analyze errors and suggest fixes
        for error in report['errors']:
            if 'Missing weight' in error:
                weight_name = error.split("'")[1]
                suggestions.append({
                    'issue': error,
                    'suggestion': f"Add weight '{weight_name}' to cmx_graph.weights dictionary",
                    'severity': 'high',
                    'fix_type': 'add_weight'
                })
            
            elif 'Dangling input' in error:
                input_name = error.split("'")[1]
                suggestions.append({
                    'issue': error,
                    'suggestion': f"Either add '{input_name}' as graph input or connect it to a node output",
                    'severity': 'high',
                    'fix_type': 'connect_input'
                })
            
            elif 'Unsupported operation' in error:
                op_type = error.split("'")[1]
                similar_ops = _find_similar_operations(op_type)
                suggestion = f"Replace '{op_type}' with supported operation"
                if similar_ops:
                    suggestion += f". Similar supported ops: {', '.join(similar_ops[:3])}"
                
                suggestions.append({
                    'issue': error,
                    'suggestion': suggestion,
                    'severity': 'high',
                    'fix_type': 'replace_operation',
                    'alternatives': similar_ops
                })
            
            elif 'Missing required attribute' in error:
                attr_name = error.split("'")[1]
                suggestions.append({
                    'issue': error,
                    'suggestion': f"Add required attribute '{attr_name}' to node attributes",
                    'severity': 'high',
                    'fix_type': 'add_attribute'
                })
        
        # Analyze warnings and suggest improvements
        for warning in report['warnings']:
            if 'Large model' in warning:
                suggestions.append({
                    'issue': warning,
                    'suggestion': "Consider model compression techniques like quantization or pruning",
                    'severity': 'medium',
                    'fix_type': 'optimize_model'
                })
            
            elif 'Dynamic input shapes' in warning:
                suggestions.append({
                    'issue': warning,
                    'suggestion': "Define fixed input shapes for embedded deployment",
                    'severity': 'medium',
                    'fix_type': 'fix_input_shapes'
                })
        
    except Exception as e:
        suggestions.append({
            'issue': f"Analysis failed: {str(e)}",
            'suggestion': "Manual inspection required",
            'severity': 'high',
            'fix_type': 'manual_review'
        })
    
    return suggestions

def _find_similar_operations(op_type: str) -> List[str]:
    """Find similar supported operations"""
    supported_ops = list(SUPPORTED_OPERATIONS.keys())
    
    # Simple similarity based on string matching
    similar = []
    op_lower = op_type.lower()
    
    for supported_op in supported_ops:
        if op_lower in supported_op or supported_op in op_lower:
            similar.append(supported_op)
    
    return similar

def auto_fix_graph(cmx_graph: CMXGraph, fix_types: List[str] = None) -> Tuple[CMXGraph, List[str]]:
    """
    Attempt to automatically fix common validation issues
    
    Args:
        cmx_graph: CMatrix graph to fix
        fix_types: List of fix types to apply (None = apply all safe fixes)
        
    Returns:
        Tuple of (fixed_graph, list_of_applied_fixes)
    """
    
    if fix_types is None:
        fix_types = ['remove_unused_weights', 'add_missing_metadata', 'normalize_names']
    
    applied_fixes = []
    
    # Create a copy to avoid modifying original
    import copy
    fixed_graph = copy.deepcopy(cmx_graph)
    
    try:
        # Remove unused weights
        if 'remove_unused_weights' in fix_types:
            referenced_weights = set()
            for node in fixed_graph.nodes.values():
                for inp in node.inputs:
                    if inp in fixed_graph.weights:
                        referenced_weights.add(inp)
            
            unused_weights = set(fixed_graph.weights.keys()) - referenced_weights
            for unused_weight in unused_weights:
                del fixed_graph.weights[unused_weight]
            
            if unused_weights:
                applied_fixes.append(f"Removed {len(unused_weights)} unused weights")
        
        # Add missing metadata
        if 'add_missing_metadata' in fix_types:
            if 'num_nodes' not in fixed_graph.metadata:
                fixed_graph.metadata['num_nodes'] = len(fixed_graph.nodes)
                applied_fixes.append("Added num_nodes to metadata")
            
            if 'num_weights' not in fixed_graph.metadata:
                fixed_graph.metadata['num_weights'] = len(fixed_graph.weights)
                applied_fixes.append("Added num_weights to metadata")
        
        # Normalize names
        if 'normalize_names' in fix_types:
            # Normalize weight names
            normalized_weights = {}
            name_changes = 0
            
            for weight_name, weight_data in fixed_graph.weights.items():
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', weight_name)
                if clean_name != weight_name:
                    name_changes += 1
                normalized_weights[clean_name] = weight_data
            
            if name_changes > 0:
                fixed_graph.weights = normalized_weights
                applied_fixes.append(f"Normalized {name_changes} weight names")
        
    except Exception as e:
        applied_fixes.append(f"Auto-fix failed: {str(e)}")
    
    return fixed_graph, applied_fixes

def validate_batch(graphs: Dict[str, CMXGraph], **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Validate multiple graphs in batch
    
    Args:
        graphs: Dictionary of {name: CMXGraph} pairs
        **kwargs: Arguments passed to get_validation_report
        
    Returns:
        Dictionary of {name: validation_report} pairs
    """
    
    results = {}
    
    for name, graph in graphs.items():
        try:
            results[name] = get_validation_report(graph, **kwargs)
        except Exception as e:
            results[name] = {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'summary': {},
                'checks_performed': []
            }
    
    return results


