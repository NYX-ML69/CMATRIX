"""
Model analysis utilities for complexity estimation and optimization suggestions.

Provides functions to analyze model complexity including MACs computation,
memory usage estimation, and optimization level recommendations.
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict


def analyze_model(graph_dict: dict) -> dict:
    """
    Analyze model complexity and suggest optimization strategies.
    
    Args:
        graph_dict: Dictionary containing model graph structure with:
            - nodes: list of operation nodes
            - edges: list of connections
            - tensors: tensor information
    
    Returns:
        dict: Analysis results containing:
            - complexity: MACs, memory, depth metrics
            - bottlenecks: identified performance bottlenecks
            - recommendations: optimization suggestions
    """
    try:
        # Extract basic graph information
        nodes = graph_dict.get('nodes', [])
        edges = graph_dict.get('edges', [])
        tensors = graph_dict.get('tensors', {})
        
        # Compute complexity metrics
        complexity = _compute_complexity_metrics(nodes, tensors)
        
        # Identify bottlenecks
        bottlenecks = _identify_bottlenecks(nodes, complexity)
        
        # Generate optimization recommendations
        recommendations = _generate_recommendations(complexity, bottlenecks, nodes)
        
        # Compute memory analysis
        memory_analysis = _analyze_memory_usage(nodes, tensors)
        
        # Analyze computational graph properties
        graph_properties = _analyze_graph_properties(nodes, edges)
        
        return {
            'complexity': complexity,
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'memory_analysis': memory_analysis,
            'graph_properties': graph_properties,
            'optimization_score': _compute_optimization_score(complexity, bottlenecks)
        }
        
    except Exception as e:
        logging.error(f"Model analysis failed: {e}")
        return {
            'error': str(e),
            'complexity': {'total_macs': 0, 'memory_mb': 0, 'depth': 0},
            'bottlenecks': [],
            'recommendations': []
        }


def estimate_complexity(model_structure: dict) -> dict:
    """
    Estimate computational complexity for a model structure.
    
    Args:
        model_structure: Model structure from inspector or parser
    
    Returns:
        dict: Complexity estimation with MACs, memory, and timing
    """
    structure_info = model_structure.get('structure', {})
    tensor_info = model_structure.get('tensors', {})
    
    # Estimate based on layer types if available
    layer_types = structure_info.get('layer_types', {})
    total_params = tensor_info.get('total_parameters', 0)
    
    estimated_macs = 0
    estimated_memory_mb = 0
    
    # MAC estimation based on common layer types
    mac_estimates = {
        'Conv': lambda: _estimate_conv_macs(layer_types.get('Conv', 0)),
        'MatMul': lambda: _estimate_matmul_macs(layer_types.get('MatMul', 0)),
        'Gemm': lambda: _estimate_matmul_macs(layer_types.get('Gemm', 0)),
        'BatchNormalization': lambda: total_params * 0.1,  # Minimal computation
        'Relu': lambda: total_params * 0.01,  # Very lightweight
        'Add': lambda: total_params * 0.05,
        'Mul': lambda: total_params * 0.05,
    }
    
    for layer_type, count in layer_types.items():
        if layer_type in mac_estimates:
            estimated_macs += mac_estimates[layer_type]() * count
        else:
            # Default estimation for unknown layers
            estimated_macs += total_params * 0.1 * count
    
    # Memory estimation (parameters + activations)
    estimated_memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    estimated_memory_mb += estimated_memory_mb * 0.5  # Activation memory approximation
    
    return {
        'estimated_macs': int(estimated_macs),
        'estimated_memory_mb': estimated_memory_mb,
        'parameter_memory_mb': (total_params * 4) / (1024 * 1024),
        'complexity_score': _compute_complexity_score(estimated_macs, estimated_memory_mb),
        'estimated_inference_time_ms': _estimate_inference_time(estimated_macs, estimated_memory_mb)
    }


def _compute_complexity_metrics(nodes: List[dict], tensors: dict) -> dict:
    """Compute detailed complexity metrics for the model."""
    total_macs = 0
    total_memory = 0
    depth = 0
    parallel_ops = 0
    
    # Analyze each node
    for node in nodes:
        op_type = node.get('op_type', node.get('type', 'unknown'))
        attributes = node.get('attributes', {})
        input_shapes = node.get('input_shapes', [])
        output_shapes = node.get('output_shapes', [])
        
        # Compute MACs for this operation
        op_macs = _compute_operation_macs(op_type, input_shapes, output_shapes, attributes)
        total_macs += op_macs
        
        # Compute memory usage for this operation
        op_memory = _compute_operation_memory(op_type, input_shapes, output_shapes)
        total_memory += op_memory
        
        # Track depth and parallelism
        node_depth = node.get('depth', 0)
        depth = max(depth, node_depth)
        
        if node.get('can_parallelize', False):
            parallel_ops += 1
    
    return {
        'total_macs': total_macs,
        'memory_mb': total_memory / (1024 * 1024),
        'depth': depth,
        'parallel_ops': parallel_ops,
        'avg_macs_per_layer': total_macs / max(len(nodes), 1),
        'memory_efficiency': _compute_memory_efficiency(total_memory, total_macs)
    }


def _compute_operation_macs(op_type: str, input_shapes: List[List[int]], 
                           output_shapes: List[List[int]], attributes: dict) -> int:
    """Compute MACs for a specific operation."""
    if not input_shapes or not output_shapes:
        return 0
    
    op_type_lower = op_type.lower()
    
    try:
        if op_type_lower in ['conv', 'conv2d', 'convolution']:
            return _compute_conv_macs(input_shapes[0], output_shapes[0], attributes)
        elif op_type_lower in ['matmul', 'gemm', 'dense', 'linear']:
            return _compute_matmul_macs(input_shapes[0], output_shapes[0])
        elif op_type_lower in ['batchnormalization', 'batchnorm']:
            return _get_tensor_size(input_shapes[0]) * 4  # 4 ops per element
        elif op_type_lower in ['relu', 'sigmoid', 'tanh', 'activation']:
            return _get_tensor_size(input_shapes[0])  # 1 op per element
        elif op_type_lower in ['add', 'sub', 'mul', 'div']:
            return _get_tensor_size(output_shapes[0])  # 1 op per output element
        elif op_type_lower in ['pooling', 'maxpool', 'avgpool']:
            kernel_size = attributes.get('kernel_shape', [3, 3])
            return _get_tensor_size(output_shapes[0]) * (kernel_size[0] * kernel_size[1])
        else:
            # Default estimation for unknown operations
            return _get_tensor_size(output_shapes[0]) if output_shapes else 0
    except (IndexError, KeyError, TypeError):
        return 0


def _compute_conv_macs(input_shape: List[int], output_shape: List[int], attributes: dict) -> int:
    """Compute MACs for convolution operation."""
    if len(input_shape) < 3 or len(output_shape) < 3:
        return 0
    
    # Extract dimensions (assuming NCHW format)
    batch_size = output_shape[0] if len(output_shape) > 0 else 1
    out_channels = output_shape[1] if len(output_shape) > 1 else 1
    out_height = output_shape[2] if len(output_shape) > 2 else 1
    out_width = output_shape[3] if len(output_shape) > 3 else 1
    
    in_channels = input_shape[1] if len(input_shape) > 1 else 1
    
    # Get kernel dimensions
    kernel_shape = attributes.get('kernel_shape', [3, 3])
    kernel_h = kernel_shape[0] if len(kernel_shape) > 0 else 3
    kernel_w = kernel_shape[1] if len(kernel_shape) > 1 else 3
    
    # MACs = output_elements * (input_channels * kernel_h * kernel_w)
    output_elements = batch_size * out_channels * out_height * out_width
    macs_per_output = in_channels * kernel_h * kernel_w
    
    return output_elements * macs_per_output


def _compute_matmul_macs(input_shape: List[int], output_shape: List[int]) -> int:
    """Compute MACs for matrix multiplication."""
    if len(input_shape) < 2 or len(output_shape) < 2:
        return 0
    
    # For matrix multiplication A @ B = C
    # MACs = batch_size * M * N * K
    # where A is [batch, M, K], B is [batch, K, N], C is [batch, M, N]
    
    batch_size = max(input_shape[0] if len(input_shape) > 0 else 1,
                    output_shape[0] if len(output_shape) > 0 else 1)
    
    m = output_shape[-2] if len(output_shape) >= 2 else 1
    n = output_shape[-1] if len(output_shape) >= 1 else 1
    k = input_shape[-1] if len(input_shape) >= 1 else 1
    
    return batch_size * m * n * k


def _compute_operation_memory(op_type: str, input_shapes: List[List[int]], 
                            output_shapes: List[List[int]]) -> int:
    """Compute memory usage for an operation (in bytes)."""
    total_memory = 0
    
    # Input tensor memory
    for shape in input_shapes:
        total_memory += _get_tensor_size(shape) * 4  # 4 bytes per float32
    
    # Output tensor memory
    for shape in output_shapes:
        total_memory += _get_tensor_size(shape) * 4
    
    # Additional memory for specific operations
    op_type_lower = op_type.lower()
    if op_type_lower in ['conv', 'conv2d', 'convolution']:
        # Additional memory for im2col transformation
        if input_shapes:
            total_memory += _get_tensor_size(input_shapes[0]) * 2
    elif op_type_lower in ['batchnormalization', 'batchnorm']:
        # Memory for running statistics
        if output_shapes:
            channels = output_shapes[0][1] if len(output_shapes[0]) > 1 else 1
            total_memory += channels * 4 * 4  # mean, var, gamma, beta
    
    return total_memory


def _get_tensor_size(shape: List[int]) -> int:
    """Compute total number of elements in a tensor."""
    if not shape:
        return 0
    
    size = 1
    for dim in shape:
        if dim > 0:
            size *= dim
    return size


def _identify_bottlenecks(nodes: List[dict], complexity: dict) -> List[dict]:
    """Identify performance bottlenecks in the model."""
    bottlenecks = []
    total_macs = complexity.get('total_macs', 1)
    
    # Analyze each node for potential bottlenecks
    for i, node in enumerate(nodes):
        op_type = node.get('op_type', node.get('type', 'unknown'))
        input_shapes = node.get('input_shapes', [])
        output_shapes = node.get('output_shapes', [])
        attributes = node.get('attributes', {})
        
        # Compute this node's contribution
        node_macs = _compute_operation_macs(op_type, input_shapes, output_shapes, attributes)
        mac_percentage = (node_macs / total_macs) * 100 if total_macs > 0 else 0
        
        # Check for various bottleneck conditions
        bottleneck_info = {
            'node_index': i,
            'op_type': op_type,
            'mac_percentage': mac_percentage,
            'issues': []
        }
        
        # High computational cost
        if mac_percentage > 20:
            bottleneck_info['issues'].append({
                'type': 'high_compute',
                'description': f'Consumes {mac_percentage:.1f}% of total MACs',
                'severity': 'high' if mac_percentage > 50 else 'medium'
            })
        
        # Large tensor operations
        if input_shapes:
            max_tensor_size = max(_get_tensor_size(shape) for shape in input_shapes)
            if max_tensor_size > 1000000:  # > 1M elements
                bottleneck_info['issues'].append({
                    'type': 'large_tensor',
                    'description': f'Large tensor operation ({max_tensor_size:,} elements)',
                    'severity': 'medium'
                })
        
        # Memory-intensive operations
        memory_usage = _compute_operation_memory(op_type, input_shapes, output_shapes)
        if memory_usage > 100 * 1024 * 1024:  # > 100MB
            bottleneck_info['issues'].append({
                'type': 'high_memory',
                'description': f'High memory usage ({memory_usage / (1024*1024):.1f} MB)',
                'severity': 'medium'
            })
        
        # Inefficient operations
        if op_type.lower() in ['reshape', 'transpose', 'permute'] and mac_percentage > 5:
            bottleneck_info['issues'].append({
                'type': 'inefficient_reshape',
                'description': 'Expensive reshape/transpose operation',
                'severity': 'low'
            })
        
        if bottleneck_info['issues']:
            bottlenecks.append(bottleneck_info)
    
    # Sort by severity and impact
    bottlenecks.sort(key=lambda x: (-x['mac_percentage'], -len(x['issues'])))
    
    return bottlenecks


def _generate_recommendations(complexity: dict, bottlenecks: List[dict], nodes: List[dict]) -> List[dict]:
    """Generate optimization recommendations based on analysis."""
    recommendations = []
    
    total_macs = complexity.get('total_macs', 0)
    memory_mb = complexity.get('memory_mb', 0)
    depth = complexity.get('depth', 0)
    
    # High-level recommendations based on overall complexity
    if total_macs > 1e9:  # > 1 GMAC
        recommendations.append({
            'type': 'quantization',
            'priority': 'high',
            'description': 'Consider quantization to reduce computational cost',
            'potential_speedup': '2-4x',
            'implementation': 'Use INT8 quantization for inference'
        })
    
    if memory_mb > 100:  # > 100MB
        recommendations.append({
            'type': 'memory_optimization',
            'priority': 'high',
            'description': 'Model has high memory requirements',
            'potential_savings': f'{memory_mb * 0.3:.1f} MB',
            'implementation': 'Apply memory-efficient attention or gradient checkpointing'
        })
    
    if depth > 50:
        recommendations.append({
            'type': 'layer_fusion',
            'priority': 'medium',
            'description': 'Deep model could benefit from layer fusion',
            'potential_speedup': '1.5-2x',
            'implementation': 'Fuse consecutive operations like Conv+BN+ReLU'
        })
    
    # Specific recommendations based on bottlenecks
    conv_bottlenecks = [b for b in bottlenecks if 'conv' in b['op_type'].lower()]
    if len(conv_bottlenecks) > 3:
        recommendations.append({
            'type': 'conv_optimization',
            'priority': 'high',
            'description': 'Multiple convolution bottlenecks detected',
            'potential_speedup': '2-3x',
            'implementation': 'Use separable convolutions or efficient convolution algorithms'
        })
    
    matmul_bottlenecks = [b for b in bottlenecks if any(term in b['op_type'].lower() 
                         for term in ['matmul', 'gemm', 'dense'])]
    if len(matmul_bottlenecks) > 2:
        recommendations.append({
            'type': 'matrix_optimization',
            'priority': 'medium',
            'description': 'Multiple matrix multiplication bottlenecks',
            'potential_speedup': '1.5-2x',
            'implementation': 'Use optimized BLAS libraries and consider sparsity'
        })
    
    # Check for parallelization opportunities
    parallel_ops = complexity.get('parallel_ops', 0)
    if parallel_ops > len(nodes) * 0.3:
        recommendations.append({
            'type': 'parallelization',
            'priority': 'medium',
            'description': 'Good parallelization potential detected',
            'potential_speedup': '1.2-1.8x',
            'implementation': 'Use multi-threading or GPU acceleration'
        })
    
    return recommendations


def _analyze_memory_usage(nodes: List[dict], tensors: dict) -> dict:
    """Analyze memory usage patterns."""
    peak_memory = 0
    current_memory = 0
    memory_timeline = []
    
    for i, node in enumerate(nodes):
        input_shapes = node.get('input_shapes', [])
        output_shapes = node.get('output_shapes', [])
        
        # Add output tensor memory
        for shape in output_shapes:
            current_memory += _get_tensor_size(shape) * 4
        
        peak_memory = max(peak_memory, current_memory)
        memory_timeline.append({
            'step': i,
            'memory_mb': current_memory / (1024 * 1024),
            'op_type': node.get('op_type', 'unknown')
        })
        
        # Simulate memory deallocation (simplified)
        if i > 10:  # Keep last 10 tensors in memory
            current_memory *= 0.8
    
    return {
        'peak_memory_mb': peak_memory / (1024 * 1024),
        'avg_memory_mb': sum(t['memory_mb'] for t in memory_timeline) / len(memory_timeline),
        'memory_timeline': memory_timeline[-20:],  # Keep last 20 steps
        'memory_efficiency_score': _compute_memory_efficiency_score(peak_memory, len(nodes))
    }


def _analyze_graph_properties(nodes: List[dict], edges: List[dict]) -> dict:
    """Analyze computational graph structure properties."""
    # Build adjacency information
    node_connections = defaultdict(list)
    for edge in edges:
        src = edge.get('source', edge.get('from'))
        dst = edge.get('target', edge.get('to'))
        if src is not None and dst is not None:
            node_connections[src].append(dst)
    
    # Compute graph metrics
    max_fanout = max(len(connections) for connections in node_connections.values()) if node_connections else 0
    avg_fanout = sum(len(connections) for connections in node_connections.values()) / max(len(node_connections), 1)
    
    # Identify parallel branches
    parallel_branches = 0
    for connections in node_connections.values():
        if len(connections) > 1:
            parallel_branches += len(connections) - 1
    
    return {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'max_fanout': max_fanout,
        'avg_fanout': avg_fanout,
        'parallel_branches': parallel_branches,
        'graph_complexity': len(edges) / max(len(nodes), 1),
        'estimated_pipeline_depth': _estimate_pipeline_depth(nodes, node_connections)
    }


def _compute_optimization_score(complexity: dict, bottlenecks: List[dict]) -> float:
    """Compute overall optimization score (0-1, higher is better optimized)."""
    base_score = 1.0
    
    # Penalize high complexity
    total_macs = complexity.get('total_macs', 0)
    if total_macs > 1e9:
        base_score -= 0.3
    elif total_macs > 1e8:
        base_score -= 0.2
    
    # Penalize high memory usage
    memory_mb = complexity.get('memory_mb', 0)
    if memory_mb > 1000:
        base_score -= 0.3
    elif memory_mb > 100:
        base_score -= 0.2
    
    # Penalize bottlenecks
    high_impact_bottlenecks = len([b for b in bottlenecks if b['mac_percentage'] > 20])
    base_score -= high_impact_bottlenecks * 0.1
    
    # Reward parallelization opportunities
    parallel_ops = complexity.get('parallel_ops', 0)
    total_ops = max(len(bottlenecks), 1)
    if parallel_ops / total_ops > 0.5:
        base_score += 0.1
    
    return max(0.0, min(1.0, base_score))


# Helper functions for estimation
def _estimate_conv_macs(conv_count: int) -> int:
    """Estimate MACs for convolution layers."""
    return conv_count * 1e6  # Assume 1M MACs per conv layer


def _estimate_matmul_macs(matmul_count: int) -> int:
    """Estimate MACs for matrix multiplication layers."""
    return matmul_count * 5e5  # Assume 500K MACs per matmul


def _compute_complexity_score(macs: float, memory_mb: float) -> float:
    """Compute normalized complexity score."""
    # Normalize based on typical model ranges
    mac_score = min(macs / 1e9, 10) / 10  # 0-1 scale, 1B MACs = 0.1
    memory_score = min(memory_mb / 1000, 10) / 10  # 0-1 scale, 1GB = 0.1
    
    return (mac_score + memory_score) / 2


def _estimate_inference_time(macs: float, memory_mb: float) -> float:
    """Estimate inference time in milliseconds."""
    # Rough estimates based on typical hardware
    cpu_throughput = 1e9  # 1 GMAC/s on CPU
    memory_bandwidth = 100  # 100 GB/s memory bandwidth
    
    compute_time = (macs / cpu_throughput) * 1000  # Convert to ms
    memory_time = (memory_mb / 1024 / memory_bandwidth) * 1000
    
    return max(compute_time, memory_time)


def _compute_memory_efficiency(total_memory: int, total_macs: int) -> float:
    """Compute memory efficiency score."""
    if total_macs == 0:
        return 0.0
    
    # Memory efficiency = MACs per byte
    return total_macs / max(total_memory, 1)


def _compute_memory_efficiency_score(peak_memory: int, num_nodes: int) -> float:
    """Compute memory efficiency score (0-1, higher is better)."""
    # Idealized memory usage would be linear with number of nodes
    ideal_memory = num_nodes * 1024 * 1024  # 1MB per node
    
    if peak_memory <= ideal_memory:
        return 1.0
    else:
        return ideal_memory / peak_memory


def _estimate_pipeline_depth(nodes: List[dict], connections: dict) -> int:
    """Estimate pipeline depth through longest path."""
    if not nodes or not connections:
        return len(nodes)
    
    # Simple longest path estimation
    max_depth = 0
    for i in range(len(nodes)):
        depth = _compute_node_depth(i, connections, set())
        max_depth = max(max_depth, depth)
    
    return max_depth


def _compute_node_depth(node_id: int, connections: dict, visited: set) -> int:
    """Recursively compute depth from a node."""
    if node_id in visited:
        return 0
    
    visited.add(node_id)
    max_child_depth = 0
    
    for child in connections.get(node_id, []):
        child_depth = _compute_node_depth(child, connections, visited.copy())
        max_child_depth = max(max_child_depth, child_depth)
    
    return 1 + max_child_depth


