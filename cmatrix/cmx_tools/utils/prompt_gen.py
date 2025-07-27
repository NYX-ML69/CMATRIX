"""
Prompt generation utilities for AI-assisted code generation.

Generates model-specific prompts for auto-generating operator/kernel code
or configurations based on model structure and requirements.
"""

from typing import Dict, List, Optional, Any
import json


def generate_op_prompt(model_structure: dict) -> str:
    """
    Generate a natural language prompt for an operator based on model structure.
    
    Args:
        model_structure: Dictionary containing model layer/operator information
                        Expected keys: 'op_type', 'input_shape', 'output_shape', 
                        'parameters', 'attributes'
    
    Returns:
        str: Natural language prompt for AI code generation
    """
    op_type = model_structure.get('op_type', 'unknown')
    input_shape = model_structure.get('input_shape', [])
    output_shape = model_structure.get('output_shape', [])
    params = model_structure.get('parameters', {})
    attrs = model_structure.get('attributes', {})
    
    prompt_parts = [
        f"Generate a {op_type} operator implementation with the following specifications:",
        f"Input tensor shape: {input_shape}",
        f"Output tensor shape: {output_shape}"
    ]
    
    if params:
        prompt_parts.append(f"Parameters: {_format_dict_for_prompt(params)}")
    
    if attrs:
        prompt_parts.append(f"Attributes: {_format_dict_for_prompt(attrs)}")
    
    # Add operation-specific requirements
    if op_type.lower() in ['conv2d', 'convolution']:
        prompt_parts.extend([
            "Requirements:",
            "- Implement efficient convolution with proper padding handling",
            "- Support both NCHW and NHWC data layouts",
            "- Include bounds checking for memory safety"
        ])
    elif op_type.lower() in ['matmul', 'gemm']:
        prompt_parts.extend([
            "Requirements:",
            "- Implement optimized matrix multiplication",
            "- Handle broadcasting for batch dimensions",
            "- Use appropriate BLAS routines where possible"
        ])
    elif op_type.lower() in ['relu', 'activation']:
        prompt_parts.extend([
            "Requirements:",
            "- Implement element-wise activation function",
            "- Ensure numerical stability",
            "- Support in-place operations where safe"
        ])
    
    prompt_parts.extend([
        "",
        "The implementation should be:",
        "- Memory efficient and numerically stable",
        "- Compatible with CMatrix tensor format",
        "- Include proper error handling and validation",
        "- Follow CMatrix coding conventions"
    ])
    
    return "\n".join(prompt_parts)


def generate_config_prompt(target_config: dict) -> str:
    """
    Generate prompt for creating configuration files.
    
    Args:
        target_config: Dictionary with target configuration requirements
                      Expected keys: 'target', 'optimization_level', 'constraints'
    
    Returns:
        str: Prompt for generating configuration
    """
    target = target_config.get('target', 'generic')
    opt_level = target_config.get('optimization_level', 'O2')
    constraints = target_config.get('constraints', {})
    
    prompt_parts = [
        f"Generate a CMatrix configuration file for {target} target with {opt_level} optimization.",
        "",
        "Configuration should include:"
    ]
    
    # Standard configuration sections
    sections = [
        "- Compiler settings and optimization flags",
        "- Memory allocation strategy",
        "- Parallelization settings",
        "- Target-specific optimizations"
    ]
    
    if constraints:
        sections.append(f"- Constraints: {_format_dict_for_prompt(constraints)}")
    
    prompt_parts.extend(sections)
    
    # Add target-specific requirements
    if 'cuda' in target.lower() or 'gpu' in target.lower():
        prompt_parts.extend([
            "",
            "GPU-specific requirements:",
            "- CUDA kernel launch parameters",
            "- Memory coalescing strategies",
            "- Shared memory usage optimization"
        ])
    elif 'cpu' in target.lower():
        prompt_parts.extend([
            "",
            "CPU-specific requirements:",
            "- SIMD instruction utilization",
            "- Cache-friendly memory access patterns",
            "- Thread pool configuration"
        ])
    
    return "\n".join(prompt_parts)


def generate_optimization_prompt(model_info: dict, target_metrics: dict) -> str:
    """
    Generate prompt for model optimization suggestions.
    
    Args:
        model_info: Dictionary with model analysis results
        target_metrics: Dictionary with target performance metrics
    
    Returns:
        str: Prompt for optimization recommendations
    """
    model_size = model_info.get('model_size_mb', 0)
    complexity = model_info.get('complexity_score', 0)
    bottlenecks = model_info.get('bottlenecks', [])
    
    target_latency = target_metrics.get('max_latency_ms', 'not specified')
    target_memory = target_metrics.get('max_memory_mb', 'not specified')
    
    prompt_parts = [
        f"Analyze and suggest optimizations for a model with:",
        f"- Size: {model_size} MB",
        f"- Complexity score: {complexity}",
        f"- Target latency: {target_latency} ms",
        f"- Target memory: {target_memory} MB"
    ]
    
    if bottlenecks:
        prompt_parts.extend([
            "",
            "Identified bottlenecks:",
            *[f"- {bottleneck}" for bottleneck in bottlenecks]
        ])
    
    prompt_parts.extend([
        "",
        "Provide specific recommendations for:",
        "- Layer fusion opportunities",
        "- Quantization strategies",
        "- Memory layout optimizations",
        "- Computational graph simplifications",
        "- Target-specific optimizations"
    ])
    
    return "\n".join(prompt_parts)


def _format_dict_for_prompt(data: Dict[str, Any]) -> str:
    """Format dictionary data for inclusion in prompts."""
    if not data:
        return "none"
    
    formatted_items = []
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            value_str = json.dumps(value, indent=None, separators=(',', ':'))
        else:
            value_str = str(value)
        formatted_items.append(f"{key}={value_str}")
    
    return ", ".join(formatted_items)


def generate_kernel_prompt(op_name: str, hardware_target: str, performance_requirements: Optional[Dict] = None) -> str:
    """
    Generate prompt for hardware-specific kernel implementation.
    
    Args:
        op_name: Name of the operation (e.g., 'conv2d', 'matmul')
        hardware_target: Target hardware (e.g., 'cuda', 'opencl', 'cpu_avx')
        performance_requirements: Optional performance constraints
    
    Returns:
        str: Prompt for kernel implementation
    """
    requirements = performance_requirements or {}
    
    prompt_parts = [
        f"Implement a high-performance {op_name} kernel for {hardware_target}.",
        "",
        "Requirements:"
    ]
    
    # Hardware-specific optimizations
    if 'cuda' in hardware_target.lower():
        prompt_parts.extend([
            "- Optimize thread block and grid dimensions",
            "- Utilize shared memory effectively",
            "- Implement memory coalescing",
            "- Consider tensor core usage if applicable"
        ])
    elif 'opencl' in hardware_target.lower():
        prompt_parts.extend([
            "- Optimize work group sizes",
            "- Use local memory efficiently",
            "- Handle different OpenCL device types",
            "- Ensure portability across vendors"
        ])
    elif 'cpu' in hardware_target.lower():
        prompt_parts.extend([
            "- Utilize SIMD instructions (AVX, NEON)",
            "- Implement cache blocking strategies",
            "- Consider OpenMP parallelization",
            "- Optimize for branch prediction"
        ])
    
    # Add performance requirements
    if requirements.get('max_latency'):
        prompt_parts.append(f"- Target latency: < {requirements['max_latency']} ms")
    if requirements.get('min_throughput'):
        prompt_parts.append(f"- Minimum throughput: > {requirements['min_throughput']} ops/sec")
    if requirements.get('max_memory'):
        prompt_parts.append(f"- Memory usage: < {requirements['max_memory']} MB")
    
    prompt_parts.extend([
        "",
        "The kernel should include:",
        "- Comprehensive input validation",
        "- Error handling and status reporting",
        "- Performance profiling hooks",
        "- Clear documentation and comments"
    ])
    
    return "\n".join(prompt_parts)

