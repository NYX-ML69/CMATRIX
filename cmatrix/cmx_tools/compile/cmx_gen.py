"""
cmx_gen.py - Main compilation entry point

Primary user-facing interface for compiling models to embedded targets.
"""

import os
import argparse
from typing import Optional, Union, Dict, Any
from pathlib import Path

from .backend_flags import get_backend_flags
from .optimization_passes import optimize_graph
from .quantization_engine import quantize
from .code_generator import generate_code


def compile_model(
    model: Union[str, Dict[str, Any]], 
    target: str = 'cortex-m',
    quantize_model: bool = True,
    output_dir: str = './output',
    optimization_level: int = 2,
    calibration_data: Optional[Any] = None
) -> str:
    """
    Compile a model to optimized C/C++ code for embedded targets.
    
    Args:
        model: Path to model file (ONNX) or internal IR representation
        target: Target backend ('cortex-m', 'riscv', 'xtensa', 'generic')
        quantize_model: Whether to apply quantization
        output_dir: Directory to save generated code
        optimization_level: Optimization level (0-3)
        calibration_data: Data for quantization calibration
        
    Returns:
        str: Path to generated C++ file
    """
    
    # Validate target
    supported_targets = ['cortex-m', 'riscv', 'xtensa', 'generic']
    if target not in supported_targets:
        raise ValueError(f"Unsupported target '{target}'. Supported: {supported_targets}")
    
    # Load model (placeholder - would parse ONNX or IR)
    if isinstance(model, str):
        print(f"Loading model from: {model}")
        # model_ir = load_model(model)  # Implementation needed
        model_ir = {"dummy": "graph"}  # Placeholder
    else:
        model_ir = model
    
    # Get backend configuration
    backend_config = get_backend_flags(target)
    print(f"Using backend config for {target}: {backend_config}")
    
    # Apply optimizations
    if optimization_level > 0:
        print("Applying graph optimizations...")
        model_ir = optimize_graph(model_ir, level=optimization_level)
    
    # Apply quantization
    if quantize_model:
        print("Applying quantization...")
        model_ir = quantize(model_ir, mode='int8', calibration_data=calibration_data)
    
    # Generate code
    print("Generating C++ code...")
    output_path = generate_code(model_ir, target, output_dir)
    
    print(f"Compilation complete! Generated: {output_path}")
    return output_path


def main():
    """CLI interface for model compilation."""
    parser = argparse.ArgumentParser(description='Compile models for embedded targets')
    
    parser.add_argument('model', help='Path to model file (ONNX)')
    parser.add_argument('--target', choices=['cortex-m', 'riscv', 'xtensa', 'generic'], 
                       default='cortex-m', help='Target backend')
    parser.add_argument('--no-quantize', action='store_true', 
                       help='Disable quantization')
    parser.add_argument('--output-dir', default='./output', 
                       help='Output directory')
    parser.add_argument('--opt-level', type=int, choices=[0, 1, 2, 3], 
                       default=2, help='Optimization level')
    
    args = parser.parse_args()
    
    try:
        output_file = compile_model(
            model=args.model,
            target=args.target,
            quantize_model=not args.no_quantize,
            output_dir=args.output_dir,
            optimization_level=args.opt_level
        )
        print(f"Success: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == '__main__':
    exit(main())


