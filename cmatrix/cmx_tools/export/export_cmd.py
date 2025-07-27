#!/usr/bin/env python3
"""
CMatrix Model Export CLI

Command-line interface for exporting ML models from various formats 
(PyTorch, TensorFlow, ONNX) to CMatrix internal format.

Usage:
  python export_cmd.py --input model.onnx --format onnx --output model.cmx.json
  python export_cmd.py --input model.pth --format torch --output model.cmx --verbose
  python export_cmd.py --input saved_model/ --format tf --output model.cmx.bin
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Import CMatrix export modules
try:
    from cmx_tools.export.cmx_exporter import export_model
    from cmx_tools.export.model_serializer import serialize_model
    from cmx_tools.export.format_validator import validate_model_format, get_validation_report
except ImportError as e:
    print(f"Error: Failed to import CMatrix modules: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed.")
    sys.exit(2)

# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 1
EXIT_EXPORT_FAILURE = 2

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Export ML models to CMatrix format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export ONNX model to JSON format
  python export_cmd.py --input model.onnx --format onnx --output model.cmx.json
  
  # Export PyTorch model to binary format with validation
  python export_cmd.py --input model.pth --format torch --output model.cmx --validate
  
  # Export TensorFlow saved model with verbose output
  python export_cmd.py --input saved_model/ --format tf --output model.cmx.bin --verbose
  
  # Auto-detect format and compress output
  python export_cmd.py --input model.onnx --output model.cmx.json.gz --compress
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input model file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file path (.json, .bin, .cmx, optionally with .gz)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['onnx', 'torch', 'tf', 'auto'],
        default='auto',
        help='Input model format (default: auto-detect)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate exported model format'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use strict validation (treat warnings as errors)'
    )
    
    parser.add_argument(
        '--check-embedded',
        action='store_true',
        help='Check compatibility for embedded deployment'
    )
    
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Compress output file with gzip'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-sanitize',
        action='store_true',
        help='Skip post-processing sanitization'
    )
    
    # Format-specific arguments
    parser.add_argument(
        '--input-shape',
        type=str,
        help='Input shape for PyTorch models (e.g., "1,3,224,224")'
    )
    
    parser.add_argument(
        '--use-onnx-fallback',
        action='store_true',
        help='Use ONNX export fallback for PyTorch models'
    )
    
    parser.add_argument(
        '--use-concrete-function',
        action='store_true',
        help='Use concrete function tracing for TensorFlow models'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        default=True,
        help='Apply graph optimizations for ONNX models (default: True)'
    )
    
    return parser.parse_args()

def validate_inputs(args: argparse.Namespace) -> bool:
    """Validate command line arguments and input files"""
    
    # Check input file/directory exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return False
    
    # Validate input based on format
    if args.format == 'onnx':
        if not (input_path.is_file() and input_path.suffix.lower() == '.onnx'):
            print(f"Error: ONNX format requires .onnx file, got: {args.input}")
            return False
    elif args.format == 'torch':
        if not (input_path.is_file() and input_path.suffix.lower() in ['.pth', '.pt']):
            print(f"Warning: PyTorch format typically uses .pth or .pt files")
    elif args.format == 'tf':
        if input_path.is_file() and not input_path.suffix.lower() in ['.pb', '.h5']:
            print(f"Warning: TensorFlow format typically uses .pb, .h5 files or SavedModel directory")
    
    # Check output directory is writable
    output_path = Path(args.output)
    output_dir = output_path.parent
    
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create output directory {output_dir}: {e}")
            return False
    
    if not os.access(output_dir, os.W_OK):
        print(f"Error: Output directory is not writable: {output_dir}")
        return False
    
    # Parse input shape if provided
    if args.input_shape:
        try:
            shape_parts = [int(x.strip()) for x in args.input_shape.split(',')]
            args.input_shape = tuple(shape_parts)
        except ValueError:
            print(f"Error: Invalid input shape format: {args.input_shape}")
            print("Expected format: '1,3,224,224'")
            return False
    
    return True

def load_model(input_path: str, format_type: str, verbose: bool = False):
    """Load model based on format type"""
    
    if verbose:
        print(f"Loading {format_type} model from: {input_path}")
    
    if format_type == 'torch':
        try:
            import torch
            if input_path.endswith(('.pth', '.pt')):
                # Load state dict and create model
                state_dict = torch.load(input_path, map_location='cpu')
                if verbose:
                    print(f"Loaded PyTorch state dict with {len(state_dict)} entries")
                # Note: This requires the model architecture to be defined elsewhere
                # For a complete CLI, you'd need additional logic to reconstruct the model
                return state_dict
            else:
                # Assume it's a model file or script
                model = torch.jit.load(input_path, map_location='cpu')
                return model
        except ImportError:
            print("Error: PyTorch not installed")
            return None
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            return None
    
    elif format_type == 'tf':
        try:
            import tensorflow as tf
            if os.path.isdir(input_path):
                # SavedModel directory
                model = tf.saved_model.load(input_path)
            else:
                # Keras model file
                model = tf.keras.models.load_model(input_path)
            return model
        except ImportError:
            print("Error: TensorFlow not installed")
            return None
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            return None
    
    elif format_type == 'onnx':
        # ONNX models are loaded directly by the converter
        return input_path
    
    else:
        print(f"Error: Unsupported format: {format_type}")
        return None

def determine_serialization_format(output_path: str) -> str:
    """Determine serialization format from output file extension"""
    
    path = Path(output_path)
    
    # Remove .gz if present
    if path.suffix == '.gz':
        path = path.with_suffix('')
    
    if path.suffix.lower() == '.json':
        return 'json'
    elif path.suffix.lower() in ['.bin', '.cmx']:
        return 'binary'
    else:
        # Default to binary for unknown extensions
        return 'binary'

def print_export_summary(cmx_graph, serialization_info: Dict[str, Any], verbose: bool = False):
    """Print summary of exported model"""
    
    print("\n" + "="*50)
    print("EXPORT SUMMARY")
    print("="*50)
    
    # Model information
    metadata = cmx_graph.metadata
    print(f"Framework: {metadata.get('framework', 'unknown')}")
    print(f"Number of operations: {len(cmx_graph.nodes)}")
    print(f"Number of weights: {len(cmx_graph.weights)}")
    print(f"Number of inputs: {len(cmx_graph.inputs)}")
    print(f"Number of outputs: {len(cmx_graph.outputs)}")
    
    # Parameter count
    total_params = sum(w.size for w in cmx_graph.weights.values() if hasattr(w, 'size'))
    print(f"Total parameters: {total_params:,}")
    
    # Model size
    total_size_mb = sum(w.nbytes for w in cmx_graph.weights.values() if hasattr(w, 'nbytes')) / (1024 * 1024)
    print(f"Model size: {total_size_mb:.2f} MB")
    
    # Input/Output shapes
    if verbose:
        print(f"\nInputs:")
        for i, inp in enumerate(cmx_graph.inputs):
            if isinstance(inp, dict):
                print(f"  {i}: {inp.get('name', 'unnamed')} - {inp.get('shape', 'unknown')}")
            else:
                print(f"  {i}: {inp}")
        
        print(f"\nOutputs:")
        for i, out in enumerate(cmx_graph.outputs):
            if isinstance(out, dict):
                print(f"  {i}: {out.get('name', 'unnamed')} - {out.get('shape', 'unknown')}")
            else:
                print(f"  {i}: {out}")
        
        print(f"\nOperation types used:")
        op_types = list(set(node.op_type for node in cmx_graph.nodes.values()))
        for op_type in sorted(op_types):
            count = sum(1 for node in cmx_graph.nodes.values() if node.op_type == op_type)
            print(f"  {op_type}: {count}")
    
    # File information
    print(f"\nOutput file: {serialization_info.get('output_path', 'unknown')}")
    print(f"Output format: {serialization_info.get('format', 'unknown')}")
    print(f"Compressed: {'Yes' if serialization_info.get('compressed', False) else 'No'}")
    print(f"File size: {serialization_info.get('file_size_bytes', 0) / (1024*1024):.2f} MB")
    print(f"Checksum: {serialization_info.get('checksum', 'unknown')[:8]}...")

def main() -> int:
    """Main CLI function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate inputs
    if not validate_inputs(args):
        return EXIT_VALIDATION_ERROR
    
    # Setup verbose output
    verbose = args.verbose
    start_time = time.time()
    
    if verbose:
        print("CMatrix Model Export CLI")
        print("="*30)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Format: {args.format}")
        print()
    
    try:
        # Determine format
        format_type = args.format
        if format_type == 'auto':
            if args.input.endswith('.onnx'):
                format_type = 'onnx'
            elif args.input.endswith(('.pth', '.pt')):
                format_type = 'torch'
            elif args.input.endswith(('.pb', '.h5')) or os.path.isdir(args.input):
                format_type = 'tf'
            else:
                print("Error: Could not auto-detect format. Please specify --format")
                return EXIT_VALIDATION_ERROR
        
        if verbose:
            print(f"Detected format: {format_type}")
        
        # Prepare export parameters
        export_kwargs = {}
        
        if format_type == 'torch':
            if args.input_shape:
                export_kwargs['input_shape'] = args.input_shape
            if args.use_onnx_fallback:
                export_kwargs['use_onnx_fallback'] = True
        
        elif format_type == 'tf':
            if args.use_concrete_function:
                export_kwargs['use_concrete_function'] = True
        
        elif format_type == 'onnx':
            export_kwargs['optimize'] = args.optimize
        
        # Load and export model
        if verbose:
            print("Exporting model...")
        
        model = load_model(args.input, format_type, verbose)
        if model is None:
            return EXIT_EXPORT_FAILURE
        
        cmx_graph = export_model(
            model,
            format_type=format_type,
            sanitize=not args.no_sanitize,
            **export_kwargs
        )
        
        if verbose:
            print("Export completed successfully!")
        
        # Validate if requested
        if args.validate:
            if verbose:
                print("Validating exported model...")
            
            try:
                validate_model_format(
                    cmx_graph,
                    strict=args.strict,
                    check_embedded=args.check_embedded
                )
                if verbose:
                    print("Validation passed!")
            except Exception as e:
                print(f"Validation failed: {e}")
                if args.strict:
                    return EXIT_VALIDATION_ERROR
        
        # Serialize model
        if verbose:
            print("Serializing model...")
        
        serialization_format = determine_serialization_format(args.output)
        
        serialization_info = serialize_model(
            cmx_graph,
            args.output,
            format_type=serialization_format,
            compress=args.compress
        )
        
        # Print summary
        print_export_summary(cmx_graph, serialization_info, verbose)
        
        # Timing information
        elapsed_time = time.time() - start_time
        print(f"\nExport completed in {elapsed_time:.2f} seconds")
        
        return EXIT_SUCCESS
        
    except KeyboardInterrupt:
        print("\nExport interrupted by user")
        return EXIT_EXPORT_FAILURE
        
    except Exception as e:
        print(f"Export failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return EXIT_EXPORT_FAILURE

if __name__ == "__main__":
    sys.exit(main())


