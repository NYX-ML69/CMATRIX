"""
Configuration parsing utilities for CMatrix tools.

Handles parsing and validation of .cmxconfig, .yaml, and .json
configuration files used during export, compile, and optimization processes.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False


def parse_config(config_path: str) -> dict:
    """
    Load and validate compile/export configuration from file.
    
    Args:
        config_path: Path to configuration file (.json, .yaml, .yml, .toml, .cmxconfig)
    
    Returns:
        dict: Parsed and validated configuration
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config_path = Path(config_path)
    extension = config_path.suffix.lower()
    
    # Parse based on file extension
    try:
        if extension == '.json':
            config = _parse_json_config(config_path)
        elif extension in ['.yaml', '.yml']:
            config = _parse_yaml_config(config_path)
        elif extension == '.toml':
            config = _parse_toml_config(config_path)
        elif extension == '.cmxconfig':
            config = _parse_cmx_config(config_path)
        else:
            # Try to auto-detect format
            config = _parse_auto_detect(config_path)
        
        # Validate and apply defaults
        config = validate_config(config)
        
        # Add metadata
        config['_metadata'] = {
            'config_file': str(config_path),
            'format': extension,
            'parsed_successfully': True
        }
        
        return config
        
    except Exception as e:
        logging.error(f"Failed to parse config {config_path}: {e}")
        raise ValueError(f"Invalid configuration file {config_path}: {e}")


def validate_config(config: dict) -> dict:
    """
    Validate configuration dictionary and apply defaults.
    
    Args:
        config: Raw configuration dictionary
    
    Returns:
        dict: Validated configuration with defaults applied
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Create a copy to avoid modifying the original
    validated_config = config.copy()
    
    # Apply default values for required sections
    validated_config = _apply_default_values(validated_config)
    
    # Validate required fields
    _validate_required_fields(validated_config)
    
    # Validate field types and values
    _validate_field_types(validated_config)
    
    # Validate cross-field dependencies
    _validate_dependencies(validated_config)
    
    # Normalize and sanitize values
    validated_config = _normalize_config_values(validated_config)
    
    return validated_config


def _parse_json_config(config_path: Path) -> dict:
    """Parse JSON configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _parse_yaml_config(config_path: Path) -> dict:
    """Parse YAML configuration file."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required to parse YAML configuration files")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _parse_toml_config(config_path: Path) -> dict:
    """Parse TOML configuration file."""
    if not HAS_TOML:
        raise ImportError("toml is required to parse TOML configuration files")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return toml.load(f)


def _parse_cmx_config(config_path: Path) -> dict:
    """Parse CMatrix-specific configuration file."""
    config = {}
    current_section = None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            try:
                # Section headers [section_name]
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1].strip()
                    if current_section not in config:
                        config[current_section] = {}
                    continue
                
                # Key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse value type
                    parsed_value = _parse_config_value(value)
                    
                    if current_section:
                        config[current_section][key] = parsed_value
                    else:
                        config[key] = parsed_value
                else:
                    logging.warning(f"Invalid line in {config_path}:{line_num}: {line}")
                    
            except Exception as e:
                logging.warning(f"Error parsing line {line_num} in {config_path}: {e}")
                continue
    
    return config


def _parse_auto_detect(config_path: Path) -> dict:
    """Auto-detect configuration file format."""
    # Try parsers in order of preference
    parsers = [
        ('JSON', _parse_json_config),
        ('YAML', _parse_yaml_config),
        ('TOML', _parse_toml_config),
        ('CMX', _parse_cmx_config)
    ]
    
    for parser_name, parser_func in parsers:
        try:
            return parser_func(config_path)
        except Exception:
            continue
    
    raise ValueError(f"Could not auto-detect format for {config_path}")


def _parse_config_value(value_str: str) -> Any:
    """Parse a configuration value string to appropriate Python type."""
    value_str = value_str.strip()
    
    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    
    # Boolean values
    if value_str.lower() in ['true', 'on', 'yes', '1']:
        return True
    if value_str.lower() in ['false', 'off', 'no', '0']:
        return False
    
    # Null/None values
    if value_str.lower() in ['null', 'none', '']:
        return None
    
    # Try to parse as number
    try:
        if '.' in value_str or 'e' in value_str.lower():
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass
    
    # Try to parse as list
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            list_content = value_str[1:-1].strip()
            if not list_content:
                return []
            
            items = [_parse_config_value(item.strip()) 
                    for item in list_content.split(',')]
            return items
        except Exception:
            pass
    
    # Return as string
    return value_str


def _apply_default_values(config: dict) -> dict:
    """Apply default values for missing configuration options."""
    defaults = {
        'target': {
            'platform': 'cpu',
            'architecture': 'x86_64',
            'optimization_level': 'O2',
            'precision': 'float32'
        },
        'compiler': {
            'backend': 'llvm',
            'debug_info': False,
            'warnings_as_errors': False,
            'optimization_flags': []
        },
        'memory': {
            'allocation_strategy': 'static',
            'max_memory_mb': 1024,
            'enable_memory_pool': True,
            'alignment': 32
        },
        'parallelization': {
            'enable_threading': True,
            'max_threads': -1,  # Auto-detect
            'enable_vectorization': True,
            'enable_gpu': False
        },
        'optimization': {
            'enable_fusion': True,
            'enable_quantization': False,
            'quantization_type': 'int8',
            'enable_pruning': False
        },
        'runtime': {
            'enable_profiling': False,
            'profile_output': 'profile.json',
            'enable_logging': True,
            'log_level': 'info'
        },
        'output': {
            'format': 'library',
            'output_directory': './output',
            'library_name': 'model',
            'generate_headers': True
        }
    }
    
    # Merge defaults with provided config
    merged_config = {}
    for section_name, section_defaults in defaults.items():
        if section_name in config:
            # Merge section-level defaults
            merged_section = section_defaults.copy()
            merged_section.update(config[section_name])
            merged_config[section_name] = merged_section
        else:
            merged_config[section_name] = section_defaults.copy()
    
    # Add any additional sections from the original config
    for section_name, section_config in config.items():
        if section_name not in merged_config:
            merged_config[section_name] = section_config
    
    return merged_config


def _validate_required_fields(config: dict) -> None:
    """Validate that required configuration fields are present."""
    required_sections = ['target', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section missing: {section}")
    
    # Section-specific required fields
    section_requirements = {
        'target': ['platform'],
        'output': ['format', 'output_directory']
    }
    
    for section, required_fields in section_requirements.items():
        section_config = config.get(section, {})
        for field in required_fields:
            if field not in section_config:
                raise ValueError(f"Required field missing in {section}: {field}")


def _validate_field_types(config: dict) -> None:
    """Validate configuration field types and values."""
    type_validations = {
        'target.platform': (str, ['cpu', 'gpu', 'cuda', 'opencl', 'vulkan']),
        'target.optimization_level': (str, ['O0', 'O1', 'O2', 'O3', 'Os', 'Oz']),
        'target.precision': (str, ['float16', 'float32', 'float64', 'int8', 'int16', 'int32']),
        'compiler.debug_info': (bool, None),
        'compiler.warnings_as_errors': (bool, None),
        'memory.max_memory_mb': (int, lambda x: x > 0),
        'memory.alignment': (int, lambda x: x > 0 and (x & (x - 1)) == 0),  # Power of 2
        'parallelization.enable_threading': (bool, None),
        'parallelization.max_threads': (int, lambda x: x == -1 or x > 0),
        'optimization.quantization_type': (str, ['int8', 'int16', 'float16']),
        'runtime.log_level': (str, ['debug', 'info', 'warning', 'error']),
        'output.format': (str, ['library', 'executable', 'shared_library', 'static_library'])
    }
    
    for field_path, (expected_type, valid_values) in type_validations.items():
        sections = field_path.split('.')
        value = config
        
        # Navigate to the field
        try:
            for section in sections:
                value = value[section]
        except KeyError:
            continue  # Field not present, handled by required field validation
        
        # Check type
        if not isinstance(value, expected_type):
            raise ValueError(f"Invalid type for {field_path}: expected {expected_type.__name__}, got {type(value).__name__}")
        
        # Check valid values
        if valid_values is not None:
            if callable(valid_values):
                if not valid_values(value):
                    raise ValueError(f"Invalid value for {field_path}: {value}")
            elif isinstance(valid_values, (list, tuple)):
                if value not in valid_values:
                    raise ValueError(f"Invalid value for {field_path}: {value} (must be one of {valid_values})")


def _validate_dependencies(config: dict) -> None:
    """Validate cross-field dependencies."""
    # GPU-specific validations
    if config.get('target', {}).get('platform') in ['gpu', 'cuda']:
        if not config.get('parallelization', {}).get('enable_gpu', False):
            logging.warning("GPU platform selected but GPU parallelization not enabled")
    
    # Quantization dependencies
    if config.get('optimization', {}).get('enable_quantization', False):
        precision = config.get('target', {}).get('precision', 'float32')
        if precision not in ['float32', 'float16']:
            raise ValueError("Quantization requires float32 or float16 base precision")
    
    # Memory pool dependencies
    if config.get('memory', {}).get('enable_memory_pool', False):
        if config.get('memory', {}).get('allocation_strategy') != 'static':
            logging.warning("Memory pool works best with static allocation strategy")
    
    # Output directory validation
    output_dir = config.get('output', {}).get('output_directory', '.')
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}")


def _normalize_config_values(config: dict) -> dict:
    """Normalize and sanitize configuration values."""
    normalized = {}
    
    for section_name, section_config in config.items():
        if isinstance(section_config, dict):
            normalized_section = {}
            for key, value in section_config.items():
                # Normalize paths
                if key.endswith('_directory') or key.endswith('_path') or key.endswith('_file'):
                    value = str(Path(value).resolve())
                
                # Normalize boolean strings
                elif isinstance(value, str) and key.startswith('enable_'):
                    value = value.lower() in ['true', 'on', 'yes', '1']
                
                # Normalize list values
                elif isinstance(value, str) and key.endswith('_list'):
                    value = [item.strip() for item in value.split(',') if item.strip()]
                
                normalized_section[key] = value
            
            normalized[section_name] = normalized_section
        else:
            normalized[section_name] = config[section_name]
    
    return normalized


def save_config(config: dict, output_path: str, format_type: Optional[str] = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Output file path
        format_type: Output format ('json', 'yaml', 'toml', 'cmx'). Auto-detected if None.
    """
    output_path = Path(output_path)
    
    # Auto-detect format from extension if not specified
    if format_type is None:
        format_type = output_path.suffix.lower().lstrip('.')
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove metadata before saving
    config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
    
    try:
        if format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, sort_keys=True)
        
        elif format_type in ['yaml', 'yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML is required to save YAML configuration")
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_to_save, f, default_flow_style=False, indent=2)
        
        elif format_type == 'toml':
            if not HAS_TOML:
                raise ImportError("toml is required to save TOML configuration")
            with open(output_path, 'w', encoding='utf-8') as f:
                toml.dump(config_to_save, f)
        
        elif format_type == 'cmxconfig':
            _save_cmx_config(config_to_save, output_path)
        
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
        
        logging.info(f"Configuration saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to save configuration: {e}")
        raise


def _save_cmx_config(config: dict, output_path: Path) -> None:
    """Save configuration in CMX format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# CMatrix Configuration File\n")
        f.write(f"# Generated automatically\n\n")
        
        for section_name, section_config in config.items():
            if isinstance(section_config, dict):
                f.write(f"[{section_name}]\n")
                for key, value in sorted(section_config.items()):
                    if isinstance(value, bool):
                        value_str = 'true' if value else 'false'
                    elif isinstance(value, list):
                        value_str = '[' + ', '.join(str(item) for item in value) + ']'
                    elif value is None:
                        value_str = 'null'
                    else:
                        value_str = str(value)
                    
                    f.write(f"{key} = {value_str}\n")
                f.write("\n")
            else:
                # Handle non-section values
                f.write(f"{section_name} = {section_config}\n")


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
    
    Returns:
        dict: Merged configuration
    """
    merged = base_config.copy()
    
    for section_name, section_config in override_config.items():
        if section_name in merged and isinstance(merged[section_name], dict) and isinstance(section_config, dict):
            # Merge section dictionaries
            merged_section = merged[section_name].copy()
            merged_section.update(section_config)
            merged[section_name] = merged_section
        else:
            # Replace entire section
            merged[section_name] = section_config
    
    return merged


def get_config_template(template_type: str = 'default') -> dict:
    """
    Get a configuration template for common use cases.
    
    Args:
        template_type: Type of template ('default', 'gpu', 'quantized', 'minimal')
    
    Returns:
        dict: Configuration template
    """
    templates = {
        'default': {
            'target': {
                'platform': 'cpu',
                'architecture': 'x86_64',
                'optimization_level': 'O2',
                'precision': 'float32'
            },
            'compiler': {
                'backend': 'llvm',
                'debug_info': False,
                'optimization_flags': ['-ffast-math', '-funroll-loops']
            },
            'memory': {
                'allocation_strategy': 'static',
                'max_memory_mb': 1024,
                'enable_memory_pool': True
            },
            'parallelization': {
                'enable_threading': True,
                'max_threads': -1,
                'enable_vectorization': True
            },
            'optimization': {
                'enable_fusion': True,
                'enable_quantization': False
            },
            'output': {
                'format': 'library',
                'output_directory': './output',
                'library_name': 'model'
            }
        },
        
        'gpu': {
            'target': {
                'platform': 'cuda',
                'architecture': 'sm_70',
                'optimization_level': 'O3',
                'precision': 'float32'
            },
            'compiler': {
                'backend': 'nvcc',
                'debug_info': False,
                'optimization_flags': ['-use_fast_math', '--maxrregcount=64']
            },
            'memory': {
                'allocation_strategy': 'dynamic',
                'max_memory_mb': 8192,
                'enable_memory_pool': True,
                'gpu_memory_fraction': 0.8
            },
            'parallelization': {
                'enable_threading': True,
                'enable_gpu': True,
                'gpu_block_size': 256,
                'gpu_grid_size': 'auto'
            },
            'optimization': {
                'enable_fusion': True,
                'enable_quantization': False,
                'enable_tensor_core': True
            },
            'output': {
                'format': 'shared_library',
                'output_directory': './output',
                'library_name': 'model_gpu'
            }
        },
        
        'quantized': {
            'target': {
                'platform': 'cpu',
                'architecture': 'x86_64',
                'optimization_level': 'Os',
                'precision': 'float32'
            },
            'compiler': {
                'backend': 'llvm',
                'debug_info': False,
                'optimization_flags': ['-ffast-math']
            },
            'memory': {
                'allocation_strategy': 'static',
                'max_memory_mb': 256,
                'enable_memory_pool': True
            },
            'parallelization': {
                'enable_threading': True,
                'max_threads': 4,
                'enable_vectorization': True
            },
            'optimization': {
                'enable_fusion': True,
                'enable_quantization': True,
                'quantization_type': 'int8',
                'quantization_mode': 'dynamic'
            },
            'output': {
                'format': 'static_library',
                'output_directory': './output',
                'library_name': 'model_quantized'
            }
        },
        
        'minimal': {
            'target': {
                'platform': 'cpu',
                'optimization_level': 'O2'
            },
            'output': {
                'format': 'library',
                'output_directory': './output'
            }
        }
    }
    
    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}")
    
    return templates[template_type].copy()


def validate_target_compatibility(config: dict) -> List[str]:
    """
    Validate target platform compatibility and return warnings.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List[str]: List of compatibility warnings
    """
    warnings = []
    
    target_config = config.get('target', {})
    platform = target_config.get('platform', 'cpu')
    architecture = target_config.get('architecture', '')
    precision = target_config.get('precision', 'float32')
    
    # Platform-specific checks
    if platform == 'cuda':
        if not architecture.startswith('sm_'):
            warnings.append("CUDA platform requires SM architecture specification (e.g., sm_70)")
        
        # Check precision compatibility
        tensor_core_precision = ['float16', 'bfloat16', 'int8']
        if precision not in tensor_core_precision and config.get('optimization', {}).get('enable_tensor_core'):
            warnings.append(f"Tensor Core optimization requires {tensor_core_precision} precision, got {precision}")
    
    elif platform == 'opencl':
        if precision == 'float64':
            warnings.append("Double precision may not be supported on all OpenCL devices")
    
    elif platform == 'cpu':
        if architecture == 'arm64' and precision == 'float64':
            warnings.append("ARM64 may have limited float64 performance compared to float32")
    
    # Memory checks
    memory_config = config.get('memory', {})
    max_memory = memory_config.get('max_memory_mb', 1024)
    
    if platform == 'gpu' and max_memory > 16384:
        warnings.append("GPU memory allocation >16GB may not be supported on all devices")
    
    # Optimization compatibility
    opt_config = config.get('optimization', {})
    if opt_config.get('enable_quantization') and precision not in ['float32', 'float16']:
        warnings.append(f"Quantization from {precision} may not be optimal")
    
    return warnings


def create_config_from_model(model_info: dict, target_platform: str = 'cpu') -> dict:
    """
    Create a configuration template based on model characteristics.
    
    Args:
        model_info: Model information from inspector
        target_platform: Target platform ('cpu', 'gpu', 'cuda')
    
    Returns:
        dict: Generated configuration
    """
    # Start with appropriate template
    if target_platform in ['gpu', 'cuda']:
        config = get_config_template('gpu')
    else:
        config = get_config_template('default')
    
    # Adjust based on model characteristics
    complexity = model_info.get('complexity', {})
    structure = model_info.get('structure', {})
    tensors = model_info.get('tensors', {})
    
    total_macs = complexity.get('total_macs', 0)
    memory_mb = complexity.get('memory_mb', 0)
    total_params = tensors.get('total_parameters', 0)
    
    # Adjust optimization level based on model complexity
    if total_macs > 1e9:  # > 1 GMAC
        config['target']['optimization_level'] = 'O3'
        config['optimization']['enable_fusion'] = True
        
        if total_params > 1e6:  # > 1M parameters
            config['optimization']['enable_quantization'] = True
            config['optimization']['quantization_type'] = 'int8'
    
    # Adjust memory settings
    estimated_memory = max(memory_mb * 2, 512)  # 2x model memory + minimum
    config['memory']['max_memory_mb'] = int(estimated_memory)
    
    # Adjust parallelization based on model depth
    depth = complexity.get('depth', 0)
    if depth > 100:
        config['parallelization']['max_threads'] = -1  # Use all available
    elif depth > 50:
        config['parallelization']['max_threads'] = 8
    else:
        config['parallelization']['max_threads'] = 4
    
    # Adjust compiler flags based on layer types
    layer_types = structure.get('layer_types', {})
    if 'Conv' in layer_types or 'Convolution' in layer_types:
        config['compiler']['optimization_flags'].append('-ffast-math')
    
    if 'MatMul' in layer_types or 'Gemm' in layer_types:
        config['compiler']['optimization_flags'].append('-march=native')
    
    return config


def export_config_documentation(config: dict, output_path: str) -> None:
    """
    Export configuration documentation in markdown format.
    
    Args:
        config: Configuration dictionary
        output_path: Output markdown file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# CMatrix Configuration Documentation\n\n")
        f.write("This document describes the current configuration settings.\n\n")
        
        for section_name, section_config in config.items():
            if section_name.startswith('_'):
                continue
            
            f.write(f"## {section_name.title()} Configuration\n\n")
            
            if isinstance(section_config, dict):
                for key, value in sorted(section_config.items()):
                    f.write(f"- **{key}**: `{value}`\n")
                    
                    # Add descriptions for common settings
                    if key in _get_config_descriptions():
                        f.write(f"  - {_get_config_descriptions()[key]}\n")
                
                f.write("\n")
            else:
                f.write(f"Value: `{section_config}`\n\n")
        
        # Add validation results if available
        if '_metadata' in config:
            f.write("## Metadata\n\n")
            metadata = config['_metadata']
            for key, value in metadata.items():
                f.write(f"- **{key}**: `{value}`\n")


def _get_config_descriptions() -> dict:
    """Get descriptions for configuration options."""
    return {
        'platform': 'Target execution platform (cpu, gpu, cuda, opencl)',
        'optimization_level': 'Compiler optimization level (O0=none, O3=aggressive)',
        'precision': 'Numerical precision for computations',
        'debug_info': 'Include debugging information in compiled output',
        'max_memory_mb': 'Maximum memory allocation in megabytes',
        'enable_threading': 'Enable multi-threading support',
        'enable_fusion': 'Enable operation fusion optimizations',
        'enable_quantization': 'Enable model quantization',
        'output_directory': 'Directory for generated output files',
        'library_name': 'Name of the generated library'
    }


def check_config_updates(config_path: str) -> dict:
    """
    Check if configuration file has been updated since last parse.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        dict: Update status information
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        return {'exists': False, 'updated': False}
    
    # Get file modification time
    mtime = config_path.stat().st_mtime
    
    # Store/retrieve last check time (in practice, this would be persistent)
    # For now, just return current status
    return {
        'exists': True,
        'updated': True,  # Assume updated for now
        'modification_time': mtime,
        'size_bytes': config_path.stat().st_size
    }
