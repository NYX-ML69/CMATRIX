"""
backend_flags.py - Target-specific flags and backend configurations

Manages configuration settings for different embedded targets.
"""

from typing import Dict, Any


# Backend configuration definitions
BACKEND_CONFIGS = {
    'cortex-m': {
        'instruction_set': 'arm',
        'architecture': 'cortex-m',
        'vector_width': 128,  # NEON when available
        'alignment': 4,
        'word_size': 32,
        'endianness': 'little',
        'has_fpu': True,
        'has_dsp': True,
        'memory_model': 'harvard',
        'stack_size': 8192,
        'heap_size': 16384,
        'compiler_flags': ['-mcpu=cortex-m4', '-mthumb', '-mfpu=fpv4-sp-d16'],
        'defines': ['ARM_MATH_CM4', 'CORTEX_M'],
        'includes': ['arm_math.h', 'cmsis_os.h'],
        'template': 'cortex_m_template.cpp'
    },
    
    'riscv': {
        'instruction_set': 'riscv',
        'architecture': 'rv32i',
        'vector_width': 256,  # RVV when available
        'alignment': 4,
        'word_size': 32,
        'endianness': 'little',
        'has_fpu': True,
        'has_dsp': False,
        'memory_model': 'von_neumann',
        'stack_size': 4096,
        'heap_size': 8192,
        'compiler_flags': ['-march=rv32imf', '-mabi=ilp32f'],
        'defines': ['RISCV', 'RV32I'],
        'includes': ['riscv_vector.h'],
        'template': 'riscv_template.cpp'
    },
    
    'xtensa': {
        'instruction_set': 'xtensa',
        'architecture': 'lx6',
        'vector_width': 128,
        'alignment': 4,
        'word_size': 32,
        'endianness': 'little',
        'has_fpu': True,
        'has_dsp': True,
        'memory_model': 'harvard',
        'stack_size': 8192,
        'heap_size': 32768,
        'compiler_flags': ['-mlongcalls'],
        'defines': ['XTENSA', 'ESP32'],
        'includes': ['esp_system.h', 'xtensa/hal.h'],
        'template': 'xtensa_template.cpp'
    },
    
    'generic': {
        'instruction_set': 'generic',
        'architecture': 'generic',
        'vector_width': 128,
        'alignment': 8,
        'word_size': 32,
        'endianness': 'little',
        'has_fpu': True,
        'has_dsp': False,
        'memory_model': 'von_neumann',
        'stack_size': 16384,
        'heap_size': 65536,
        'compiler_flags': ['-O2'],
        'defines': ['GENERIC'],
        'includes': ['iostream', 'cstdint'],
        'template': 'generic_template.cpp'
    }
}


def get_backend_flags(target_name: str) -> Dict[str, Any]:
    """
    Get backend configuration for specified target.
    
    Args:
        target_name: Name of target backend
        
    Returns:
        Dict containing backend configuration
        
    Raises:
        ValueError: If target_name is not supported
    """
    if target_name not in BACKEND_CONFIGS:
        available = list(BACKEND_CONFIGS.keys())
        raise ValueError(f"Unknown target '{target_name}'. Available: {available}")
    
    return BACKEND_CONFIGS[target_name].copy()


def list_backends() -> list:
    """Return list of supported backend names."""
    return list(BACKEND_CONFIGS.keys())


def get_compiler_flags(target_name: str) -> list:
    """Get compiler flags for target."""
    config = get_backend_flags(target_name)
    return config.get('compiler_flags', [])


def get_defines(target_name: str) -> list:
    """Get preprocessor defines for target."""
    config = get_backend_flags(target_name)
    return config.get('defines', [])


def get_includes(target_name: str) -> list:
    """Get required include headers for target."""
    config = get_backend_flags(target_name)
    return config.get('includes', [])


def get_template_name(target_name: str) -> str:
    """Get template filename for target."""
    config = get_backend_flags(target_name)
    return config.get('template', 'generic_template.cpp')


def has_vector_support(target_name: str) -> bool:
    """Check if target supports vector operations."""
    config = get_backend_flags(target_name)
    return config.get('vector_width', 0) > 0


def get_memory_constraints(target_name: str) -> Dict[str, int]:
    """Get memory constraints for target."""
    config = get_backend_flags(target_name)
    return {
        'stack_size': config.get('stack_size', 4096),
        'heap_size': config.get('heap_size', 8192),
        'alignment': config.get('alignment', 4)
    }

