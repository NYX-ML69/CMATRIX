"""
cmx_tools.compile - Model compilation module for embedded targets

This module provides tools for compiling high-level models into optimized 
C/C++ code tailored for embedded targets like Cortex-M, RISC-V, and Xtensa.
"""

from .cmx_gen import compile_model
from .backend_flags import get_backend_flags
from .code_generator import generate_code
from .optimization_passes import optimize_graph
from .quantization_engine import quantize

__version__ = "1.0.0"
__all__ = [
    "compile_model",
    "get_backend_flags", 
    "generate_code",
    "optimize_graph",
    "quantize"
]

