"""
CMatrix Tools Utilities Package

Lightweight utility modules for configuration handling, file management,
logging, data I/O, and model inspection within the CMatrix toolchain.
"""

from .prompt_gen import generate_op_prompt, generate_config_prompt
from .inspector import inspect_model, get_model_summary
from .model_analyzer import analyze_model, estimate_complexity
from .data_loader import load_data, load_test_data, save_data
from .config_parser import parse_config, validate_config
from .file_utils import ensure_dir_exists, get_temp_path, cleanup_temp_files
from .logging_utils import setup_logger, get_logger

__version__ = "0.1.0"
__author__ = "CMatrix Tools Team"

__all__ = [
    # Prompt generation
    "generate_op_prompt",
    "generate_config_prompt",
    
    # Model inspection
    "inspect_model",
    "get_model_summary",
    
    # Model analysis
    "analyze_model",
    "estimate_complexity",
    
    # Data handling
    "load_data",
    "load_test_data",
    "save_data",
    
    # Configuration
    "parse_config",
    "validate_config",
    
    # File utilities
    "ensure_dir_exists",
    "get_temp_path",
    "cleanup_temp_files",
    
    # Logging
    "setup_logger",
    "get_logger",
]

