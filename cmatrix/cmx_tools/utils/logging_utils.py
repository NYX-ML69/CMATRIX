"""
Logging utilities for CMatrix toolchain.

Centralized logging setup for CLI tools with consistent formatting
and appropriate output levels for different use cases.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for terminal output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset to default
    }
    
    def format(self, record):
        """Format log record with colors if outputting to terminal."""
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            # Add color to level name
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str = "cmx_tools",
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[str] = None,
    file_level: str = "DEBUG"
) -> logging.Logger:
    """Initialize logger with CLI-friendly output.
    
    Args:
        name: Logger name
        verbose: Enable debug level logging to console
        quiet: Suppress all console output except errors
        log_file: Optional file path for logging
        file_level: Log level for file output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    logger.propagate = False
    
    # Set base logger level to DEBUG so handlers can filter
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stderr)
        
        # Set console log level based on verbosity
        if verbose:
            console_level = logging.DEBUG
            console_format = '[%(levelname)s] %(name)s:%(lineno)d - %(message)s'
        else:
            console_level = logging.INFO
            console_format = '[%(levelname)s] %(message)s'
        
        console_handler.setLevel(console_level)
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, file_level.upper()))
            
            file_format = '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
            file_formatter = logging.Formatter(file_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except (OSError, PermissionError) as e:
            # Log to console if file logging fails
            logger.error(f"Failed to setup file logging to {log_file}: {e}")
    
    return logger


def get_progress_logger(name: str = "progress") -> logging.Logger:
    """Get a logger specifically for progress reporting.
    
    Args:
        name: Logger name for progress reporting
        
    Returns:
        Logger configured for progress output
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    # Simple handler for progress updates
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log basic system information for debugging.
    
    Args:
        logger: Logger instance to use
    """
    import platform
    import sys
    
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {platform.platform()}")
    logger.debug(f"Architecture: {platform.machine()}")
    logger.debug(f"Processor: {platform.processor()}")


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """Log exception with context information.
    
    Args:
        logger: Logger instance to use
        exc: Exception to log
        context: Additional context about where exception occurred
    """
    context_msg = f" in {context}" if context else ""
    logger.error(f"Exception{context_msg}: {type(exc).__name__}: {exc}")
    logger.debug("Exception details:", exc_info=True)


def create_benchmark_logger(output_file: str) -> logging.Logger:
    """Create a logger specifically for benchmark results.
    
    Args:
        output_file: File path for benchmark output
        
    Returns:
        Logger configured for benchmark data
    """
    logger = logging.getLogger("benchmark")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    try:
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # File handler for benchmark data
        handler = logging.FileHandler(output_file)
        handler.setLevel(logging.INFO)
        
        # Simple format for benchmark data
        formatter = logging.Formatter('%(asctime)s,%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    except (OSError, PermissionError) as e:
        # Fallback to console if file fails
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[BENCHMARK] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.error(f"Failed to setup benchmark file logging: {e}")
    
    return logger


def set_third_party_log_levels(level: str = "WARNING") -> None:
    """Set log levels for common third-party libraries to reduce noise.
    
    Args:
        level: Log level to set for third-party libraries
    """
    third_party_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'tensorflow',
        'torch',
        'onnx'
    ]
    
    log_level = getattr(logging, level.upper())
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(log_level)


def setup_cli_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Convenience function to setup logging for CLI applications.
    
    Args:
        verbose: Enable debug output
        quiet: Suppress non-error output
        log_file: Optional log file path
        
    Returns:
        Configured main logger
    """
    # Setup main logger
    logger = setup_logger(
        name="cmx_tools",
        verbose=verbose,
        quiet=quiet,
        log_file=log_file
    )
    
    # Reduce third-party library noise
    set_third_party_log_levels("WARNING")
    
    # Log system info in debug mode
    if verbose:
        log_system_info(logger)
    
    return logger

