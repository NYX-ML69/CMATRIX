"""
File utilities for CMatrix toolchain.

Handles path resolution, temp file creation, directory traversal,
and other file system operations needed across cmx_tools.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def ensure_dir_exists(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Raises:
        OSError: If directory creation fails
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    """Resolve relative path to absolute path.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        base_dir: Base directory for relative paths (defaults to cwd)
        
    Returns:
        Absolute path string
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj.resolve())
    else:
        return str((Path(base_dir) / path).resolve())


def create_temp_dir(prefix: str = "cmx_temp_") -> str:
    """Create temporary directory for processing.
    
    Args:
        prefix: Prefix for temp directory name
        
    Returns:
        Path to created temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir


def cleanup_temp_dir(temp_dir: str) -> None:
    """Remove temporary directory and all contents.
    
    Args:
        temp_dir: Path to temporary directory to remove
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
    except OSError as e:
        logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")


def list_files_with_extension(directory: str, extension: str, recursive: bool = False) -> List[str]:
    """List all files with specific extension in directory.
    
    Args:
        directory: Directory to search
        extension: File extension to filter (with or without dot)
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths matching the extension
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    directory_path = Path(directory)
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    pattern = f"**/*{extension}" if recursive else f"*{extension}"
    files = [str(f) for f in directory_path.glob(pattern) if f.is_file()]
    
    logger.debug(f"Found {len(files)} files with extension {extension} in {directory}")
    return files


def copy_file(src: str, dst: str, create_dirs: bool = True) -> None:
    """Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create destination directories if they don't exist
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If copy operation fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if create_dirs:
        ensure_dir_exists(str(dst_path.parent))
    
    try:
        shutil.copy2(src, dst)
        logger.debug(f"Copied file from {src} to {dst}")
    except OSError as e:
        logger.error(f"Failed to copy file from {src} to {dst}: {e}")
        raise


def get_file_size(file_path: str) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size = path.stat().st_size
    logger.debug(f"File {file_path} size: {size} bytes")
    return size


def is_file_newer(file1: str, file2: str) -> bool:
    """Check if file1 is newer than file2 based on modification time.
    
    Args:
        file1: First file path
        file2: Second file path
        
    Returns:
        True if file1 is newer than file2, False otherwise
        
    Raises:
        FileNotFoundError: If either file doesn't exist
    """
    path1 = Path(file1)
    path2 = Path(file2)
    
    if not path1.exists():
        raise FileNotFoundError(f"File not found: {file1}")
    if not path2.exists():
        raise FileNotFoundError(f"File not found: {file2}")
    
    return path1.stat().st_mtime > path2.stat().st_mtime


def safe_remove_file(file_path: str) -> bool:
    """Safely remove a file without raising exceptions.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if file was removed or didn't exist, False if removal failed
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Removed file: {file_path}")
        return True
    except OSError as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")
        return False


def find_config_file(start_dir: str, config_names: List[str]) -> Optional[str]:
    """Find configuration file by searching up directory tree.
    
    Args:
        start_dir: Directory to start search from
        config_names: List of config file names to search for
        
    Returns:
        Path to first config file found, or None if not found
    """
    current_dir = Path(start_dir).resolve()
    
    while current_dir != current_dir.parent:  # Not at root
        for config_name in config_names:
            config_path = current_dir / config_name
            if config_path.is_file():
                logger.debug(f"Found config file: {config_path}")
                return str(config_path)
        current_dir = current_dir.parent
    
    logger.debug(f"No config file found starting from {start_dir}")
    return None


def normalize_path_separators(path: str) -> str:
    """Normalize path separators for current OS.
    
    Args:
        path: Path with potentially mixed separators
        
    Returns:
        Path with normalized separators
    """
    return str(Path(path))
