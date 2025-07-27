"""
Data loading utilities for testing and benchmarking.

Provides functions to load sample data (images, audio, text) during
model testing, validation, and benchmarking processes.
"""

import os
import json
import struct
import logging
from typing import Union, Optional, Tuple, List, Any
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_data(file_path: str, dtype: str = 'float32') -> np.ndarray:
    """
    Load input data from file (binary, CSV, image, or numpy format).
    
    Args:
        file_path: Path to the data file
        dtype: Target data type ('float32', 'float64', 'int32', etc.)
    
    Returns:
        np.ndarray: Loaded data array
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    try:
        if extension == '.npy':
            return _load_numpy_file(file_path, dtype)
        elif extension == '.npz':
            return _load_numpy_archive(file_path, dtype)
        elif extension in ['.csv', '.txt']:
            return _load_text_file(file_path, dtype)
        elif extension == '.bin':
            return _load_binary_file(file_path, dtype)
        elif extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return _load_image_file(file_path, dtype)
        elif extension == '.json':
            return _load_json_file(file_path, dtype)
        else:
            # Try to infer format from content
            return _load_auto_detect(file_path, dtype)
            
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise ValueError(f"Unable to load data from {file_path}: {e}")


def load_test_data(data_type: str, shape: Tuple[int, ...], 
                   dtype: str = 'float32', **kwargs) -> np.ndarray:
    """
    Generate test data for model validation and benchmarking.
    
    Args:
        data_type: Type of test data ('random', 'zeros', 'ones', 'range', 'gaussian')
        shape: Shape of the output tensor
        dtype: Data type for the array
        **kwargs: Additional parameters for data generation
    
    Returns:
        np.ndarray: Generated test data
    """
    np_dtype = np.dtype(dtype)
    
    if data_type == 'random':
        low = kwargs.get('low', 0.0)
        high = kwargs.get('high', 1.0)
        return np.random.uniform(low, high, shape).astype(np_dtype)
    
    elif data_type == 'gaussian':
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 1.0)
        return np.random.normal(mean, std, shape).astype(np_dtype)
    
    elif data_type == 'zeros':
        return np.zeros(shape, dtype=np_dtype)
    
    elif data_type == 'ones':
        return np.ones(shape, dtype=np_dtype)
    
    elif data_type == 'range':
        start = kwargs.get('start', 0)
        return np.arange(start, start + np.prod(shape)).reshape(shape).astype(np_dtype)
    
    elif data_type == 'constant':
        value = kwargs.get('value', 1.0)
        return np.full(shape, value, dtype=np_dtype)
    
    elif data_type == 'image_like':
        # Generate image-like data (normalized to 0-1 or 0-255)
        normalize = kwargs.get('normalize', True)
        data = np.random.randint(0, 256, shape, dtype=np.uint8)
        if normalize:
            return data.astype(np_dtype) / 255.0
        return data.astype(np_dtype)
    
    else:
        raise ValueError(f"Unsupported test data type: {data_type}")


def save_data(data: np.ndarray, file_path: str, format_type: Optional[str] = None) -> None:
    """
    Save data array to file in specified format.
    
    Args:
        data: NumPy array to save
        file_path: Output file path
        format_type: Format type ('npy', 'csv', 'bin', 'json'). Auto-detected if None.
    """
    file_path = Path(file_path)
    
    # Auto-detect format from extension if not specified
    if format_type is None:
        format_type = file_path.suffix.lower().lstrip('.')
    
    # Ensure output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format_type == 'npy':
            np.save(file_path, data)
        
        elif format_type == 'npz':
            np.savez_compressed(file_path, data=data)
        
        elif format_type == 'csv':
            if data.ndim > 2:
                # Flatten high-dimensional arrays for CSV
                data_2d = data.reshape(data.shape[0], -1)
                np.savetxt(file_path, data_2d, delimiter=',', fmt='%.6f')
            else:
                np.savetxt(file_path, data, delimiter=',', fmt='%.6f')
        
        elif format_type == 'bin':
            data.astype(np.float32).tobytes()
            with open(file_path, 'wb') as f:
                f.write(data.tobytes())
        
        elif format_type == 'json':
            data_list = data.tolist()
            with open(file_path, 'w') as f:
                json.dump({
                    'data': data_list,
                    'shape': list(data.shape),
                    'dtype': str(data.dtype)
                }, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported save format: {format_type}")
            
        logging.info(f"Data saved to {file_path} in {format_type} format")
        
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")
        raise


def _load_numpy_file(file_path: Path, dtype: str) -> np.ndarray:
    """Load NumPy .npy file."""
    data = np.load(file_path)
    return data.astype(dtype)


def _load_numpy_archive(file_path: Path, dtype: str) -> np.ndarray:
    """Load NumPy .npz archive file."""
    archive = np.load(file_path)
    
    # If multiple arrays, take the first one or look for 'data' key
    if 'data' in archive:
        data = archive['data']
    else:
        keys = list(archive.keys())
        if keys:
            data = archive[keys[0]]
            logging.warning(f"Multiple arrays in archive, using '{keys[0]}'")
        else:
            raise ValueError("Empty numpy archive")
    
    return data.astype(dtype)


def _load_text_file(file_path: Path, dtype: str) -> np.ndarray:
    """Load CSV or text file."""
    try:
        # Try comma-separated first
        data = np.loadtxt(file_path, delimiter=',', dtype=dtype)
    except ValueError:
        try:
            # Try space/tab separated
            data = np.loadtxt(file_path, dtype=dtype)
        except ValueError:
            # Try as single column
            data = np.loadtxt(file_path, dtype=dtype, ndmin=1)
    
    return data


def _load_binary_file(file_path: Path, dtype: str, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Load binary file."""
    np_dtype = np.dtype(dtype)
    
    with open(file_path, 'rb') as f:
        data_bytes = f.read()
    
    # Convert bytes to numpy array
    data = np.frombuffer(data_bytes, dtype=np_dtype)
    
    if shape is not None:
        data = data.reshape(shape)
    
    return data


def _load_image_file(file_path: Path, dtype: str) -> np.ndarray:
    """Load image file using PIL or OpenCV."""
    if HAS_PIL:
        return _load_image_pil(file_path, dtype)
    elif HAS_CV2:
        return _load_image_cv2(file_path, dtype)
    else:
        raise ImportError("Neither PIL nor OpenCV available for image loading")


def _load_image_pil(file_path: Path, dtype: str) -> np.ndarray:
    """Load image using PIL."""
    image = Image.open(file_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    data = np.array(image, dtype=dtype)
    
    # Normalize to 0-1 range if float type
    if np.issubdtype(np.dtype(dtype), np.floating):
        data = data / 255.0
    
    return data


def _load_image_cv2(file_path: Path, dtype: str) -> np.ndarray:
    """Load image using OpenCV."""
    # OpenCV loads as BGR by default
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"Could not load image: {file_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to specified dtype
    data = image.astype(dtype)
    
    # Normalize to 0-1 range if float type
    if np.issubdtype(np.dtype(dtype), np.floating):
        data = data / 255.0
    
    return data


def _load_json_file(file_path: Path, dtype: str) -> np.ndarray:
    """Load JSON file containing array data."""
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(json_data, dict):
        if 'data' in json_data:
            array_data = json_data['data']
            shape = json_data.get('shape')
        elif 'array' in json_data:
            array_data = json_data['array']
            shape = json_data.get('shape')
        else:
            # Assume the entire dict contains the data
            array_data = json_data
            shape = None
    else:
        # Direct array data
        array_data = json_data
        shape = None
    
    # Convert to numpy array
    data = np.array(array_data, dtype=dtype)
    
    # Reshape if shape information is available
    if shape is not None:
        data = data.reshape(shape)
    
    return data


def _load_auto_detect(file_path: Path, dtype: str) -> np.ndarray:
    """Auto-detect file format and load accordingly."""
    # Try different loaders in order of likelihood
    loaders = [
        ('numpy', _load_numpy_file),
        ('text', _load_text_file),
        ('json', _load_json_file),
        ('binary', _load_binary_file)
    ]
    
    for loader_name, loader_func in loaders:
        try:
            return loader_func(file_path, dtype)
        except Exception:
            continue
    
    raise ValueError(f"Could not auto-detect format for {file_path}")


def load_batch_data(file_paths: List[str], batch_size: int = 32, 
                   dtype: str = 'float32', **kwargs) -> List[np.ndarray]:
    """
    Load multiple data files as batches.
    
    Args:
        file_paths: List of file paths to load
        batch_size: Number of samples per batch
        dtype: Target data type
        **kwargs: Additional arguments for load_data
    
    Returns:
        List[np.ndarray]: List of batched arrays
    """
    batches = []
    current_batch = []
    
    for file_path in file_paths:
        try:
            data = load_data(file_path, dtype=dtype, **kwargs)
            current_batch.append(data)
            
            if len(current_batch) >= batch_size:
                # Stack current batch
                batch_array = np.stack(current_batch, axis=0)
                batches.append(batch_array)
                current_batch = []
                
        except Exception as e:
            logging.warning(f"Skipping {file_path} due to error: {e}")
            continue
    
    # Handle remaining samples
    if current_batch:
        batch_array = np.stack(current_batch, axis=0)
        batches.append(batch_array)
    
    return batches


def create_dataset_manifest(data_dir: str, output_path: str, 
                          file_extensions: Optional[List[str]] = None) -> None:
    """
    Create a manifest file listing all data files in a directory.
    
    Args:
        data_dir: Directory containing data files
        output_path: Path for output manifest file
        file_extensions: List of file extensions to include (e.g., ['.npy', '.csv'])
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Default extensions if not specified
    if file_extensions is None:
        file_extensions = ['.npy', '.npz', '.csv', '.txt', '.bin', '.json', 
                          '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Find all matching files
    manifest_data = {
        'data_directory': str(data_dir),
        'created_at': str(np.datetime64('now')),
        'files': []
    }
    
    for ext in file_extensions:
        pattern = f"*{ext}"
        matching_files = list(data_dir.glob(pattern))
        
        for file_path in matching_files:
            try:
                # Get basic file info
                stat_info = file_path.stat()
                file_info = {
                    'path': str(file_path.relative_to(data_dir)),
                    'absolute_path': str(file_path),
                    'size_bytes': stat_info.st_size,
                    'extension': ext,
                    'modified_time': stat_info.st_mtime
                }
                
                # Try to get data shape information
                try:
                    if ext == '.npy':
                        data = np.load(file_path, mmap_mode='r')
                        file_info['shape'] = list(data.shape)
                        file_info['dtype'] = str(data.dtype)
                    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] and HAS_PIL:
                        image = Image.open(file_path)
                        file_info['shape'] = [image.height, image.width, len(image.getbands())]
                        file_info['dtype'] = 'uint8'
                except Exception:
                    # Skip shape info if we can't determine it
                    pass
                
                manifest_data['files'].append(file_info)
                
            except Exception as e:
                logging.warning(f"Skipping {file_path}: {e}")
                continue
    
    # Sort files by path for consistency
    manifest_data['files'].sort(key=lambda x: x['path'])
    manifest_data['total_files'] = len(manifest_data['files'])
    
    # Save manifest
    with open(output_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    logging.info(f"Created manifest with {len(manifest_data['files'])} files: {output_path}")


def load_from_manifest(manifest_path: str, indices: Optional[List[int]] = None,
                      dtype: str = 'float32') -> List[np.ndarray]:
    """
    Load data files listed in a manifest.
    
    Args:
        manifest_path: Path to manifest JSON file
        indices: List of file indices to load (None = load all)
        dtype: Target data type
    
    Returns:
        List[np.ndarray]: Loaded data arrays
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    data_dir = Path(manifest['data_directory'])
    files_info = manifest['files']
    
    # Select files to load
    if indices is not None:
        files_to_load = [files_info[i] for i in indices if i < len(files_info)]
    else:
        files_to_load = files_info
    
    loaded_data = []
    for file_info in files_to_load:
        file_path = data_dir / file_info['path']
        try:
            data = load_data(str(file_path), dtype=dtype)
            loaded_data.append(data)
        except Exception as e:
            logging.warning(f"Failed to load {file_path}: {e}")
            continue
    
    return loaded_data


def validate_data_format(data: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None,
                        expected_dtype: Optional[str] = None,
                        value_range: Optional[Tuple[float, float]] = None) -> bool:
    """
    Validate data array against expected format.
    
    Args:
        data: NumPy array to validate
        expected_shape: Expected shape (None values are wildcards)
        expected_dtype: Expected data type
        value_range: Expected value range (min, max)
    
    Returns:
        bool: True if data is valid
    """
    try:
        # Check shape
        if expected_shape is not None:
            if len(data.shape) != len(expected_shape):
                logging.error(f"Shape dimension mismatch: {data.shape} vs {expected_shape}")
                return False
            
            for actual, expected in zip(data.shape, expected_shape):
                if expected is not None and actual != expected:
                    logging.error(f"Shape mismatch: {data.shape} vs {expected_shape}")
                    return False
        
        # Check data type
        if expected_dtype is not None:
            if str(data.dtype) != expected_dtype:
                logging.error(f"Data type mismatch: {data.dtype} vs {expected_dtype}")
                return False
        
        # Check value range
        if value_range is not None:
            min_val, max_val = value_range
            data_min, data_max = np.min(data), np.max(data)
            
            if data_min < min_val or data_max > max_val:
                logging.error(f"Value range violation: [{data_min}, {data_max}] vs [{min_val}, {max_val}]")
                return False
        
        # Check for invalid values
        if np.issubdtype(data.dtype, np.floating):
            if np.any(np.isnan(data)):
                logging.error("Data contains NaN values")
                return False
            
            if np.any(np.isinf(data)):
                logging.error("Data contains infinite values")
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"Data validation failed: {e}")
        return False


def convert_data_format(input_path: str, output_path: str, 
                       target_format: str, target_dtype: str = 'float32',
                       **kwargs) -> None:
    """
    Convert data from one format to another.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        target_format: Target format ('npy', 'csv', 'bin', 'json')
        target_dtype: Target data type
        **kwargs: Additional conversion options
    """
    # Load data in original format
    data = load_data(input_path, dtype=target_dtype)
    
    # Apply any transformations
    if 'normalize' in kwargs and kwargs['normalize']:
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
    
    if 'reshape' in kwargs:
        new_shape = kwargs['reshape']
        data = data.reshape(new_shape)
    
    if 'transpose' in kwargs:
        axes = kwargs['transpose']
        data = np.transpose(data, axes)
    
    # Save in target format
    save_data(data, output_path, target_format)
    
    logging.info(f"Converted {input_path} to {output_path} in {target_format} format")


def get_data_statistics(data: np.ndarray) -> dict:
    """
    Compute basic statistics for data array.
    
    Args:
        data: NumPy array to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        'shape': list(data.shape),
        'dtype': str(data.dtype),
        'size': data.size,
        'memory_mb': data.nbytes / (1024 * 1024)
    }
    
    # Numerical statistics for numeric types
    if np.issubdtype(data.dtype, np.number):
        stats.update({
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data))
        })
        
        # Check for special values
        if np.issubdtype(data.dtype, np.floating):
            stats.update({
                'nan_count': int(np.sum(np.isnan(data))),
                'inf_count': int(np.sum(np.isinf(data))),
                'finite_count': int(np.sum(np.isfinite(data)))
            })
    
    return stats