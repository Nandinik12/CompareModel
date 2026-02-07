import gc
import contextlib
import numpy as np
from typing import Optional, Generator, Any

# Constants for memory management
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for processing
MAX_ARRAY_SIZE = 1024 * 1024 * 1024 * 8  # 8GB max array size

@contextlib.contextmanager
def limit_memory_context(max_size: Optional[int] = None):
    """Context manager to ensure memory is properly released after operations.
    
    Args:
        max_size: Optional maximum array size in bytes. Defaults to 8GB if not specified.
    """
    try:
        yield
    finally:
        # Force garbage collection
        gc.collect()

def chunk_array(array: np.ndarray, chunk_size: int = CHUNK_SIZE) -> Generator[np.ndarray, None, None]:
    """Split large arrays into smaller chunks for processing.
    
    Args:
        array: Input numpy array to chunk
        chunk_size: Size of chunks in number of elements
        
    Yields:
        Generator of array chunks
    """
    total_size = len(array)
    for i in range(0, total_size, chunk_size):
        yield array[i:min(i + chunk_size, total_size)]

def clear_array(array: np.ndarray) -> None:
    """Explicitly clear numpy array from memory.
    
    Args:
        array: Numpy array to clear
    """
    if array is not None:
        array.fill(0)
        del array
        gc.collect()

def validate_array_size(array: np.ndarray, max_size: Optional[int] = None) -> bool:
    """Validate array size is within acceptable limits.
    
    Args:
        array: Array to validate
        max_size: Optional maximum size in bytes
        
    Returns:
        bool: True if valid, False if too large
        
    Raises:
        ValueError: If array size exceeds maximum
    """
    array_size = array.nbytes
    max_allowed = max_size or MAX_ARRAY_SIZE
    
    if array_size > max_allowed:
        raise ValueError(f"Array size {array_size} bytes exceeds maximum allowed {max_allowed}")
    return True

def process_large_array(array: np.ndarray, 
                       processing_func: Any,
                       chunk_size: int = CHUNK_SIZE,
                       max_size: Optional[int] = None) -> np.ndarray:
    """Process large arrays in chunks with proper memory management.
    
    Args:
        array: Input array to process
        processing_func: Function to apply to chunks
        chunk_size: Size of chunks to process
        max_size: Optional maximum array size
        
    Returns:
        Processed array
    """
    validate_array_size(array, max_size)
    
    results = []
    with limit_memory_context(max_size):
        for chunk in chunk_array(array, chunk_size):
            try:
                processed = processing_func(chunk)
                results.append(processed)
            except Exception as e:
                clear_array(chunk)
                raise RuntimeError(f"Error processing chunk: {str(e)}")
            finally:
                clear_array(chunk)
                
        try:
            final_result = np.concatenate(results)
            return final_result
        finally:
            for result in results:
                clear_array(result)
            results.clear()
