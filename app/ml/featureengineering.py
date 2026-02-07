import numpy as np
from contextlib import contextmanager
import gc
import logging
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1000  # Process data in 1000-row chunks to limit memory
MAX_ARRAY_SIZE = 9_000_000_000  # 9GB max array size threshold

@contextmanager
def managed_numpy_array(array: np.ndarray) -> Iterator[np.ndarray]:
    """Context manager to properly clean up numpy arrays"""
    try:
        yield array
    finally:
        # Explicitly delete array and run garbage collection
        del array
        gc.collect()

def chunk_array(array: np.ndarray, chunk_size: int = CHUNK_SIZE) -> Iterator[np.ndarray]:
    """Generator to process large arrays in chunks"""
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]

class FeatureEngineering:
    def __init__(self):
        self._features = None
        
    def engineer_features(self, input_data: np.ndarray) -> np.ndarray:
        """
        Main feature engineering pipeline with memory management
        """
        try:
            # Validate input size
            if input_data.nbytes > MAX_ARRAY_SIZE:
                raise ValueError(f"Input array too large: {input_data.nbytes} bytes")
            
            processed_chunks = []
            
            # Process in chunks to limit memory usage
            for chunk in chunk_array(input_data):
                with managed_numpy_array(chunk) as managed_chunk:
                    # Feature engineering operations
                    processed = self._process_chunk(managed_chunk)
                    processed_chunks.append(processed)
            
            # Combine processed chunks
            with managed_numpy_array(np.concatenate(processed_chunks)) as result:
                self._features = result.copy()
                
            # Clean up chunk list
            processed_chunks.clear()
            
            return self._features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
            
    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process individual chunks with memory cleanup"""
        try:
            # Example feature engineering operations
            with managed_numpy_array(chunk * 2) as scaled:
                with managed_numpy_array(np.log1p(scaled)) as logged:
                    return logged.copy()
                    
        except Exception as e:
            logger.error(f"Chunk processing failed: {str(e)}")
            raise
            
    def get_features(self) -> Optional[np.ndarray]:
        """Safe accessor for engineered features"""
        return self._features
        
    def clear_features(self):
        """Explicitly clear stored features"""
        if self._features is not None:
            del self._features
            self._features = None
            gc.collect()
