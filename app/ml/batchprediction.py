import numpy as np
import gc
from contextlib import contextmanager
from typing import Iterator, List, Optional

class BatchPredictor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    @contextmanager
    def _array_lifecycle(self, array: np.ndarray) -> Iterator[np.ndarray]:
        """Context manager to handle proper cleanup of large arrays"""
        try:
            yield array
        finally:
            # Explicitly delete array and run garbage collection
            del array
            gc.collect()

    def _chunk_array(self, array: np.ndarray) -> Iterator[np.ndarray]:
        """Split large array into smaller chunks for processing"""
        for i in range(0, len(array), self.chunk_size):
            chunk = array[i:i + self.chunk_size].copy()
            yield chunk

    def process_batch(self, features: np.ndarray) -> List[float]:
        """
        Process feature batch with proper memory management
        
        Args:
            features: Input feature array
            
        Returns:
            List of prediction results
        """
        results = []
        
        try:
            # Process in chunks to limit memory usage
            for chunk in self._chunk_array(features):
                with self._array_lifecycle(chunk) as active_chunk:
                    # Perform feature engineering
                    processed = self._engineer_features(active_chunk)
                    
                    # Generate predictions
                    chunk_results = self._predict(processed)
                    results.extend(chunk_results)
                    
                    # Explicit cleanup of processed features
                    del processed
                    
        except Exception as e:
            # Log error and cleanup
            print(f"Error processing batch: {str(e)}")
            gc.collect()
            raise

        return results

    def _engineer_features(self, chunk: np.ndarray) -> np.ndarray:
        """Apply feature engineering to chunk"""
        try:
            # Feature engineering logic here
            processed = chunk * 2  # Example transformation
            return processed
        except Exception as e:
            print(f"Feature engineering failed: {str(e)}")
            raise

    def _predict(self, processed_chunk: np.ndarray) -> List[float]:
        """Generate predictions for processed chunk"""
        try:
            # Prediction logic here
            predictions = processed_chunk.mean(axis=1).tolist()
            return predictions
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            raise

    def cleanup(self):
        """Force cleanup of any remaining memory"""
        gc.collect()
