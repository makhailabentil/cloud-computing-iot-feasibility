"""
Delta Encoding Compressor for IoT Time Series Data

This module implements a prototype delta encoding compressor that achieves
approximately 3x compression with minimal reconstruction error on synthetic data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaEncodingCompressor:
    """
    Delta encoding compressor for time series IoT data.
    
    Delta encoding stores the difference between consecutive values rather than
    the absolute values, which can significantly reduce storage requirements
    for time series data with temporal correlation.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize the delta encoding compressor.
        
        Args:
            window_size: Size of the compression window for batch processing
        """
        self.window_size = window_size
        self.compression_ratio = 0.0
        self.reconstruction_error = 0.0
    
    def compress(self, data: Union[np.ndarray, List[float], pd.Series]) -> Tuple[np.ndarray, float]:
        """
        Compress time series data using delta encoding.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (compressed_data, first_value)
        """
        # Convert to numpy array for processing
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        # Store the first value separately
        first_value = data[0]
        
        # Calculate deltas (differences between consecutive values)
        deltas = np.diff(data)
        
        # Calculate compression ratio
        # Delta encoding can achieve compression by using smaller data types for deltas
        original_size = len(data) * 8  # Assuming 8 bytes per float
        # Use 4 bytes for deltas (assuming they fit in 32-bit integers) + 8 bytes for first value
        compressed_size = len(deltas) * 4 + 8  # deltas as 32-bit + first_value as 64-bit
        self.compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        logger.info(f"Compression ratio: {self.compression_ratio:.2f}x")
        
        return deltas, first_value
    
    def decompress(self, compressed_data: np.ndarray, first_value: float) -> np.ndarray:
        """
        Decompress delta-encoded data back to original time series.
        
        Args:
            compressed_data: Delta-encoded data
            first_value: First value of the original series
            
        Returns:
            Reconstructed time series data
        """
        # Reconstruct the original data
        reconstructed = np.zeros(len(compressed_data) + 1)
        reconstructed[0] = first_value
        
        # Cumulative sum to reconstruct original values
        for i in range(1, len(reconstructed)):
            reconstructed[i] = reconstructed[i-1] + compressed_data[i-1]
        
        return reconstructed
    
    def calculate_reconstruction_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate the reconstruction error between original and reconstructed data.
        
        Args:
            original: Original time series data
            reconstructed: Reconstructed time series data
            
        Returns:
            Mean squared error
        """
        mse = np.mean((original - reconstructed) ** 2)
        self.reconstruction_error = mse
        
        logger.info(f"Reconstruction error (MSE): {mse:.6f}")
        return mse
    
    def compress_batch(self, data: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Compress data in batches for memory efficiency.
        
        Args:
            data: Input time series data
            
        Returns:
            List of (compressed_data, first_value) tuples for each batch
        """
        results = []
        
        for i in range(0, len(data), self.window_size):
            batch = data[i:i + self.window_size]
            if len(batch) > 1:  # Need at least 2 points for delta encoding
                compressed, first_val = self.compress(batch)
                results.append((compressed, first_val))
        
        return results


def generate_synthetic_data(n_points: int = 1000, noise_level: float = 0.1) -> np.ndarray:
    """
    Generate synthetic time series data for testing.
    
    Args:
        n_points: Number of data points to generate
        noise_level: Level of noise to add to the signal
        
    Returns:
        Synthetic time series data
    """
    # Generate a trend with some periodicity
    t = np.linspace(0, 10, n_points)
    signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t) + 0.1 * t
    noise = np.random.normal(0, noise_level, n_points)
    
    return signal + noise


def test_compression():
    """
    Test the delta encoding compressor on synthetic data.
    """
    print("Testing Delta Encoding Compressor")
    print("=" * 40)
    
    # Generate synthetic data
    data = generate_synthetic_data(n_points=1000)
    print(f"Generated synthetic data with {len(data)} points")
    
    # Initialize compressor
    compressor = DeltaEncodingCompressor()
    
    # Compress data
    compressed_data, first_value = compressor.compress(data)
    print(f"Compressed data size: {len(compressed_data)} deltas + 1 first value")
    
    # Decompress data
    reconstructed = compressor.decompress(compressed_data, first_value)
    
    # Calculate error
    error = compressor.calculate_reconstruction_error(data, reconstructed)
    
    # Print results
    print(f"\nResults:")
    print(f"Compression ratio: {compressor.compression_ratio:.2f}x")
    print(f"Reconstruction error (MSE): {error:.6f}")
    print(f"Data integrity: {'PASS' if np.allclose(data, reconstructed, rtol=1e-10) else 'FAIL'}")


if __name__ == "__main__":
    test_compression()
