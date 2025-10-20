"""
Run Length Encoding Compressor for IoT Time Series Data

This module implements run length encoding (RLE) compression for time series
data with repeated values, which is common in IoT sensor data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RunLengthCompressor:
    """
    Run Length Encoding compressor for time series IoT data.
    
    RLE is effective for data with many consecutive repeated values,
    which is common in IoT sensor readings where values may remain
    constant for extended periods.
    """
    
    def __init__(self, threshold: float = 1e-6):
        """
        Initialize the run length compressor.
        
        Args:
            threshold: Threshold for considering values as "equal" for compression
        """
        self.threshold = threshold
        self.compression_ratio = 0.0
    
    def compress(self, data: Union[np.ndarray, List[float], pd.Series]) -> Tuple[List[Tuple[float, int]], int]:
        """
        Compress time series data using run length encoding.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (compressed_runs, original_length)
        """
        # Convert to numpy array for processing
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        original_length = len(data)
        runs = []
        current_value = data[0]
        current_count = 1
        
        # Group consecutive similar values
        for i in range(1, len(data)):
            if abs(data[i] - current_value) <= self.threshold:
                current_count += 1
            else:
                # Store the run and start a new one
                runs.append((current_value, current_count))
                current_value = data[i]
                current_count = 1
        
        # Don't forget the last run
        runs.append((current_value, current_count))
        
        # Calculate compression ratio
        original_size = len(data) * 8  # Assuming 8 bytes per float
        compressed_size = len(runs) * (8 + 4)  # 8 bytes for value + 4 bytes for count
        self.compression_ratio = original_size / compressed_size
        
        logger.info(f"Compression ratio: {self.compression_ratio:.2f}x")
        logger.info(f"Number of runs: {len(runs)}")
        
        return runs, original_length
    
    def decompress(self, compressed_data: List[Tuple[float, int]], original_length: int) -> np.ndarray:
        """
        Decompress run length encoded data back to original time series.
        
        Args:
            compressed_data: List of (value, count) tuples
            original_length: Original length of the time series
            
        Returns:
            Reconstructed time series data
        """
        reconstructed = []
        
        for value, count in compressed_data:
            reconstructed.extend([value] * count)
        
        return np.array(reconstructed[:original_length])  # Ensure correct length
    
    def compress_adaptive(self, data: Union[np.ndarray, List[float], pd.Series], 
                         min_run_length: int = 3) -> Tuple[List[Union[float, Tuple[float, int]]], int]:
        """
        Adaptive RLE that only compresses runs longer than min_run_length.
        
        Args:
            data: Input time series data
            min_run_length: Minimum run length to compress
            
        Returns:
            Tuple of (compressed_data, original_length)
        """
        # Convert to numpy array for processing
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        original_length = len(data)
        compressed = []
        current_value = data[0]
        current_count = 1
        
        for i in range(1, len(data)):
            if abs(data[i] - current_value) <= self.threshold:
                current_count += 1
            else:
                # Decide whether to compress this run
                if current_count >= min_run_length:
                    compressed.append((current_value, current_count))
                else:
                    # Store individual values
                    compressed.extend([current_value] * current_count)
                
                current_value = data[i]
                current_count = 1
        
        # Handle the last run
        if current_count >= min_run_length:
            compressed.append((current_value, current_count))
        else:
            compressed.extend([current_value] * current_count)
        
        return compressed, original_length
    
    def decompress_adaptive(self, compressed_data: List[Union[float, Tuple[float, int]]], 
                           original_length: int) -> np.ndarray:
        """
        Decompress adaptive RLE data.
        
        Args:
            compressed_data: Mixed list of values and (value, count) tuples
            original_length: Original length of the time series
            
        Returns:
            Reconstructed time series data
        """
        reconstructed = []
        
        for item in compressed_data:
            if isinstance(item, tuple):
                # It's a run: (value, count)
                value, count = item
                reconstructed.extend([value] * count)
            else:
                # It's a single value
                reconstructed.append(item)
        
        return np.array(reconstructed[:original_length])


def generate_repetitive_data(n_points: int = 1000, repetition_factor: float = 0.3) -> np.ndarray:
    """
    Generate synthetic time series data with repetitive patterns.
    
    Args:
        n_points: Number of data points to generate
        repetition_factor: Fraction of data that should be repetitive
        
    Returns:
        Synthetic time series data with repetitive patterns
    """
    data = []
    current_value = 0.0
    
    for i in range(n_points):
        if np.random.random() < repetition_factor:
            # Repeat the current value
            data.append(current_value)
        else:
            # Generate a new value
            current_value = np.random.normal(0, 1)
            data.append(current_value)
    
    return np.array(data)


def test_run_length_compression():
    """
    Test the run length compressor on synthetic data.
    """
    print("Testing Run Length Encoding Compressor")
    print("=" * 40)
    
    # Generate synthetic data with repetitive patterns
    data = generate_repetitive_data(n_points=1000, repetition_factor=0.4)
    print(f"Generated synthetic data with {len(data)} points")
    
    # Initialize compressor
    compressor = RunLengthCompressor(threshold=1e-6)
    
    # Test standard RLE
    print("\n--- Standard RLE ---")
    compressed_data, original_length = compressor.compress(data)
    reconstructed = compressor.decompress(compressed_data, original_length)
    
    print(f"Compression ratio: {compressor.compression_ratio:.2f}x")
    print(f"Data integrity: {'PASS' if np.allclose(data, reconstructed, rtol=1e-10) else 'FAIL'}")
    
    # Test adaptive RLE
    print("\n--- Adaptive RLE ---")
    adaptive_compressed, original_length = compressor.compress_adaptive(data, min_run_length=3)
    adaptive_reconstructed = compressor.decompress_adaptive(adaptive_compressed, original_length)
    
    print(f"Data integrity: {'PASS' if np.allclose(data, adaptive_reconstructed, rtol=1e-10) else 'FAIL'}")


if __name__ == "__main__":
    test_run_length_compression()
