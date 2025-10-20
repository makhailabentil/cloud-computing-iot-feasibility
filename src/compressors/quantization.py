"""
Quantization Compressor for IoT Time Series Data

This module implements quantization-based compression for time series data,
which reduces precision to achieve compression while maintaining acceptable
reconstruction quality.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationCompressor:
    """
    Quantization compressor for time series IoT data.
    
    Quantization reduces the number of bits used to represent each value,
    trading precision for compression. This is particularly effective for
    IoT data where high precision may not be necessary.
    """
    
    def __init__(self, n_bits: int = 8, method: str = 'uniform'):
        """
        Initialize the quantization compressor.
        
        Args:
            n_bits: Number of bits to use for quantization (1-16)
            method: Quantization method ('uniform', 'logarithmic', 'adaptive')
        """
        self.n_bits = n_bits
        self.method = method
        self.n_levels = 2 ** n_bits
        self.min_val = None
        self.max_val = None
        self.quantization_levels = None
        self.compression_ratio = 0.0
    
    def _compute_uniform_levels(self, data: np.ndarray) -> np.ndarray:
        """Compute uniform quantization levels."""
        self.min_val = np.min(data)
        self.max_val = np.max(data)
        return np.linspace(self.min_val, self.max_val, self.n_levels)
    
    def _compute_logarithmic_levels(self, data: np.ndarray) -> np.ndarray:
        """Compute logarithmic quantization levels."""
        # Use log scale for data with exponential-like distribution
        log_data = np.log(np.abs(data) + 1e-10)  # Add small value to avoid log(0)
        min_log = np.min(log_data)
        max_log = np.max(log_data)
        log_levels = np.linspace(min_log, max_log, self.n_levels)
        return np.exp(log_levels) * np.sign(data[np.argmax(np.abs(data))])
    
    def _compute_adaptive_levels(self, data: np.ndarray) -> np.ndarray:
        """Compute adaptive quantization levels using k-means-like approach."""
        # Simple adaptive quantization using histogram
        hist, bin_edges = np.histogram(data, bins=self.n_levels)
        return (bin_edges[:-1] + bin_edges[1:]) / 2
    
    def fit(self, data: Union[np.ndarray, List[float], pd.Series]) -> None:
        """
        Fit the quantizer to the data (compute quantization levels).
        
        Args:
            data: Input time series data for fitting
        """
        # Convert to numpy array for processing
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        if self.method == 'uniform':
            self.quantization_levels = self._compute_uniform_levels(data)
        elif self.method == 'logarithmic':
            self.quantization_levels = self._compute_logarithmic_levels(data)
        elif self.method == 'adaptive':
            self.quantization_levels = self._compute_adaptive_levels(data)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")
        
        logger.info(f"Fitted quantizer with {len(self.quantization_levels)} levels")
    
    def compress(self, data: Union[np.ndarray, List[float], pd.Series]) -> Tuple[np.ndarray, dict]:
        """
        Compress time series data using quantization.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (quantized_indices, metadata)
        """
        if self.quantization_levels is None:
            raise ValueError("Quantizer must be fitted before compression")
        
        # Convert to numpy array for processing
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        # Find closest quantization level for each value
        indices = np.zeros(len(data), dtype=np.uint16)
        for i, value in enumerate(data):
            distances = np.abs(self.quantization_levels - value)
            indices[i] = np.argmin(distances)
        
        # Calculate compression ratio
        original_size = len(data) * 8  # Assuming 8 bytes per float
        # Ensure at least 1 byte per quantized value
        bytes_per_value = max(1, self.n_bits // 8)
        compressed_size = len(indices) * bytes_per_value
        self.compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        metadata = {
            'quantization_levels': self.quantization_levels,
            'n_bits': self.n_bits,
            'method': self.method,
            'min_val': self.min_val,
            'max_val': self.max_val
        }
        
        logger.info(f"Compression ratio: {self.compression_ratio:.2f}x")
        
        return indices, metadata
    
    def decompress(self, compressed_data: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Decompress quantized data back to original time series.
        
        Args:
            compressed_data: Quantized indices
            metadata: Compression metadata
            
        Returns:
            Reconstructed time series data
        """
        quantization_levels = metadata['quantization_levels']
        reconstructed = quantization_levels[compressed_data]
        
        return reconstructed
    
    def calculate_quantization_error(self, original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """
        Calculate various error metrics for quantization.
        
        Args:
            original: Original time series data
            reconstructed: Reconstructed time series data
            
        Returns:
            Dictionary of error metrics
        """
        mse = np.mean((original - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original - reconstructed))
        max_error = np.max(np.abs(original - reconstructed))
        
        # Signal-to-noise ratio
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        errors = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'snr_db': snr_db
        }
        
        logger.info(f"Quantization errors - MSE: {mse:.6f}, RMSE: {rmse:.6f}, SNR: {snr_db:.2f} dB")
        
        return errors


def generate_sensor_data(n_points: int = 1000, noise_level: float = 0.1) -> np.ndarray:
    """
    Generate synthetic sensor data for testing.
    
    Args:
        n_points: Number of data points to generate
        noise_level: Level of noise to add to the signal
        
    Returns:
        Synthetic sensor data
    """
    # Generate realistic sensor data with trend and noise
    t = np.linspace(0, 10, n_points)
    signal = 20 + 5 * np.sin(2 * np.pi * t) + 2 * np.sin(4 * np.pi * t) + 0.1 * t
    noise = np.random.normal(0, noise_level, n_points)
    
    return signal + noise


def test_quantization():
    """
    Test the quantization compressor on synthetic data.
    """
    print("Testing Quantization Compressor")
    print("=" * 40)
    
    # Generate synthetic sensor data
    data = generate_sensor_data(n_points=1000, noise_level=0.5)
    print(f"Generated synthetic sensor data with {len(data)} points")
    
    # Test different quantization methods
    methods = ['uniform', 'logarithmic', 'adaptive']
    n_bits_options = [4, 6, 8]
    
    for method in methods:
        print(f"\n--- {method.upper()} Quantization ---")
        
        for n_bits in n_bits_options:
            print(f"\nTesting {n_bits}-bit quantization:")
            
            # Initialize compressor
            compressor = QuantizationCompressor(n_bits=n_bits, method=method)
            
            # Fit and compress
            compressor.fit(data)
            compressed_data, metadata = compressor.compress(data)
            reconstructed = compressor.decompress(compressed_data, metadata)
            
            # Calculate errors
            errors = compressor.calculate_quantization_error(data, reconstructed)
            
            print(f"  Compression ratio: {compressor.compression_ratio:.2f}x")
            print(f"  RMSE: {errors['rmse']:.4f}")
            print(f"  SNR: {errors['snr_db']:.2f} dB")


if __name__ == "__main__":
    test_quantization()
