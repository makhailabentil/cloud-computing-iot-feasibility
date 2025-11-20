"""
Hybrid Compression Methods

This module implements hybrid compression techniques that combine multiple
algorithms for optimal compression/quality trade-offs.

For Milestone 3: Hybrid Compression Methods
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from compressors.delta_encoding import DeltaEncodingCompressor
from compressors.run_length import RunLengthCompressor
from compressors.quantization import QuantizationCompressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridCompressor:
    """
    Hybrid compressor that combines multiple compression algorithms.
    
    Supported hybrid methods:
    1. Delta + Quantization: Delta encoding followed by quantization
    2. Delta + RLE: Delta encoding followed by run-length encoding
    3. Quantization + Delta: Quantization followed by delta encoding
    """
    
    def __init__(self):
        """Initialize the hybrid compressor."""
        self.delta_compressor = DeltaEncodingCompressor()
        self.rle_compressor = RunLengthCompressor(threshold=0.01)
        self.quant_compressor = QuantizationCompressor(n_bits=8, method='uniform')
    
    def delta_then_quantization(self, data: np.ndarray) -> Tuple[Dict, float, float]:
        """
        Apply delta encoding, then quantize the deltas.
        
        This can achieve better compression than quantization alone on
        time series data with temporal correlation.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (compressed_data_dict, compression_ratio, reconstruction_error)
        """
        # Step 1: Delta encoding
        deltas, first_val = self.delta_compressor.compress(data)
        
        # Step 2: Fit quantizer on deltas, then quantize
        # Create a new quantizer instance to fit on deltas
        quantizer = QuantizationCompressor(n_bits=8, method='uniform')
        quantizer.fit(deltas)  # Fit quantizer on delta values
        quantized, quant_metadata = quantizer.compress(deltas)
        
        # Calculate compression ratio
        original_size = len(data) * 8  # 8 bytes per float
        compressed_size = len(quantized) * 1  # 1 byte per quantized value
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Decompress to calculate error
        dequantized = quantizer.decompress(quantized, quant_metadata)
        reconstructed = self.delta_compressor.decompress(dequantized, first_val)
        
        # Calculate reconstruction error
        mse = np.mean((data - reconstructed) ** 2)
        
        return {
            'method': 'delta_quantization',
            'first_val': first_val,
            'quantized_deltas': quantized,
            'quant_metadata': quant_metadata,
            'quantizer_fitted': True  # Flag to indicate quantizer was fitted
        }, compression_ratio, mse
    
    def delta_then_rle(self, data: np.ndarray) -> Tuple[Dict, float, float]:
        """
        Apply delta encoding, then run-length encode the deltas.
        
        This is effective when deltas have many repeated values.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (compressed_data_dict, compression_ratio, reconstruction_error)
        """
        # Step 1: Delta encoding
        deltas, first_val = self.delta_compressor.compress(data)
        
        # Step 2: Run-length encode the deltas
        rle_compressed, original_len = self.rle_compressor.compress(deltas)
        
        # Calculate compression ratio
        original_size = len(data) * 8  # 8 bytes per float
        # RLE stores (value, count) pairs, each as float (8 bytes)
        compressed_size = len(rle_compressed) * 8
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Decompress to verify (should be lossless)
        rle_decompressed = self.rle_compressor.decompress(rle_compressed, original_len)
        reconstructed = self.delta_compressor.decompress(rle_decompressed, first_val)
        
        # Calculate reconstruction error (should be 0 for lossless)
        mse = np.mean((data - reconstructed) ** 2)
        
        return {
            'method': 'delta_rle',
            'first_val': first_val,
            'rle_compressed': rle_compressed,
            'original_len': original_len
        }, compression_ratio, mse
    
    def quantization_then_delta(self, data: np.ndarray) -> Tuple[Dict, float, float]:
        """
        Apply quantization, then delta encoding on quantized values.
        
        This can sometimes achieve better compression by exploiting
        temporal correlation in quantized values.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (compressed_data_dict, compression_ratio, reconstruction_error)
        """
        # Step 1: Fit quantizer on data, then quantize
        quantizer = QuantizationCompressor(n_bits=8, method='uniform')
        quantizer.fit(data)  # Fit quantizer on original data
        quantized, quant_metadata = quantizer.compress(data)
        
        # Step 2: Delta encoding on quantized values
        # Convert quantized to float array for delta encoding
        quantized_float = quantized.astype(np.float32)
        deltas, first_val = self.delta_compressor.compress(quantized_float)
        
        # Calculate compression ratio
        original_size = len(data) * 8  # 8 bytes per float
        compressed_size = len(deltas) * 4  # 4 bytes per float (delta)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Decompress
        dequantized_deltas = self.delta_compressor.decompress(deltas, first_val)
        reconstructed = quantizer.decompress(dequantized_deltas.astype(np.uint8), quant_metadata)
        
        # Calculate reconstruction error
        mse = np.mean((data - reconstructed) ** 2)
        
        return {
            'method': 'quantization_delta',
            'quant_metadata': quant_metadata,
            'deltas': deltas,
            'first_val': first_val,
            'quantizer_fitted': True  # Flag to indicate quantizer was fitted
        }, compression_ratio, mse
    
    def decompress_hybrid(self, compressed_data: Dict) -> np.ndarray:
        """
        Decompress hybrid-compressed data.
        
        Args:
            compressed_data: Dictionary from hybrid compression methods
            
        Returns:
            Decompressed data array
        """
        method = compressed_data['method']
        
        if method == 'delta_quantization':
            # Create quantizer and fit it using metadata
            quantizer = QuantizationCompressor(n_bits=8, method='uniform')
            # Reconstruct quantization levels from metadata
            quantizer.min_val = compressed_data['quant_metadata'].get('min_val')
            quantizer.max_val = compressed_data['quant_metadata'].get('max_val')
            quantizer.quantization_levels = np.linspace(quantizer.min_val, quantizer.max_val, 256)
            
            # Dequantize
            dequantized = quantizer.decompress(
                compressed_data['quantized_deltas'],
                compressed_data['quant_metadata']
            )
            # Delta decompress
            reconstructed = self.delta_compressor.decompress(
                dequantized,
                compressed_data['first_val']
            )
            
        elif method == 'delta_rle':
            # RLE decompress
            rle_decompressed = self.rle_compressor.decompress(
                compressed_data['rle_compressed'],
                compressed_data['original_len']
            )
            # Delta decompress
            reconstructed = self.delta_compressor.decompress(
                rle_decompressed,
                compressed_data['first_val']
            )
            
        elif method == 'quantization_delta':
            # Create quantizer and fit it using metadata
            quantizer = QuantizationCompressor(n_bits=8, method='uniform')
            # Reconstruct quantization levels from metadata
            quantizer.min_val = compressed_data['quant_metadata'].get('min_val')
            quantizer.max_val = compressed_data['quant_metadata'].get('max_val')
            quantizer.quantization_levels = np.linspace(quantizer.min_val, quantizer.max_val, 256)
            
            # Delta decompress
            dequantized_deltas = self.delta_compressor.decompress(
                compressed_data['deltas'],
                compressed_data['first_val']
            )
            # Dequantize
            reconstructed = quantizer.decompress(
                dequantized_deltas.astype(np.uint8),
                compressed_data['quant_metadata']
            )
            
        else:
            raise ValueError(f"Unknown hybrid method: {method}")
        
        return reconstructed
    
    def compare_hybrid_methods(self, data: np.ndarray) -> Dict:
        """
        Compare all hybrid methods on the same data.
        
        Args:
            data: Input time series data
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # Test each hybrid method
        methods = [
            ('delta_quantization', self.delta_then_quantization),
            ('delta_rle', self.delta_then_rle),
            ('quantization_delta', self.quantization_then_delta)
        ]
        
        for method_name, method_func in methods:
            try:
                compressed, ratio, error = method_func(data)
                results[method_name] = {
                    'compression_ratio': ratio,
                    'reconstruction_error': error,
                    'rmse': np.sqrt(error),
                    'compressed_size': len(str(compressed))  # Approximate
                }
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
                results[method_name] = {
                    'compression_ratio': 0.0,
                    'reconstruction_error': float('inf'),
                    'error': str(e)
                }
        
        return results


def test_hybrid_compressor():
    """Test the hybrid compressor."""
    compressor = HybridCompressor()
    
    # Generate test data
    t = np.linspace(0, 10, 1000)
    data = 0.5 * np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.sin(2 * np.pi * 10.0 * t)
    
    print("Hybrid Compressor Test")
    print("=" * 60)
    
    # Compare all methods
    results = compressor.compare_hybrid_methods(data)
    
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  Reconstruction Error (MSE): {metrics['reconstruction_error']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")


if __name__ == "__main__":
    test_hybrid_compressor()

