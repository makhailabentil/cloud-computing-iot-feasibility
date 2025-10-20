"""
Test suite for compression algorithms.

This module contains comprehensive tests for all compression algorithms
implemented in the project.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from compressors.delta_encoding import DeltaEncodingCompressor
from compressors.run_length import RunLengthCompressor
from compressors.quantization import QuantizationCompressor


class TestDeltaEncodingCompressor:
    """Test cases for delta encoding compressor."""
    
    def test_compression_decompression(self):
        """Test basic compression and decompression."""
        compressor = DeltaEncodingCompressor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        compressed, first_value = compressor.compress(data)
        reconstructed = compressor.decompress(compressed, first_value)
        
        assert np.allclose(data, reconstructed, rtol=1e-10)
        assert compressor.compression_ratio > 1.0
    
    def test_synthetic_data(self):
        """Test compression on synthetic data."""
        compressor = DeltaEncodingCompressor()
        data = np.sin(np.linspace(0, 2*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        
        compressed, first_value = compressor.compress(data)
        reconstructed = compressor.decompress(compressed, first_value)
        
        # Check data integrity
        assert np.allclose(data, reconstructed, rtol=1e-10)
        
        # Check compression ratio is reasonable
        assert compressor.compression_ratio > 1.0
        assert compressor.compression_ratio < 10.0  # Should not be too high for smooth data
    
    def test_batch_compression(self):
        """Test batch compression functionality."""
        compressor = DeltaEncodingCompressor(window_size=100)
        data = np.random.randn(500)
        
        results = compressor.compress_batch(data)
        
        assert len(results) > 0
        assert all(len(compressed) > 0 for compressed, _ in results)
    
    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        compressor = DeltaEncodingCompressor()
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        compressed, first_value = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, first_value)
        
        error = compressor.calculate_reconstruction_error(original, reconstructed)
        assert error == 0.0  # Should be perfect for this simple case


class TestRunLengthCompressor:
    """Test cases for run length encoding compressor."""
    
    def test_compression_decompression(self):
        """Test basic compression and decompression."""
        compressor = RunLengthCompressor()
        data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        
        compressed, original_length = compressor.compress(data)
        reconstructed = compressor.decompress(compressed, original_length)
        
        assert np.allclose(data, reconstructed, rtol=1e-10)
        assert compressor.compression_ratio > 1.0
    
    def test_repetitive_data(self):
        """Test compression on highly repetitive data."""
        compressor = RunLengthCompressor()
        data = np.array([1.0] * 100 + [2.0] * 50 + [3.0] * 200)
        
        compressed, original_length = compressor.compress(data)
        reconstructed = compressor.decompress(compressed, original_length)
        
        assert np.allclose(data, reconstructed, rtol=1e-10)
        assert compressor.compression_ratio > 5.0  # Should be very high for repetitive data
    
    def test_adaptive_compression(self):
        """Test adaptive RLE compression."""
        compressor = RunLengthCompressor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        compressed, original_length = compressor.compress_adaptive(data, min_run_length=3)
        reconstructed = compressor.decompress_adaptive(compressed, original_length)
        
        assert np.allclose(data, reconstructed, rtol=1e-10)
    
    def test_threshold_parameter(self):
        """Test compression with different threshold values."""
        data = np.array([1.0, 1.01, 1.02, 2.0, 2.01, 2.02])
        
        # Test with small threshold (should not compress much)
        compressor1 = RunLengthCompressor(threshold=1e-6)
        compressed1, _ = compressor1.compress(data)
        
        # Test with larger threshold (should compress more)
        compressor2 = RunLengthCompressor(threshold=0.1)
        compressed2, _ = compressor2.compress(data)
        
        assert len(compressed2) <= len(compressed1)


class TestQuantizationCompressor:
    """Test cases for quantization compressor."""
    
    def test_uniform_quantization(self):
        """Test uniform quantization."""
        compressor = QuantizationCompressor(n_bits=4, method='uniform')
        data = np.linspace(0, 10, 100)
        
        compressor.fit(data)
        compressed, metadata = compressor.compress(data)
        reconstructed = compressor.decompress(compressed, metadata)
        
        # Check that quantization reduces precision
        assert not np.allclose(data, reconstructed, rtol=1e-10)
        
        # Check that error is reasonable
        errors = compressor.calculate_quantization_error(data, reconstructed)
        assert errors['mse'] > 0
        assert errors['snr_db'] > 0
    
    def test_logarithmic_quantization(self):
        """Test logarithmic quantization."""
        compressor = QuantizationCompressor(n_bits=6, method='logarithmic')
        data = np.exp(np.linspace(0, 5, 100))  # Exponential data
        
        compressor.fit(data)
        compressed, metadata = compressor.compress(data)
        reconstructed = compressor.decompress(compressed, metadata)
        
        # Check compression works
        assert len(compressed) == len(data)
        assert compressor.compression_ratio > 1.0
    
    def test_adaptive_quantization(self):
        """Test adaptive quantization."""
        compressor = QuantizationCompressor(n_bits=8, method='adaptive')
        data = np.random.normal(0, 1, 1000)
        
        compressor.fit(data)
        compressed, metadata = compressor.compress(data)
        reconstructed = compressor.decompress(compressed, metadata)
        
        # Check that quantization works
        assert len(compressed) == len(data)
        assert compressor.compression_ratio > 1.0
    
    def test_different_bit_depths(self):
        """Test quantization with different bit depths."""
        data = np.linspace(0, 10, 100)
        
        for n_bits in [2, 4, 6, 8]:
            compressor = QuantizationCompressor(n_bits=n_bits, method='uniform')
            compressor.fit(data)
            compressed, metadata = compressor.compress(data)
            reconstructed = compressor.decompress(compressed, metadata)
            
            errors = compressor.calculate_quantization_error(data, reconstructed)
            
            # Higher bit depth should give better quality
            if n_bits > 2:
                assert errors['mse'] > 0
                assert errors['snr_db'] > 0
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        compressor = QuantizationCompressor(n_bits=4, method='uniform')
        data = np.linspace(0, 10, 100)
        
        compressor.fit(data)
        compressed, metadata = compressor.compress(data)
        
        # Check that compression ratio is calculated correctly
        expected_ratio = 8 / (compressor.n_bits / 8)  # 8 bytes per float / bits per quantized value
        assert abs(compressor.compression_ratio - expected_ratio) < 0.1


class TestCompressionIntegration:
    """Integration tests for compression algorithms."""
    
    def test_all_compressors_on_same_data(self):
        """Test all compressors on the same dataset."""
        data = np.sin(np.linspace(0, 4*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        
        # Test delta encoding
        delta_compressor = DeltaEncodingCompressor()
        delta_compressed, delta_first = delta_compressor.compress(data)
        delta_reconstructed = delta_compressor.decompress(delta_compressed, delta_first)
        
        # Test run length encoding
        rle_compressor = RunLengthCompressor()
        rle_compressed, rle_length = rle_compressor.compress(data)
        rle_reconstructed = rle_compressor.decompress(rle_compressed, rle_length)
        
        # Test quantization
        quant_compressor = QuantizationCompressor(n_bits=8, method='uniform')
        quant_compressor.fit(data)
        quant_compressed, quant_metadata = quant_compressor.compress(data)
        quant_reconstructed = quant_compressor.decompress(quant_compressed, quant_metadata)
        
        # Check that all methods work
        assert np.allclose(data, delta_reconstructed, rtol=1e-10)
        assert np.allclose(data, rle_reconstructed, rtol=1e-10)
        # Quantization will have some error
        assert not np.allclose(data, quant_reconstructed, rtol=1e-10)
        
        # Check compression ratios
        assert delta_compressor.compression_ratio > 1.0
        assert rle_compressor.compression_ratio > 1.0
        assert quant_compressor.compression_ratio > 1.0
    
    def test_performance_comparison(self):
        """Compare performance of different compression methods."""
        data = np.random.randn(10000)
        
        # Delta encoding
        delta_compressor = DeltaEncodingCompressor()
        delta_compressed, delta_first = delta_compressor.compress(data)
        delta_reconstructed = delta_compressor.decompress(delta_compressed, delta_first)
        
        # Run length encoding
        rle_compressor = RunLengthCompressor()
        rle_compressed, rle_length = rle_compressor.compress(data)
        rle_reconstructed = rle_compressor.decompress(rle_compressed, rle_length)
        
        # Quantization
        quant_compressor = QuantizationCompressor(n_bits=8, method='uniform')
        quant_compressor.fit(data)
        quant_compressed, quant_metadata = quant_compressor.compress(data)
        quant_reconstructed = quant_compressor.decompress(quant_compressed, quant_metadata)
        
        # Compare compression ratios
        print(f"Delta encoding ratio: {delta_compressor.compression_ratio:.2f}x")
        print(f"RLE ratio: {rle_compressor.compression_ratio:.2f}x")
        print(f"Quantization ratio: {quant_compressor.compression_ratio:.2f}x")
        
        # All should achieve some compression
        assert delta_compressor.compression_ratio > 1.0
        assert rle_compressor.compression_ratio > 1.0
        assert quant_compressor.compression_ratio > 1.0


if __name__ == "__main__":
    pytest.main([__file__])
