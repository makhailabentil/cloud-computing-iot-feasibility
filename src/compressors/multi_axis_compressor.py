"""
Multi-Axis Compression Strategies

This module implements compression strategies optimized for triaxial
accelerometer data (x, y, z axes).

For Milestone 3: Multi-Axis Compression Strategies
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


class MultiAxisCompressor:
    """
    Multi-axis compressor for triaxial accelerometer data.
    
    Implements several strategies:
    1. Joint compression: Compress all axes together
    2. Vector-based compression: Treat (x,y,z) as 3D vectors
    3. Magnitude-based compression: Compress magnitude instead of individual axes
    4. Axis-specific selection: Different algorithms per axis
    """
    
    def __init__(self):
        """Initialize the multi-axis compressor."""
        self.delta_compressor = DeltaEncodingCompressor()
        self.rle_compressor = RunLengthCompressor(threshold=0.01)
        self.quant_compressor = QuantizationCompressor(n_bits=8, method='uniform')
    
    def joint_compression(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         algorithm: str = 'delta_encoding') -> Dict:
        """
        Compress all axes together as a single interleaved stream.
        
        Interleaving: [x0, y0, z0, x1, y1, z1, ...]
        
        Args:
            x: X-axis data
            y: Y-axis data
            z: Z-axis data
            algorithm: Compression algorithm to use
            
        Returns:
            Dictionary with compressed data and metadata
        """
        # Ensure same length
        min_len = min(len(x), len(y), len(z))
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        
        # Interleave axes
        interleaved = np.zeros(min_len * 3)
        interleaved[0::3] = x
        interleaved[1::3] = y
        interleaved[2::3] = z
        
        # Compress interleaved data
        if algorithm == 'delta_encoding':
            compressed, first_val = self.delta_compressor.compress(interleaved)
            decompressed = self.delta_compressor.decompress(compressed, first_val)
            ratio = self.delta_compressor.compression_ratio
            error = self.delta_compressor.reconstruction_error
            
        elif algorithm == 'quantization':
            compressed, metadata = self.quant_compressor.compress(interleaved)
            decompressed = self.quant_compressor.decompress(compressed, metadata)
            ratio = self.quant_compressor.compression_ratio
            error = self.quant_compressor.reconstruction_error
            
        else:
            raise ValueError(f"Unsupported algorithm for joint compression: {algorithm}")
        
        # Deinterleave for error calculation
        x_recon = decompressed[0::3]
        y_recon = decompressed[1::3]
        z_recon = decompressed[2::3]
        
        # Calculate per-axis errors
        mse_x = np.mean((x - x_recon) ** 2)
        mse_y = np.mean((y - y_recon) ** 2)
        mse_z = np.mean((z - z_recon) ** 2)
        
        return {
            'method': 'joint',
            'algorithm': algorithm,
            'compressed_data': compressed,
            'first_val': first_val if algorithm == 'delta_encoding' else None,
            'quant_metadata': metadata if algorithm == 'quantization' else None,
            'compression_ratio': ratio,
            'reconstruction_error': error,
            'axis_errors': {'x': mse_x, 'y': mse_y, 'z': mse_z}
        }
    
    def vector_compression(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         algorithm: str = 'delta_encoding') -> Dict:
        """
        Compress (x,y,z) as 3D vectors.
        
        Stores magnitude and direction (spherical coordinates) instead of
        Cartesian coordinates.
        
        Args:
            x: X-axis data
            y: Y-axis data
            z: Z-axis data
            algorithm: Compression algorithm to use
            
        Returns:
            Dictionary with compressed data and metadata
        """
        # Ensure same length
        min_len = min(len(x), len(y), len(z))
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        
        # Convert to spherical coordinates
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)  # Azimuth angle
        phi = np.arccos(z / (magnitude + 1e-10))  # Polar angle
        
        # Compress each component
        if algorithm == 'delta_encoding':
            mag_compressed, mag_first = self.delta_compressor.compress(magnitude)
            theta_compressed, theta_first = self.delta_compressor.compress(theta)
            phi_compressed, phi_first = self.delta_compressor.compress(phi)
            
            # Decompress for error calculation
            mag_recon = self.delta_compressor.decompress(mag_compressed, mag_first)
            theta_recon = self.delta_compressor.decompress(theta_compressed, theta_first)
            phi_recon = self.delta_compressor.decompress(phi_compressed, phi_first)
            
            # Convert back to Cartesian
            x_recon = mag_recon * np.sin(phi_recon) * np.cos(theta_recon)
            y_recon = mag_recon * np.sin(phi_recon) * np.sin(theta_recon)
            z_recon = mag_recon * np.cos(phi_recon)
            
            ratio = self.delta_compressor.compression_ratio
            error_x = np.mean((x - x_recon) ** 2)
            error_y = np.mean((y - y_recon) ** 2)
            error_z = np.mean((z - z_recon) ** 2)
            error = (error_x + error_y + error_z) / 3.0
            
        else:
            raise ValueError(f"Unsupported algorithm for vector compression: {algorithm}")
        
        return {
            'method': 'vector',
            'algorithm': algorithm,
            'magnitude_compressed': mag_compressed,
            'theta_compressed': theta_compressed,
            'phi_compressed': phi_compressed,
            'magnitude_first': mag_first,
            'theta_first': theta_first,
            'phi_first': phi_first,
            'compression_ratio': ratio,
            'reconstruction_error': error,
            'axis_errors': {'x': error_x, 'y': error_y, 'z': error_z}
        }
    
    def magnitude_compression(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            algorithm: str = 'delta_encoding') -> Dict:
        """
        Compress only the magnitude, discarding directional information.
        
        Args:
            x: X-axis data
            y: Y-axis data
            z: Z-axis data
            algorithm: Compression algorithm to use
            
        Returns:
            Dictionary with compressed data and metadata
        """
        # Calculate magnitude
        min_len = min(len(x), len(y), len(z))
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Compress magnitude
        if algorithm == 'delta_encoding':
            compressed, first_val = self.delta_compressor.compress(magnitude)
            decompressed = self.delta_compressor.decompress(compressed, first_val)
            ratio = self.delta_compressor.compression_ratio
            error = self.delta_compressor.reconstruction_error
            
        elif algorithm == 'quantization':
            compressed, metadata = self.quant_compressor.compress(magnitude)
            decompressed = self.quant_compressor.decompress(compressed, metadata)
            ratio = self.quant_compressor.compression_ratio
            error = self.quant_compressor.reconstruction_error
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Note: Cannot reconstruct x, y, z from magnitude alone
        # This is a lossy compression that discards directional information
        
        return {
            'method': 'magnitude_only',
            'algorithm': algorithm,
            'compressed_data': compressed,
            'first_val': first_val if algorithm == 'delta_encoding' else None,
            'quant_metadata': metadata if algorithm == 'quantization' else None,
            'compression_ratio': ratio * 3.0,  # 3x better since we compress 1 value instead of 3
            'reconstruction_error': error,
            'note': 'Directional information lost - cannot reconstruct x, y, z'
        }
    
    def axis_specific_compression(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                 axis_algorithms: Dict[str, str] = None) -> Dict:
        """
        Compress each axis with a different algorithm.
        
        Args:
            x: X-axis data
            y: Y-axis data
            z: Z-axis data
            axis_algorithms: Dictionary mapping axis to algorithm
                           e.g., {'x': 'delta_encoding', 'y': 'quantization', 'z': 'run_length'}
            
        Returns:
            Dictionary with compressed data and metadata
        """
        if axis_algorithms is None:
            # Default: use delta encoding for all
            axis_algorithms = {'x': 'delta_encoding', 'y': 'delta_encoding', 'z': 'delta_encoding'}
        
        results = {}
        compression_ratios = []
        reconstruction_errors = []
        
        for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
            algorithm = axis_algorithms.get(axis_name, 'delta_encoding')
            
            if algorithm == 'delta_encoding':
                compressed, first_val = self.delta_compressor.compress(axis_data)
                decompressed = self.delta_compressor.decompress(compressed, first_val)
                ratio = self.delta_compressor.compression_ratio
                error = self.delta_compressor.reconstruction_error
                metadata = {'first_val': first_val}
                
            elif algorithm == 'run_length':
                compressed, original_len = self.rle_compressor.compress(axis_data)
                decompressed = self.rle_compressor.decompress(compressed, original_len)
                ratio = self.rle_compressor.compression_ratio
                error = 0.0
                metadata = {'original_len': original_len}
                
            elif algorithm == 'quantization':
                compressed, quant_metadata = self.quant_compressor.compress(axis_data)
                decompressed = self.quant_compressor.decompress(compressed, quant_metadata)
                ratio = self.quant_compressor.compression_ratio
                error = self.quant_compressor.reconstruction_error
                metadata = quant_metadata
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            results[axis_name] = {
                'algorithm': algorithm,
                'compressed_data': compressed,
                'metadata': metadata
            }
            compression_ratios.append(ratio)
            reconstruction_errors.append(error)
        
        return {
            'method': 'axis_specific',
            'axis_results': results,
            'compression_ratio': np.mean(compression_ratios),
            'reconstruction_error': np.mean(reconstruction_errors),
            'axis_ratios': {'x': compression_ratios[0], 'y': compression_ratios[1], 'z': compression_ratios[2]},
            'axis_errors': {'x': reconstruction_errors[0], 'y': reconstruction_errors[1], 'z': reconstruction_errors[2]}
        }
    
    def compare_strategies(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """
        Compare all multi-axis compression strategies.
        
        Args:
            x: X-axis data
            y: Y-axis data
            z: Z-axis data
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # Test joint compression
        try:
            results['joint_delta'] = self.joint_compression(x, y, z, 'delta_encoding')
        except Exception as e:
            logger.warning(f"Joint compression failed: {e}")
        
        # Test vector compression
        try:
            results['vector_delta'] = self.vector_compression(x, y, z, 'delta_encoding')
        except Exception as e:
            logger.warning(f"Vector compression failed: {e}")
        
        # Test magnitude compression
        try:
            results['magnitude_delta'] = self.magnitude_compression(x, y, z, 'delta_encoding')
        except Exception as e:
            logger.warning(f"Magnitude compression failed: {e}")
        
        # Test axis-specific (use best algorithm per axis based on variance)
        try:
            var_x = np.var(x)
            var_y = np.var(y)
            var_z = np.var(z)
            
            # Use RLE for low variance, delta for high variance
            axis_algs = {
                'x': 'run_length' if var_x < 0.1 else 'delta_encoding',
                'y': 'run_length' if var_y < 0.1 else 'delta_encoding',
                'z': 'run_length' if var_z < 0.1 else 'delta_encoding'
            }
            results['axis_specific'] = self.axis_specific_compression(x, y, z, axis_algs)
        except Exception as e:
            logger.warning(f"Axis-specific compression failed: {e}")
        
        return results


def test_multi_axis_compressor():
    """Test the multi-axis compressor."""
    compressor = MultiAxisCompressor()
    
    # Generate test data
    t = np.linspace(0, 10, 1000)
    x = 0.5 * np.sin(2 * np.pi * 1.0 * t)
    y = 0.8 + 0.3 * np.cos(2 * np.pi * 1.0 * t)
    z = 0.6 + 0.2 * np.sin(2 * np.pi * 1.0 * t)
    
    print("Multi-Axis Compressor Test")
    print("=" * 60)
    
    # Compare all strategies
    results = compressor.compare_strategies(x, y, z)
    
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  Reconstruction Error: {metrics['reconstruction_error']:.6f}")


if __name__ == "__main__":
    test_multi_axis_compressor()

