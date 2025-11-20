"""
Adaptive Compressor

This module implements an adaptive compression system that automatically selects
the best compression algorithm based on detected activity type.

For Milestone 3: Activity-Aware Adaptive Compression
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
from activity.activity_detector import ActivityDetector, ActivityType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveCompressor:
    """
    Adaptive compressor that selects algorithms based on activity type.
    
    Activity-to-algorithm mapping:
    - Sleep/Rest → Run-Length Encoding (excellent for static periods)
    - Active movement → Delta Encoding (reliable, consistent)
    - High compression needs → Quantization (8x compression)
    - Mixed/Unknown → Delta Encoding (safe default)
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the adaptive compressor.
        
        Args:
            sampling_rate: Sampling rate in Hz (default 100Hz for CAPTURE-24)
        """
        self.sampling_rate = sampling_rate
        self.activity_detector = ActivityDetector(sampling_rate=sampling_rate)
        
        # Initialize compressors
        self.delta_compressor = DeltaEncodingCompressor()
        self.rle_compressor = RunLengthCompressor(threshold=0.01)
        self.quant_compressor = QuantizationCompressor(n_bits=8, method='uniform')
        
        # Activity-to-algorithm mapping
        self.activity_mapping = {
            ActivityType.SLEEP: 'run_length',
            ActivityType.REST: 'run_length',
            ActivityType.WALKING: 'delta_encoding',
            ActivityType.ACTIVE: 'delta_encoding',
            ActivityType.MIXED: 'delta_encoding'
        }
        
        # Statistics
        self.stats = {
            'total_segments': 0,
            'algorithm_usage': {'delta_encoding': 0, 'run_length': 0, 'quantization': 0},
            'activity_distribution': {act.value: 0 for act in ActivityType}
        }
    
    def select_algorithm(self, activity: ActivityType, 
                        compression_target: Optional[str] = None) -> str:
        """
        Select compression algorithm based on activity and optional target.
        
        Args:
            activity: Detected activity type
            compression_target: Optional target ('ratio', 'quality', 'speed')
            
        Returns:
            Algorithm name to use
        """
        # Override with compression target if specified
        if compression_target == 'ratio':
            # Prioritize compression ratio
            if activity in [ActivityType.SLEEP, ActivityType.REST]:
                return 'run_length'  # Can achieve very high ratios
            else:
                return 'quantization'  # Consistent 8x
        elif compression_target == 'quality':
            # Prioritize quality (lossless)
            return 'delta_encoding'  # Perfect reconstruction
        elif compression_target == 'speed':
            # Prioritize speed
            return 'delta_encoding'  # Fastest
        
        # Default: use activity-based selection
        return self.activity_mapping.get(activity, 'delta_encoding')
    
    def compress(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 compression_target: Optional[str] = None) -> Dict:
        """
        Compress triaxial accelerometer data using adaptive algorithm selection.
        
        Args:
            x: X-axis accelerometer data
            y: Y-axis accelerometer data
            z: Z-axis accelerometer data
            compression_target: Optional target ('ratio', 'quality', 'speed')
            
        Returns:
            Dictionary with compressed data, algorithm used, and metadata
        """
        # Detect activity
        activity = self.activity_detector.detect_activity(x, y, z)
        
        # Select algorithm
        algorithm = self.select_algorithm(activity, compression_target)
        
        # Update statistics
        self.stats['total_segments'] += 1
        self.stats['algorithm_usage'][algorithm] += 1
        self.stats['activity_distribution'][activity.value] += 1
        
        # Compress each axis with selected algorithm
        compressed_data = {}
        compression_ratios = []
        reconstruction_errors = []
        
        for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
            if algorithm == 'delta_encoding':
                compressed, first_val = self.delta_compressor.compress(axis_data)
                decompressed = self.delta_compressor.decompress(compressed, first_val)
                ratio = self.delta_compressor.compression_ratio
                error = self.delta_compressor.reconstruction_error
                
            elif algorithm == 'run_length':
                compressed, original_len = self.rle_compressor.compress(axis_data)
                decompressed = self.rle_compressor.decompress(compressed, original_len)
                ratio = self.rle_compressor.compression_ratio
                error = 0.0  # RLE is lossless for exact matches
                
            elif algorithm == 'quantization':
                compressed, metadata = self.quant_compressor.compress(axis_data)
                decompressed = self.quant_compressor.decompress(compressed, metadata)
                ratio = self.quant_compressor.compression_ratio
                error = self.quant_compressor.reconstruction_error
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            compressed_data[axis_name] = {
                'data': compressed,
                'algorithm': algorithm,
                'metadata': {
                    'first_val': first_val if algorithm == 'delta_encoding' else None,
                    'original_len': original_len if algorithm == 'run_length' else None,
                    'quant_metadata': metadata if algorithm == 'quantization' else None
                }
            }
            compression_ratios.append(ratio)
            reconstruction_errors.append(error)
        
        return {
            'compressed_data': compressed_data,
            'algorithm': algorithm,
            'activity': activity.value,
            'compression_ratio': np.mean(compression_ratios),
            'reconstruction_error': np.mean(reconstruction_errors),
            'axis_ratios': {'x': compression_ratios[0], 'y': compression_ratios[1], 'z': compression_ratios[2]}
        }
    
    def compress_single_axis(self, data: np.ndarray,
                            compression_target: Optional[str] = None) -> Dict:
        """
        Compress single-axis data using adaptive algorithm selection.
        
        Args:
            data: Single-axis accelerometer data
            compression_target: Optional target ('ratio', 'quality', 'speed')
            
        Returns:
            Dictionary with compressed data and metadata
        """
        # Detect activity from single axis
        activity = self.activity_detector.detect_activity_from_single_axis(data)
        
        # Select algorithm
        algorithm = self.select_algorithm(activity, compression_target)
        
        # Update statistics
        self.stats['total_segments'] += 1
        self.stats['algorithm_usage'][algorithm] += 1
        self.stats['activity_distribution'][activity.value] += 1
        
        # Compress
        if algorithm == 'delta_encoding':
            compressed, first_val = self.delta_compressor.compress(data)
            decompressed = self.delta_compressor.decompress(compressed, first_val)
            ratio = self.delta_compressor.compression_ratio
            error = self.delta_compressor.reconstruction_error
            metadata = {'first_val': first_val}
            
        elif algorithm == 'run_length':
            compressed, original_len = self.rle_compressor.compress(data)
            decompressed = self.rle_compressor.decompress(compressed, original_len)
            ratio = self.rle_compressor.compression_ratio
            error = 0.0
            metadata = {'original_len': original_len}
            
        elif algorithm == 'quantization':
            compressed, quant_metadata = self.quant_compressor.compress(data)
            decompressed = self.quant_compressor.decompress(compressed, quant_metadata)
            ratio = self.quant_compressor.compression_ratio
            error = self.quant_compressor.reconstruction_error
            metadata = quant_metadata
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return {
            'compressed_data': compressed,
            'algorithm': algorithm,
            'activity': activity.value,
            'compression_ratio': ratio,
            'reconstruction_error': error,
            'metadata': metadata
        }
    
    def get_statistics(self) -> Dict:
        """Get compression statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset compression statistics."""
        self.stats = {
            'total_segments': 0,
            'algorithm_usage': {'delta_encoding': 0, 'run_length': 0, 'quantization': 0},
            'activity_distribution': {act.value: 0 for act in ActivityType}
        }


def test_adaptive_compressor():
    """Test the adaptive compressor."""
    compressor = AdaptiveCompressor()
    
    # Generate test data
    t = np.linspace(0, 10, 1000)
    
    # Sleep data (low variation)
    sleep_x = np.random.normal(0, 0.01, 1000)
    sleep_y = np.random.normal(0.8, 0.01, 1000)
    sleep_z = np.random.normal(0.6, 0.01, 1000)
    
    # Active data (high variation)
    active_x = 0.5 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 0.3, 1000)
    active_y = 0.8 + 0.5 * np.cos(2 * np.pi * 5.0 * t) + np.random.normal(0, 0.3, 1000)
    active_z = 0.6 + 0.4 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 0.3, 1000)
    
    print("Adaptive Compressor Test")
    print("=" * 60)
    
    # Test sleep data
    result_sleep = compressor.compress(sleep_x, sleep_y, sleep_z)
    print(f"Sleep data: {result_sleep['algorithm']} (ratio: {result_sleep['compression_ratio']:.2f}x)")
    
    # Test active data
    result_active = compressor.compress(active_x, active_y, active_z)
    print(f"Active data: {result_active['algorithm']} (ratio: {result_active['compression_ratio']:.2f}x)")
    
    # Print statistics
    stats = compressor.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Algorithm usage: {stats['algorithm_usage']}")
    print(f"  Activity distribution: {stats['activity_distribution']}")


if __name__ == "__main__":
    test_adaptive_compressor()

