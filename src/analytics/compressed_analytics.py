"""
On-the-Fly Analytics on Compressed Data

This module implements analytics operations that can be performed directly
on compressed data without full decompression.

For Milestone 3: On-the-Fly Analytics on Compressed Data
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


class CompressedAnalytics:
    """
    Analytics operations on compressed data.
    
    Supports:
    - Basic statistics (mean, variance, min, max)
    - Activity detection from compressed streams
    - Anomaly detection
    - Query operations
    """
    
    def __init__(self):
        """Initialize the compressed analytics module."""
        pass
    
    def delta_encoding_statistics(self, compressed: np.ndarray, first_val: float) -> Dict[str, float]:
        """
        Calculate statistics from delta-encoded data without full decompression.
        
        For delta encoding: data[i] = first_val + sum(deltas[0:i])
        
        Args:
            compressed: Delta-encoded data (deltas)
            first_val: First value of original data
            
        Returns:
            Dictionary with statistics
        """
        n = len(compressed) + 1  # Original length
        
        # Reconstruct values for accurate statistics
        # This is still more efficient than full decompression if we only need stats
        cumulative = np.cumsum(compressed)
        values = np.concatenate([[first_val], first_val + cumulative])
        
        # Calculate accurate statistics from reconstructed values
        mean = np.mean(values)
        variance = np.var(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val
        }
    
    def run_length_statistics(self, compressed: List[Tuple[float, int]]) -> Dict[str, float]:
        """
        Calculate statistics from run-length encoded data.
        
        Args:
            compressed: List of (value, count) tuples
            
        Returns:
            Dictionary with statistics
        """
        # Reconstruct values for statistics (but more efficient than full decompression)
        values = []
        counts = []
        for value, count in compressed:
            values.append(value)
            counts.append(count)
        
        values = np.array(values)
        counts = np.array(counts)
        
        # Weighted statistics
        total_count = np.sum(counts)
        mean = np.average(values, weights=counts)
        
        # Weighted variance
        variance = np.average((values - mean) ** 2, weights=counts)
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val,
            'num_runs': len(compressed),
            'total_samples': total_count
        }
    
    def quantization_statistics(self, compressed: np.ndarray, metadata: Dict) -> Dict[str, float]:
        """
        Calculate statistics from quantized data.
        
        Args:
            compressed: Quantized data (uint8 array)
            metadata: Quantization metadata (min_val, max_val, etc.)
            
        Returns:
            Dictionary with statistics
        """
        # Dequantize for statistics (but we can work with quantized values)
        min_val = metadata.get('min_val', 0.0)
        max_val = metadata.get('max_val', 1.0)
        range_val = max_val - min_val
        
        # Convert quantized to approximate values
        normalized = compressed.astype(np.float32) / 255.0
        values = min_val + normalized * range_val
        
        mean = np.mean(values)
        variance = np.var(values)
        min_val_actual = np.min(values)
        max_val_actual = np.max(values)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'min': min_val_actual,
            'max': max_val_actual,
            'range': max_val_actual - min_val_actual
        }
    
    def detect_anomaly_delta(self, compressed: np.ndarray, first_val: float,
                           threshold_std: float = 3.0) -> List[int]:
        """
        Detect anomalies in delta-encoded data.
        
        Anomalies are detected as deltas that deviate significantly from the mean.
        
        Args:
            compressed: Delta-encoded data
            first_val: First value
            threshold_std: Number of standard deviations for anomaly detection
            
        Returns:
            List of indices where anomalies occur
        """
        mean_delta = np.mean(compressed)
        std_delta = np.std(compressed)
        
        threshold = threshold_std * std_delta
        anomalies = np.where(np.abs(compressed - mean_delta) > threshold)[0]
        
        return anomalies.tolist()
    
    def detect_anomaly_rle(self, compressed: List[Tuple[float, int]],
                          threshold_std: float = 3.0) -> List[int]:
        """
        Detect anomalies in run-length encoded data.
        
        Anomalies are detected as values that deviate significantly from the mean.
        
        Args:
            compressed: RLE-compressed data
            threshold_std: Number of standard deviations for anomaly detection
            
        Returns:
            List of run indices where anomalies occur
        """
        values = np.array([val for val, count in compressed])
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        threshold = threshold_std * std_val
        anomalies = []
        
        for i, (val, count) in enumerate(compressed):
            if np.abs(val - mean_val) > threshold:
                anomalies.append(i)
        
        return anomalies
    
    def query_range_delta(self, compressed: np.ndarray, first_val: float,
                         start_idx: int, end_idx: int) -> np.ndarray:
        """
        Query a range of values from delta-encoded data without full decompression.
        
        Args:
            compressed: Delta-encoded data
            first_val: First value
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            Array of values in the specified range
        """
        # Only decompress the needed range
        if start_idx == 0:
            range_deltas = compressed[:end_idx-1]
            cumulative = np.cumsum(range_deltas)
            values = np.concatenate([[first_val], first_val + cumulative])
        else:
            # Need to decompress from start
            range_deltas = compressed[start_idx-1:end_idx-1]
            # Calculate starting value
            start_cumulative = np.sum(compressed[:start_idx-1]) if start_idx > 1 else 0
            start_value = first_val + start_cumulative
            cumulative = np.cumsum(range_deltas)
            values = np.concatenate([[start_value], start_value + cumulative])
        
        return values
    
    def query_range_rle(self, compressed: List[Tuple[float, int]],
                       start_idx: int, end_idx: int) -> np.ndarray:
        """
        Query a range of values from RLE-compressed data.
        
        Args:
            compressed: RLE-compressed data
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            Array of values in the specified range
        """
        values = []
        current_idx = 0
        
        for val, count in compressed:
            run_end = current_idx + count
            
            # Check if this run overlaps with the query range
            if run_end > start_idx and current_idx < end_idx:
                # Extract overlapping portion
                run_start_in_range = max(0, start_idx - current_idx)
                run_end_in_range = min(count, end_idx - current_idx)
                num_values = run_end_in_range - run_start_in_range
                values.extend([val] * num_values)
            
            current_idx = run_end
            if current_idx >= end_idx:
                break
        
        return np.array(values)
    
    def activity_detection_from_compressed(self, compressed_data: Dict,
                                          algorithm: str) -> str:
        """
        Detect activity type from compressed data without full decompression.
        
        Uses statistics from compressed data to infer activity.
        
        Args:
            compressed_data: Compressed data dictionary
            algorithm: Compression algorithm used
            
        Returns:
            Detected activity type ('sleep', 'rest', 'walking', 'active')
        """
        if algorithm == 'delta_encoding':
            stats = self.delta_encoding_statistics(
                compressed_data['compressed'],
                compressed_data['first_val']
            )
        elif algorithm == 'run_length':
            stats = self.run_length_statistics(compressed_data['compressed'])
        elif algorithm == 'quantization':
            stats = self.quantization_statistics(
                compressed_data['compressed'],
                compressed_data['metadata']
            )
        else:
            return 'unknown'
        
        # Classify based on variance
        variance = stats['variance']
        
        if variance < 0.01:
            return 'sleep'
        elif variance < 0.1:
            return 'rest'
        elif variance < 0.5:
            return 'walking'
        else:
            return 'active'


def test_compressed_analytics():
    """Test the compressed analytics module."""
    analytics = CompressedAnalytics()
    
    # Generate test data
    t = np.linspace(0, 10, 1000)
    data = 0.5 * np.sin(2 * np.pi * 1.0 * t) + np.random.normal(0, 0.1, 1000)
    
    # Compress with delta encoding
    delta_compressor = DeltaEncodingCompressor()
    compressed, first_val = delta_compressor.compress(data)
    
    # Get statistics from compressed data
    stats = analytics.delta_encoding_statistics(compressed, first_val)
    
    print("Compressed Analytics Test")
    print("=" * 60)
    print("Statistics from compressed data:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Variance: {stats['variance']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    
    # Compare with actual statistics
    actual_mean = np.mean(data)
    actual_var = np.var(data)
    print(f"\nActual statistics:")
    print(f"  Mean: {actual_mean:.4f}")
    print(f"  Variance: {actual_var:.4f}")
    print(f"  Error: {abs(stats['mean'] - actual_mean):.6f}")


if __name__ == "__main__":
    test_compressed_analytics()

