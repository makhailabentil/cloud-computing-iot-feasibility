"""
Systematic Evaluation Module for Compression Algorithms

This module provides comprehensive evaluation of compression algorithms on
real and synthetic IoT sensor data, measuring compression ratio, reconstruction
error, and resource consumption.

For Milestone 2, this implements systematic evaluation of delta encoding,
run-length encoding, and quantization on CAPTURE-24 data segments.
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json
from dataclasses import dataclass, asdict
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from compressors.delta_encoding import DeltaEncodingCompressor
from compressors.run_length import RunLengthCompressor
from compressors.quantization import QuantizationCompressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Comprehensive metrics for compression evaluation."""
    algorithm: str
    compression_ratio: float
    compression_time: float
    decompression_time: float
    original_size: int
    compressed_size: int
    reconstruction_error: float
    mse: float
    rmse: float
    mae: float
    max_error: float
    snr_db: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SystematicEvaluator:
    """
    Systematic evaluator for compression algorithms on IoT sensor data.
    
    This class implements comprehensive evaluation including:
    - Compression ratio measurement
    - Reconstruction error calculation
    - Resource consumption monitoring
    - Performance comparison across algorithms
    """
    
    def __init__(self, output_dir: str = "results/evaluation"):
        """
        Initialize the systematic evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.compressors = {
            'delta_encoding': DeltaEncodingCompressor(),
            'run_length': RunLengthCompressor(),
            'quantization': QuantizationCompressor(n_bits=8, method='uniform')
        }
        
        self.results = []
        
    def evaluate_compressor(self, data: np.ndarray, 
                           algorithm: str,
                           axis_name: str = "unknown") -> CompressionMetrics:
        """
        Evaluate a single compression algorithm on a data segment.
        
        Args:
            data: Input time series data
            algorithm: Algorithm name ('delta_encoding', 'run_length', 'quantization')
            axis_name: Name of the data axis (for logging)
            
        Returns:
            CompressionMetrics object with all evaluation metrics
        """
        if algorithm not in self.compressors:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        compressor = self.compressors[algorithm]
        original_size = len(data) * 8  # 8 bytes per float
        
        # Monitor resource usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        # Compress
        start_time = time.time()
        
        if algorithm == 'delta_encoding':
            compressed_data, first_value = compressor.compress(data)
            compressed_size = len(compressed_data) * 4 + 8  # 4 bytes per delta + first value
            compression_time = time.time() - start_time
            
            # Decompress
            decompress_start = time.time()
            reconstructed = compressor.decompress(compressed_data, first_value)
            decompression_time = time.time() - decompress_start
            
            # Calculate errors
            error = compressor.calculate_reconstruction_error(data, reconstructed)
            mse = error
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(data - reconstructed))
            max_error = np.max(np.abs(data - reconstructed))
            
            # SNR calculation
            signal_power = np.mean(data ** 2)
            noise_power = mse
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
        elif algorithm == 'run_length':
            compressed_data, original_length = compressor.compress(data)
            compressed_size = len(compressed_data) * (8 + 4)  # 8 bytes value + 4 bytes count
            compression_time = time.time() - start_time
            
            # Decompress
            decompress_start = time.time()
            reconstructed = compressor.decompress(compressed_data, original_length)
            decompression_time = time.time() - decompress_start
            
            # Calculate errors (should be perfect for RLE)
            mse = np.mean((data - reconstructed) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(data - reconstructed))
            max_error = np.max(np.abs(data - reconstructed))
            error = mse
            
            # SNR calculation
            signal_power = np.mean(data ** 2)
            noise_power = mse
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
        elif algorithm == 'quantization':
            # Fit quantizer if needed
            if not hasattr(compressor, 'quantization_levels') or compressor.quantization_levels is None:
                compressor.fit(data)
            
            compressed_data, metadata = compressor.compress(data)
            compressed_size = len(compressed_data) * (compressor.n_bits // 8)
            compression_time = time.time() - start_time
            
            # Decompress
            decompress_start = time.time()
            reconstructed = compressor.decompress(compressed_data, metadata)
            decompression_time = time.time() - decompress_start
            
            # Calculate errors
            errors = compressor.calculate_quantization_error(data, reconstructed)
            mse = errors['mse']
            rmse = errors['rmse']
            mae = errors['mae']
            max_error = errors['max_error']
            snr_db = errors['snr_db']
            error = mse
        
        # Measure resource usage
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        cpu_usage = process.cpu_percent() - cpu_before
        
        # Calculate compression ratio
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        metrics = CompressionMetrics(
            algorithm=algorithm,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            decompression_time=decompression_time,
            original_size=original_size,
            compressed_size=compressed_size,
            reconstruction_error=error,
            mse=mse,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            snr_db=snr_db,
            memory_usage_mb=memory_usage,
            cpu_percent=cpu_usage
        )
        
        logger.info(f"{algorithm} on {axis_name}: "
                   f"ratio={compression_ratio:.2f}x, "
                   f"error={error:.6f}, "
                   f"time={compression_time:.4f}s")
        
        return metrics
    
    def evaluate_all_algorithms(self, data: np.ndarray, 
                               axis_name: str = "unknown",
                               participant_id: str = "unknown") -> List[CompressionMetrics]:
        """
        Evaluate all compression algorithms on a data segment.
        
        Args:
            data: Input time series data
            axis_name: Name of the data axis
            participant_id: Participant ID for tracking
            
        Returns:
            List of CompressionMetrics for each algorithm
        """
        results = []
        
        for algorithm in self.compressors.keys():
            try:
                metrics = self.evaluate_compressor(data, algorithm, axis_name)
                results.append(metrics)
                self.results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {algorithm} on {axis_name}: {e}")
        
        return results
    
    def evaluate_segments(self, segments: List[np.ndarray],
                         axis_name: str = "unknown",
                         participant_id: str = "unknown") -> pd.DataFrame:
        """
        Evaluate compression on multiple data segments and aggregate results.
        
        Args:
            segments: List of data segments to evaluate
            axis_name: Name of the data axis
            participant_id: Participant ID
            
        Returns:
            DataFrame with aggregated results
        """
        all_results = []
        
        for i, segment in enumerate(segments):
            logger.info(f"Evaluating segment {i+1}/{len(segments)} for {axis_name}")
            results = self.evaluate_all_algorithms(segment, axis_name, participant_id)
            all_results.extend(results)
        
        # Convert to DataFrame
        df = pd.DataFrame([m.to_dict() for m in all_results])
        
        return df
    
    def compare_algorithms(self, data: np.ndarray,
                          axis_name: str = "unknown") -> pd.DataFrame:
        """
        Compare all algorithms on the same data segment.
        
        Args:
            data: Input time series data
            axis_name: Name of the data axis
            
        Returns:
            DataFrame with comparison results
        """
        results = self.evaluate_all_algorithms(data, axis_name)
        df = pd.DataFrame([m.to_dict() for m in results])
        
        return df
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """
        Save evaluation results to file.
        
        Args:
            filename: Output filename
        """
        output_file = self.output_dir / filename
        
        # Convert results to JSON-serializable format
        results_dict = {
            'evaluations': [m.to_dict() for m in self.results],
            'summary': self.get_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all evaluations.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        df = pd.DataFrame([m.to_dict() for m in self.results])
        
        summary = {}
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]
            summary[algorithm] = {
                'mean_compression_ratio': float(alg_data['compression_ratio'].mean()),
                'std_compression_ratio': float(alg_data['compression_ratio'].std()),
                'mean_reconstruction_error': float(alg_data['reconstruction_error'].mean()),
                'mean_compression_time': float(alg_data['compression_time'].mean()),
                'mean_memory_usage_mb': float(alg_data['memory_usage_mb'].mean()),
                'n_evaluations': int(len(alg_data))
            }
        
        return summary
    
    def generate_report(self, output_file: str = "evaluation_report.md"):
        """
        Generate a markdown report of evaluation results.
        
        Args:
            output_file: Output filename
        """
        if not self.results:
            logger.warning("No results to report")
            return
        
        report_file = self.output_dir / output_file
        
        df = pd.DataFrame([m.to_dict() for m in self.results])
        summary = self.get_summary()
        
        with open(report_file, 'w') as f:
            f.write("# Compression Algorithm Evaluation Report\n\n")
            f.write("## Summary Statistics\n\n")
            
            for algorithm, stats in summary.items():
                f.write(f"### {algorithm.replace('_', ' ').title()}\n\n")
                f.write(f"- Mean Compression Ratio: {stats['mean_compression_ratio']:.2f}x "
                       f"(±{stats['std_compression_ratio']:.2f})\n")
                f.write(f"- Mean Reconstruction Error (MSE): {stats['mean_reconstruction_error']:.6f}\n")
                f.write(f"- Mean Compression Time: {stats['mean_compression_time']:.4f}s\n")
                f.write(f"- Mean Memory Usage: {stats['mean_memory_usage_mb']:.2f} MB\n")
                f.write(f"- Number of Evaluations: {stats['n_evaluations']}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write(df.to_markdown(index=False))
        
        logger.info(f"Report saved to {report_file}")


def test_evaluator():
    """Test the systematic evaluator."""
    print("Testing Systematic Evaluator")
    print("=" * 60)
    
    evaluator = SystematicEvaluator()
    
    # Generate test data
    print("\n1. Generating test data...")
    test_data = np.sin(np.linspace(0, 4*np.pi, 10000)) + np.random.normal(0, 0.1, 10000)
    print(f"   Generated {len(test_data)} samples")
    
    # Evaluate all algorithms
    print("\n2. Evaluating all algorithms...")
    results = evaluator.compare_algorithms(test_data, axis_name="test_signal")
    print("\nResults:")
    print(results[['algorithm', 'compression_ratio', 'reconstruction_error', 
                   'compression_time', 'memory_usage_mb']].to_string(index=False))
    
    # Test segmentation evaluation
    print("\n3. Testing segmentation evaluation...")
    segments = [test_data[i:i+1000] for i in range(0, len(test_data), 1000)]
    segment_results = evaluator.evaluate_segments(segments, axis_name="test_signal")
    print(f"   Evaluated {len(segments)} segments")
    
    # Generate summary
    print("\n4. Generating summary...")
    summary = evaluator.get_summary()
    print("\nSummary:")
    for alg, stats in summary.items():
        print(f"  {alg}: {stats['mean_compression_ratio']:.2f}x compression, "
              f"{stats['mean_reconstruction_error']:.6f} MSE")
    
    # Save results
    evaluator.save_results()
    evaluator.generate_report()
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    test_evaluator()

