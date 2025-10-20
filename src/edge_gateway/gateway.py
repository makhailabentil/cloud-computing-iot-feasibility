"""
Edge Gateway for IoT Data Compression

This module implements a simple edge gateway that compresses sensor data
before forwarding it to a cloud collector, demonstrating the practical
application of compression algorithms in IoT systems.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Import our compression modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from compressors.delta_encoding import DeltaEncodingCompressor
from compressors.run_length import RunLengthCompressor
from compressors.quantization import QuantizationCompressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics for compression performance."""
    algorithm: str
    compression_ratio: float
    compression_time: float
    decompression_time: float
    original_size: int
    compressed_size: int
    reconstruction_error: float = 0.0


class EdgeGateway:
    """
    Edge gateway that compresses IoT sensor data before cloud transmission.
    
    This gateway demonstrates the practical application of compression
    algorithms in IoT systems, where edge devices need to reduce data
    transmission costs while maintaining data quality.
    """
    
    def __init__(self, gateway_id: str = "gateway_001", 
                 compression_algorithm: str = "delta_encoding"):
        """
        Initialize the edge gateway.
        
        Args:
            gateway_id: Unique identifier for this gateway
            compression_algorithm: Compression algorithm to use
        """
        self.gateway_id = gateway_id
        self.compression_algorithm = compression_algorithm
        self.compressor = self._initialize_compressor()
        self.stats = []
        self.data_buffer = []
        self.buffer_size = 1000  # Buffer size before compression
        
        logger.info(f"Initialized edge gateway {gateway_id} with {compression_algorithm}")
    
    def _initialize_compressor(self):
        """Initialize the compression algorithm."""
        if self.compression_algorithm == "delta_encoding":
            return DeltaEncodingCompressor()
        elif self.compression_algorithm == "run_length":
            return RunLengthCompressor()
        elif self.compression_algorithm == "quantization":
            return QuantizationCompressor(n_bits=8, method='uniform')
        else:
            raise ValueError(f"Unknown compression algorithm: {self.compression_algorithm}")
    
    def add_sensor_data(self, sensor_id: str, timestamp: datetime, 
                       values: Dict[str, float]) -> None:
        """
        Add sensor data to the gateway buffer.
        
        Args:
            sensor_id: ID of the sensor
            timestamp: Timestamp of the data
            values: Dictionary of sensor values
        """
        data_point = {
            'sensor_id': sensor_id,
            'timestamp': timestamp.isoformat(),
            'values': values
        }
        
        self.data_buffer.append(data_point)
        
        # Compress and send if buffer is full
        if len(self.data_buffer) >= self.buffer_size:
            self._compress_and_send()
    
    def _compress_and_send(self) -> CompressionStats:
        """
        Compress buffered data and send to cloud.
        
        Returns:
            Compression statistics
        """
        if not self.data_buffer:
            return None
        
        logger.info(f"Compressing {len(self.data_buffer)} data points")
        
        # Extract time series data for compression
        timestamps = [point['timestamp'] for point in self.data_buffer]
        sensor_values = {}
        
        # Group values by sensor type
        for point in self.data_buffer:
            for sensor_type, value in point['values'].items():
                if sensor_type not in sensor_values:
                    sensor_values[sensor_type] = []
                sensor_values[sensor_type].append(value)
        
        # Compress each sensor type
        compression_results = {}
        total_original_size = 0
        total_compressed_size = 0
        total_compression_time = 0
        
        for sensor_type, values in sensor_values.items():
            values_array = np.array(values)
            original_size = len(values_array) * 8  # 8 bytes per float
            
            # Compress
            start_time = time.time()
            if self.compression_algorithm == "delta_encoding":
                compressed_data, first_value = self.compressor.compress(values_array)
                compressed_size = len(compressed_data) * 8 + 8  # deltas + first value
            elif self.compression_algorithm == "run_length":
                compressed_data, original_length = self.compressor.compress(values_array)
                compressed_size = len(compressed_data) * (8 + 4)  # value + count pairs
            elif self.compression_algorithm == "quantization":
                # Fit quantizer if not already fitted
                if not hasattr(self.compressor, 'quantization_levels') or self.compressor.quantization_levels is None:
                    self.compressor.fit(values_array)
                compressed_data, metadata = self.compressor.compress(values_array)
                compressed_size = len(compressed_data) * (self.compressor.n_bits // 8)
            
            compression_time = time.time() - start_time
            
            compression_results[sensor_type] = {
                'compressed_data': compressed_data,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_time': compression_time
            }
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            total_compression_time += compression_time
        
        # Calculate overall compression ratio
        compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0
        
        # Create compression stats
        stats = CompressionStats(
            algorithm=self.compression_algorithm,
            compression_ratio=compression_ratio,
            compression_time=total_compression_time,
            decompression_time=0.0,  # Would be measured during decompression
            original_size=total_original_size,
            compressed_size=total_compressed_size
        )
        
        self.stats.append(stats)
        
        # Simulate sending to cloud (in practice, this would be network transmission)
        self._send_to_cloud(compression_results, timestamps)
        
        # Clear buffer
        self.data_buffer = []
        
        logger.info(f"Compression completed: {compression_ratio:.2f}x ratio, {total_compression_time:.4f}s")
        
        return stats
    
    def _send_to_cloud(self, compression_results: Dict, timestamps: List[str]) -> None:
        """
        Simulate sending compressed data to cloud collector.
        
        Args:
            compression_results: Compressed data for each sensor type
            timestamps: List of timestamps
        """
        # In a real implementation, this would involve:
        # 1. Serializing compressed data
        # 2. Adding metadata (gateway_id, compression_algorithm, etc.)
        # 3. Sending via network protocol (MQTT, HTTP, etc.)
        
        cloud_payload = {
            'gateway_id': self.gateway_id,
            'compression_algorithm': self.compression_algorithm,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(timestamps),
            'compressed_data': compression_results
        }
        
        # Simulate network delay
        time.sleep(0.001)  # 1ms delay
        
        logger.info(f"Sent compressed data to cloud: {len(compression_results)} sensor types")
    
    def get_compression_stats(self) -> List[CompressionStats]:
        """Get compression statistics."""
        return self.stats
    
    def get_average_compression_ratio(self) -> float:
        """Get average compression ratio across all compressions."""
        if not self.stats:
            return 0.0
        
        return sum(stat.compression_ratio for stat in self.stats) / len(self.stats)
    
    def force_compression(self) -> Optional[CompressionStats]:
        """
        Force compression of current buffer (even if not full).
        
        Returns:
            Compression statistics if data was compressed
        """
        if self.data_buffer:
            return self._compress_and_send()
        return None
    
    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self.stats = []
        logger.info("Compression statistics reset")


def simulate_iot_data_stream(gateway: EdgeGateway, duration_seconds: int = 60,
                            sampling_rate: float = 1.0) -> None:
    """
    Simulate an IoT data stream for testing the edge gateway.
    
    Args:
        gateway: Edge gateway instance
        duration_seconds: Duration of simulation in seconds
        sampling_rate: Data sampling rate in Hz
    """
    logger.info(f"Starting IoT data stream simulation for {duration_seconds} seconds")
    
    start_time = time.time()
    sample_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Generate synthetic sensor data
        current_time = datetime.now()
        
        # Simulate multiple sensors
        sensor_data = {
            'temperature': 20 + 5 * np.sin(2 * np.pi * sample_count * 0.01) + np.random.normal(0, 0.5),
            'humidity': 50 + 10 * np.sin(2 * np.pi * sample_count * 0.005) + np.random.normal(0, 2),
            'pressure': 1013 + 2 * np.sin(2 * np.pi * sample_count * 0.02) + np.random.normal(0, 1),
            'accelerometer_x': 0.1 * np.sin(2 * np.pi * sample_count * 0.1) + np.random.normal(0, 0.05),
            'accelerometer_y': 0.1 * np.cos(2 * np.pi * sample_count * 0.1) + np.random.normal(0, 0.05),
            'accelerometer_z': 9.8 + 0.1 * np.sin(2 * np.pi * sample_count * 0.05) + np.random.normal(0, 0.1)
        }
        
        # Add data to gateway
        gateway.add_sensor_data("sensor_001", current_time, sensor_data)
        
        sample_count += 1
        
        # Wait for next sample
        time.sleep(1.0 / sampling_rate)
    
    # Force compression of remaining data
    gateway.force_compression()
    
    logger.info(f"Simulation completed: {sample_count} samples processed")


def test_edge_gateway():
    """
    Test the edge gateway functionality.
    """
    print("Testing Edge Gateway")
    print("=" * 40)
    
    # Test different compression algorithms
    algorithms = ["delta_encoding", "run_length", "quantization"]
    
    for algorithm in algorithms:
        print(f"\n--- Testing {algorithm.upper()} ---")
        
        # Create gateway
        gateway = EdgeGateway(gateway_id=f"test_gateway_{algorithm}", 
                            compression_algorithm=algorithm)
        
        # Simulate data stream
        simulate_iot_data_stream(gateway, duration_seconds=10, sampling_rate=2.0)
        
        # Get statistics
        stats = gateway.get_compression_stats()
        if stats:
            avg_ratio = gateway.get_average_compression_ratio()
            print(f"Average compression ratio: {avg_ratio:.2f}x")
            print(f"Total compressions: {len(stats)}")
            print(f"Total data points processed: {sum(len(gateway.data_buffer) for _ in range(1))}")


if __name__ == "__main__":
    test_edge_gateway()
