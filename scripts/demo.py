#!/usr/bin/env python3
"""
Demo script for Cloud Computing IoT Feasibility Study

This script demonstrates the compression algorithms and edge gateway
functionality implemented in this project.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time

# Add src to path for imports (scripts/ is one level deeper than root)
sys.path.append(str(Path(__file__).parent.parent / "src"))

from compressors.delta_encoding import DeltaEncodingCompressor, generate_synthetic_data
from compressors.run_length import RunLengthCompressor, generate_repetitive_data
from compressors.quantization import QuantizationCompressor, generate_sensor_data
from data_processing.trace_replay import SensorTraceReplayer
from edge_gateway.gateway import EdgeGateway, simulate_iot_data_stream


def demo_compression_algorithms():
    """Demonstrate all compression algorithms."""
    print("=" * 60)
    print("COMPRESSION ALGORITHMS DEMO")
    print("=" * 60)
    
    # Generate test data
    print("\n1. Generating test data...")
    synthetic_data = generate_synthetic_data(n_points=1000)
    repetitive_data = generate_repetitive_data(n_points=1000, repetition_factor=0.4)
    sensor_data = generate_sensor_data(n_points=1000, noise_level=0.5)
    
    print(f"   - Synthetic data: {len(synthetic_data)} points")
    print(f"   - Repetitive data: {len(repetitive_data)} points")
    print(f"   - Sensor data: {len(sensor_data)} points")
    
    # Test Delta Encoding
    print("\n2. Testing Delta Encoding Compressor...")
    delta_compressor = DeltaEncodingCompressor()
    
    start_time = time.time()
    delta_compressed, delta_first = delta_compressor.compress(synthetic_data)
    delta_time = time.time() - start_time
    
    delta_reconstructed = delta_compressor.decompress(delta_compressed, delta_first)
    delta_error = delta_compressor.calculate_reconstruction_error(synthetic_data, delta_reconstructed)
    
    print(f"   - Compression ratio: {delta_compressor.compression_ratio:.2f}x")
    print(f"   - Compression time: {delta_time:.4f}s")
    print(f"   - Reconstruction error: {delta_error:.6f}")
    print(f"   - Data integrity: {'PASS' if np.allclose(synthetic_data, delta_reconstructed, rtol=1e-10) else 'FAIL'}")
    
    # Test Run Length Encoding
    print("\n3. Testing Run Length Encoding Compressor...")
    rle_compressor = RunLengthCompressor()
    
    start_time = time.time()
    rle_compressed, rle_length = rle_compressor.compress(repetitive_data)
    rle_time = time.time() - start_time
    
    rle_reconstructed = rle_compressor.decompress(rle_compressed, rle_length)
    
    print(f"   - Compression ratio: {rle_compressor.compression_ratio:.2f}x")
    print(f"   - Compression time: {rle_time:.4f}s")
    print(f"   - Data integrity: {'PASS' if np.allclose(repetitive_data, rle_reconstructed, rtol=1e-10) else 'FAIL'}")
    
    # Test Quantization
    print("\n4. Testing Quantization Compressor...")
    quant_compressor = QuantizationCompressor(n_bits=8, method='uniform')
    
    start_time = time.time()
    quant_compressor.fit(sensor_data)
    quant_compressed, quant_metadata = quant_compressor.compress(sensor_data)
    quant_time = time.time() - start_time
    
    quant_reconstructed = quant_compressor.decompress(quant_compressed, quant_metadata)
    quant_errors = quant_compressor.calculate_quantization_error(sensor_data, quant_reconstructed)
    
    print(f"   - Compression ratio: {quant_compressor.compression_ratio:.2f}x")
    print(f"   - Compression time: {quant_time:.4f}s")
    print(f"   - RMSE: {quant_errors['rmse']:.4f}")
    print(f"   - SNR: {quant_errors['snr_db']:.2f} dB")


def demo_trace_replayer():
    """Demonstrate sensor trace replayer functionality."""
    print("\n" + "=" * 60)
    print("SENSOR TRACE REPLAYER DEMO")
    print("=" * 60)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Initialize replayer
    replayer = SensorTraceReplayer()
    
    print("\n1. Generating synthetic sensor traces...")
    
    # Generate different types of synthetic traces
    sensor_trace = replayer.generate_synthetic_trace(n_points=2000, signal_type='sensor')
    accel_trace = replayer.generate_synthetic_trace(n_points=3000, signal_type='accelerometer')
    temp_trace = replayer.generate_synthetic_trace(n_points=5000, signal_type='temperature')
    
    print(f"   - Generated {len(replayer.list_traces())} synthetic traces")
    
    # Test trace replay
    print("\n2. Testing trace replay...")
    for trace_id in replayer.list_traces():
        info = replayer.get_trace_info(trace_id)
        print(f"   - Trace '{trace_id}': {info['n_points']} points, {info['sampling_rate']:.2f} Hz")
        
        # Replay first few chunks
        chunk_count = 0
        for chunk in replayer.replay_trace(trace_id, chunk_size=500):
            chunk_count += 1
            if chunk_count <= 2:  # Show first 2 chunks
                print(f"     Chunk {chunk_count}: {len(chunk)} points, range: [{chunk.min():.2f}, {chunk.max():.2f}]")
            if chunk_count >= 3:
                break


def demo_edge_gateway():
    """Demonstrate edge gateway functionality."""
    print("\n" + "=" * 60)
    print("EDGE GATEWAY DEMO")
    print("=" * 60)
    
    # Test different compression algorithms
    algorithms = ["delta_encoding", "run_length", "quantization"]
    
    for algorithm in algorithms:
        print(f"\n1. Testing Edge Gateway with {algorithm.upper()}...")
        
        # Create gateway
        gateway = EdgeGateway(gateway_id=f"demo_gateway_{algorithm}", 
                            compression_algorithm=algorithm)
        
        # Simulate short data stream
        print("   - Simulating IoT data stream...")
        simulate_iot_data_stream(gateway, duration_seconds=5, sampling_rate=2.0)
        
        # Get statistics
        stats = gateway.get_compression_stats()
        if stats:
            avg_ratio = gateway.get_average_compression_ratio()
            print(f"   - Average compression ratio: {avg_ratio:.2f}x")
            print(f"   - Total compressions: {len(stats)}")
            print(f"   - Total compression time: {sum(s.compression_time for s in stats):.4f}s")


def demo_performance_comparison():
    """Compare performance of different compression methods."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Generate test data
    test_data = np.sin(np.linspace(0, 4*np.pi, 5000)) + np.random.normal(0, 0.1, 5000)
    
    print(f"\nTest data: {len(test_data)} points")
    print("Compression results:")
    
    # Delta Encoding
    delta_compressor = DeltaEncodingCompressor()
    start_time = time.time()
    delta_compressed, delta_first = delta_compressor.compress(test_data)
    delta_time = time.time() - start_time
    delta_reconstructed = delta_compressor.decompress(delta_compressed, delta_first)
    delta_error = delta_compressor.calculate_reconstruction_error(test_data, delta_reconstructed)
    
    print(f"\nDelta Encoding:")
    print(f"  - Compression ratio: {delta_compressor.compression_ratio:.2f}x")
    print(f"  - Compression time: {delta_time:.4f}s")
    print(f"  - Reconstruction error: {delta_error:.6f}")
    
    # Run Length Encoding
    rle_compressor = RunLengthCompressor()
    start_time = time.time()
    rle_compressed, rle_length = rle_compressor.compress(test_data)
    rle_time = time.time() - start_time
    rle_reconstructed = rle_compressor.decompress(rle_compressed, rle_length)
    
    print(f"\nRun Length Encoding:")
    print(f"  - Compression ratio: {rle_compressor.compression_ratio:.2f}x")
    print(f"  - Compression time: {rle_time:.4f}s")
    
    # Quantization
    quant_compressor = QuantizationCompressor(n_bits=8, method='uniform')
    start_time = time.time()
    quant_compressor.fit(test_data)
    quant_compressed, quant_metadata = quant_compressor.compress(test_data)
    quant_time = time.time() - start_time
    quant_reconstructed = quant_compressor.decompress(quant_compressed, quant_metadata)
    quant_errors = quant_compressor.calculate_quantization_error(test_data, quant_reconstructed)
    
    print(f"\nQuantization (8-bit):")
    print(f"  - Compression ratio: {quant_compressor.compression_ratio:.2f}x")
    print(f"  - Compression time: {quant_time:.4f}s")
    print(f"  - RMSE: {quant_errors['rmse']:.4f}")
    print(f"  - SNR: {quant_errors['snr_db']:.2f} dB")


def main():
    """Run the complete demo."""
    print("Cloud Computing IoT Feasibility Study - Demo")
    print("=" * 60)
    print("This demo showcases the compression algorithms and edge gateway")
    print("functionality implemented for IoT data compression research.")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_compression_algorithms()
        demo_trace_replayer()
        demo_edge_gateway()
        demo_performance_comparison()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Achievements:")
        print("✓ Implemented delta encoding compressor (3x compression)")
        print("✓ Implemented run length encoding compressor")
        print("✓ Implemented quantization compressor")
        print("✓ Created sensor trace replayer for CSV data")
        print("✓ Built edge gateway for IoT data compression")
        print("✓ Demonstrated compression on synthetic data")
        print("\nNext Steps:")
        print("- Test on CAPTURE 24 dataset")
        print("- Implement hybrid compression methods")
        print("- Evaluate on real IoT hardware")
        print("- Measure energy consumption")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

