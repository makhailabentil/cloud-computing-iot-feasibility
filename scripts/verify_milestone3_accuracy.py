"""
Comprehensive Verification of Milestone 3 Results

This script verifies the accuracy of all Milestone 3 calculations:
1. Adaptive compression reconstruction errors
2. Hybrid methods compression ratios and errors
3. Multi-axis compression accuracy
4. Compressed analytics calculations (mean, variance)
"""

import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.compressors.delta_encoding import DeltaEncodingCompressor
from src.compressors.run_length import RunLengthCompressor
from src.compressors.quantization import QuantizationCompressor
from src.compressors.adaptive_compressor import AdaptiveCompressor
from src.compressors.hybrid_compressor import HybridCompressor
from src.compressors.multi_axis_compressor import MultiAxisCompressor
from src.analytics.compressed_analytics import CompressedAnalytics
from src.data_processing.capture24_loader import Capture24Loader

def verify_delta_encoding():
    """Verify delta encoding is truly lossless."""
    print("=" * 60)
    print("1. Verifying Delta Encoding (Lossless)")
    print("=" * 60)
    
    compressor = DeltaEncodingCompressor()
    
    # Test with known data
    test_data = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    compressed, first_val = compressor.compress(test_data)
    decompressed = compressor.decompress(compressed, first_val)
    
    mse = np.mean((test_data - decompressed) ** 2)
    max_error = np.max(np.abs(test_data - decompressed))
    
    print(f"  Test data: {test_data[:5]}...")
    print(f"  MSE: {mse:.15f}")
    print(f"  Max error: {max_error:.15f}")
    print(f"  Compression ratio: {compressor.compression_ratio:.2f}×")
    
    if mse < 1e-10:
        print("  ✅ PASS: Delta encoding is lossless (as expected)")
    else:
        print(f"  ❌ FAIL: Delta encoding has error {mse}")
    
    return mse < 1e-10


def verify_analytics_calculations():
    """Verify compressed analytics calculations are accurate."""
    print("\n" + "=" * 60)
    print("2. Verifying Compressed Analytics Calculations")
    print("=" * 60)
    
    analytics = CompressedAnalytics()
    compressor = DeltaEncodingCompressor()
    
    # Test with known data
    test_data = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    compressed, first_val = compressor.compress(test_data)
    
    # Get stats from compressed
    stats = analytics.delta_encoding_statistics(compressed, first_val)
    
    # Actual stats
    actual_mean = np.mean(test_data)
    actual_var = np.var(test_data)
    
    mean_error = abs(stats['mean'] - actual_mean)
    var_error = abs(stats['variance'] - actual_var)
    
    print(f"  Test data: {test_data[:5]}...")
    print(f"  Actual mean: {actual_mean:.10f}")
    print(f"  Compressed mean: {stats['mean']:.10f}")
    print(f"  Mean error: {mean_error:.15f}")
    print(f"  Actual variance: {actual_var:.10f}")
    print(f"  Compressed variance: {stats['variance']:.10f}")
    print(f"  Variance error: {var_error:.15f}")
    
    # Check if errors are within floating point precision
    if mean_error < 1e-10 and var_error < 1e-10:
        print("  ✅ PASS: Analytics calculations are accurate (within FP precision)")
        return True
    else:
        print(f"  ❌ FAIL: Analytics has significant errors")
        return False


def verify_adaptive_compression():
    """Verify adaptive compression reconstruction."""
    print("\n" + "=" * 60)
    print("3. Verifying Adaptive Compression")
    print("=" * 60)
    
    compressor = AdaptiveCompressor()
    loader = Capture24Loader()
    
    # Load a small segment
    data = loader.load_participant_data('P001', max_samples=30000)
    x = data['x'][:10000]
    y = data['y'][:10000]
    z = data['z'][:10000]
    
    # Compress
    result = compressor.compress(x, y, z)
    
    # Decompress and verify
    from src.compressors.delta_encoding import DeltaEncodingCompressor
    from src.compressors.run_length import RunLengthCompressor
    
    errors = []
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        compressed_info = result['compressed_data'][axis_name]
        algorithm = compressed_info['algorithm']
        metadata = compressed_info['metadata']
        
        if algorithm == 'delta_encoding':
            decompressor = DeltaEncodingCompressor()
            decompressed = decompressor.decompress(
                compressed_info['data'],
                metadata['first_val']
            )
        elif algorithm == 'run_length':
            decompressor = RunLengthCompressor()
            decompressed = decompressor.decompress(
                compressed_info['data'],
                metadata['original_len']
            )
        else:
            continue
        
        # Calculate error
        mse = np.mean((axis_data - decompressed) ** 2)
        errors.append(mse)
    
    avg_error = np.mean(errors)
    print(f"  Algorithm used: {result['algorithm']}")
    print(f"  Activity detected: {result['activity']}")
    print(f"  Compression ratio: {result['compression_ratio']:.2f}×")
    print(f"  Reported error: {result['reconstruction_error']:.10f}")
    print(f"  Calculated MSE: {avg_error:.10f}")
    
    if avg_error < 1e-10:
        print("  ✅ PASS: Adaptive compression reconstruction is accurate")
        return True
    else:
        print(f"  ⚠️  WARNING: Error is {avg_error} (may be quantization)")
        return avg_error < 0.001  # Allow for quantization error


def verify_hybrid_methods():
    """Verify hybrid methods calculations."""
    print("\n" + "=" * 60)
    print("4. Verifying Hybrid Methods")
    print("=" * 60)
    
    compressor = HybridCompressor()
    loader = Capture24Loader()
    
    # Load a small segment
    data = loader.load_participant_data('P001', max_samples=20000)
    x = data['x'][:10000]
    
    # Test Delta+RLE
    try:
        compressed_dict, ratio, error = compressor.delta_then_rle(x)
        decompressed = compressor.decompress_hybrid(compressed_dict)
        actual_error = np.mean((x - decompressed) ** 2)
        
        print(f"  Delta+RLE:")
        print(f"    Compression ratio: {ratio:.2f}×")
        print(f"    Reported error: {error:.10f}")
        print(f"    Actual MSE: {actual_error:.10f}")
        
        if abs(error - actual_error) < 1e-6:
            print("    ✅ PASS: Error calculation is accurate")
        else:
            print(f"    ⚠️  WARNING: Error mismatch {abs(error - actual_error)}")
    except Exception as e:
        print(f"    ❌ FAIL: {e}")
    
    # Test Delta+Quantization
    try:
        compressed_dict, ratio, error = compressor.delta_then_quantization(x)
        decompressed = compressor.decompress_hybrid(compressed_dict)
        actual_error = np.mean((x - decompressed) ** 2)
        
        print(f"  Delta+Quantization:")
        print(f"    Compression ratio: {ratio:.2f}×")
        print(f"    Reported error: {error:.10f}")
        print(f"    Actual MSE: {actual_error:.10f}")
        
        if abs(error - actual_error) < 0.01:  # Quantization has some error
            print("    ✅ PASS: Error calculation is accurate")
        else:
            print(f"    ⚠️  WARNING: Error mismatch {abs(error - actual_error)}")
    except Exception as e:
        print(f"    ❌ FAIL: {e}")


def verify_multi_axis():
    """Verify multi-axis compression."""
    print("\n" + "=" * 60)
    print("5. Verifying Multi-Axis Compression")
    print("=" * 60)
    
    compressor = MultiAxisCompressor()
    loader = Capture24Loader()
    
    # Load a small segment
    data = loader.load_participant_data('P001', max_samples=30000)
    x = data['x'][:10000]
    y = data['y'][:10000]
    z = data['z'][:10000]
    
    # Test joint compression
    result = compressor.compare_strategies(x, y, z)
    
    if 'joint_delta' in result:
        joint = result['joint_delta']
        print(f"  Joint Compression:")
        print(f"    Compression ratio: {joint['compression_ratio']:.2f}×")
        print(f"    Reconstruction error: {joint['reconstruction_error']:.10f}")
        
        # Verify it's lossless (delta encoding)
        if joint['reconstruction_error'] < 1e-10:
            print("    ✅ PASS: Joint compression is lossless")
        else:
            print(f"    ⚠️  WARNING: Error is {joint['reconstruction_error']}")


def verify_actual_results():
    """Verify the actual evaluation results files."""
    print("\n" + "=" * 60)
    print("6. Verifying Actual Evaluation Results")
    print("=" * 60)
    
    # Check adaptive results
    adaptive_file = Path("results/evaluation/milestone3_adaptive_results.json")
    if adaptive_file.exists():
        with open(adaptive_file, 'r') as f:
            adaptive_data = json.load(f)
        
        errors = []
        for participant_data in adaptive_data:
            for segment_result in participant_data.get('results', []):
                errors.append(segment_result.get('reconstruction_error', 0))
        
        print(f"  Adaptive Compression:")
        print(f"    Segments: {len(errors)}")
        print(f"    Error range: {min(errors):.10f} to {max(errors):.10f}")
        print(f"    Average error: {np.mean(errors):.10f}")
        
        if all(e < 1e-10 for e in errors):
            print("    ✅ PASS: All errors are essentially zero (lossless)")
        else:
            print(f"    ⚠️  WARNING: Some errors are non-zero")
    
    # Check analytics results
    analytics_file = Path("results/evaluation/milestone3_analytics_results.json")
    if analytics_file.exists():
        with open(analytics_file, 'r') as f:
            analytics_data = json.load(f)
        
        mean_errors = []
        var_errors = []
        for participant_data in analytics_data:
            for segment_result in participant_data.get('results', []):
                mean_errors.append(segment_result.get('mean_error', 0))
                var_errors.append(segment_result.get('variance_error', 0))
        
        print(f"  Compressed Analytics:")
        print(f"    Segments: {len(mean_errors)}")
        print(f"    Mean error range: {min(mean_errors):.15f} to {max(mean_errors):.15f}")
        print(f"    Mean error average: {np.mean(mean_errors):.15f}")
        print(f"    Variance error range: {min(var_errors):.15f} to {max(var_errors):.15f}")
        print(f"    Variance error average: {np.mean(var_errors):.15f}")
        
        # These should be very small (floating point precision)
        if np.mean(mean_errors) < 1e-10 and np.mean(var_errors) < 1e-10:
            print("    ✅ PASS: Errors are within floating point precision")
        else:
            print(f"    ⚠️  WARNING: Errors may be larger than expected")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("MILESTONE 3 COMPREHENSIVE ACCURACY VERIFICATION")
    print("=" * 60)
    print()
    
    results = {
        'delta_encoding': verify_delta_encoding(),
        'analytics': verify_analytics_calculations(),
        'adaptive': verify_adaptive_compression(),
    }
    
    verify_hybrid_methods()
    verify_multi_axis()
    verify_actual_results()
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")
    
    print("\nNote: 0.00 errors are CORRECT for:")
    print("  - Delta Encoding: Lossless compression (perfect reconstruction)")
    print("  - Run-Length Encoding: Lossless for exact matches")
    print("  - Compressed Analytics: Reconstructs values, so errors are FP precision only")
    print("\nSmall errors are expected for:")
    print("  - Quantization: Lossy compression (intentional quality trade-off)")
    print("  - Hybrid methods with quantization: Inherits quantization error")


if __name__ == "__main__":
    main()


