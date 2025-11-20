"""
Milestone 3 Evaluation Script

Evaluates all new compression methods:
- Adaptive compression
- Hybrid compression
- Multi-axis compression
- On-the-fly analytics

For Milestone 3: Complete Evaluation
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.capture24_loader import Capture24Loader
from src.compressors.adaptive_compressor import AdaptiveCompressor
from src.compressors.hybrid_compressor import HybridCompressor
from src.compressors.multi_axis_compressor import MultiAxisCompressor
from src.analytics.compressed_analytics import CompressedAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def evaluate_adaptive_compression(loader: Capture24Loader, participant_id: str,
                                max_segments: int = 10) -> Dict:
    """Evaluate adaptive compression on CAPTURE-24 data."""
    logger.info(f"Evaluating adaptive compression for {participant_id}")
    
    # Load data
    data = loader.load_participant_data(participant_id, max_samples=100000)
    x = data['x']
    y = data['y']
    z = data['z']
    
    # Segment data
    window_size = 10000
    segments_x = loader.segment_data(x, window_size=window_size)
    segments_y = loader.segment_data(y, window_size=window_size)
    segments_z = loader.segment_data(z, window_size=window_size)
    
    # Initialize adaptive compressor
    compressor = AdaptiveCompressor()
    
    results = []
    for i in range(min(max_segments, len(segments_x))):
        result = compressor.compress(segments_x[i], segments_y[i], segments_z[i])
        results.append(result)
    
    # Get statistics
    stats = compressor.get_statistics()
    
    return {
        'participant_id': participant_id,
        'num_segments': len(results),
        'results': results,
        'statistics': stats,
        'avg_compression_ratio': np.mean([r['compression_ratio'] for r in results]),
        'avg_reconstruction_error': np.mean([r['reconstruction_error'] for r in results])
    }


def evaluate_hybrid_compression(loader: Capture24Loader, participant_id: str,
                               max_segments: int = 10) -> Dict:
    """Evaluate hybrid compression methods."""
    logger.info(f"Evaluating hybrid compression for {participant_id}")
    
    # Load data
    data = loader.load_participant_data(participant_id, max_samples=100000)
    x = data['x']
    
    # Segment data
    window_size = 10000
    segments = loader.segment_data(x, window_size=window_size)
    
    # Initialize hybrid compressor
    compressor = HybridCompressor()
    
    results = []
    for i in range(min(max_segments, len(segments))):
        comparison = compressor.compare_hybrid_methods(segments[i])
        results.append(comparison)
    
    # Aggregate results
    methods = ['delta_quantization', 'delta_rle', 'quantization_delta']
    aggregated = {}
    for method in methods:
        ratios = [r[method]['compression_ratio'] for r in results if method in r]
        errors = [r[method]['reconstruction_error'] for r in results if method in r]
        aggregated[method] = {
            'avg_compression_ratio': np.mean(ratios) if ratios else 0.0,
            'avg_reconstruction_error': np.mean(errors) if errors else float('inf')
        }
    
    return {
        'participant_id': participant_id,
        'num_segments': len(results),
        'results': results,
        'aggregated': aggregated
    }


def evaluate_multi_axis_compression(loader: Capture24Loader, participant_id: str,
                                   max_segments: int = 10) -> Dict:
    """Evaluate multi-axis compression strategies."""
    logger.info(f"Evaluating multi-axis compression for {participant_id}")
    
    # Load data
    data = loader.load_participant_data(participant_id, max_samples=100000)
    x = data['x']
    y = data['y']
    z = data['z']
    
    # Segment data
    window_size = 10000
    segments_x = loader.segment_data(x, window_size=window_size)
    segments_y = loader.segment_data(y, window_size=window_size)
    segments_z = loader.segment_data(z, window_size=window_size)
    
    # Initialize multi-axis compressor
    compressor = MultiAxisCompressor()
    
    results = []
    for i in range(min(max_segments, len(segments_x))):
        comparison = compressor.compare_strategies(
            segments_x[i], segments_y[i], segments_z[i]
        )
        results.append(comparison)
    
    # Aggregate results
    strategies = ['joint_delta', 'vector_delta', 'magnitude_delta', 'axis_specific']
    aggregated = {}
    for strategy in strategies:
        ratios = [r[strategy]['compression_ratio'] for r in results if strategy in r]
        errors = [r[strategy]['reconstruction_error'] for r in results if strategy in r]
        aggregated[strategy] = {
            'avg_compression_ratio': np.mean(ratios) if ratios else 0.0,
            'avg_reconstruction_error': np.mean(errors) if errors else float('inf')
        }
    
    return {
        'participant_id': participant_id,
        'num_segments': len(results),
        'results': results,
        'aggregated': aggregated
    }


def evaluate_analytics(loader: Capture24Loader, participant_id: str,
                      max_segments: int = 10) -> Dict:
    """Evaluate on-the-fly analytics."""
    logger.info(f"Evaluating compressed analytics for {participant_id}")
    
    # Load data
    data = loader.load_participant_data(participant_id, max_samples=100000)
    x = data['x']
    
    # Segment data
    window_size = 10000
    segments = loader.segment_data(x, window_size=window_size)
    
    # Initialize analytics and compressors
    analytics = CompressedAnalytics()
    from src.compressors.delta_encoding import DeltaEncodingCompressor
    delta_compressor = DeltaEncodingCompressor()
    
    results = []
    for i in range(min(max_segments, len(segments))):
        # Compress
        compressed, first_val = delta_compressor.compress(segments[i])
        
        # Get statistics from compressed data
        stats = analytics.delta_encoding_statistics(compressed, first_val)
        
        # Detect anomalies
        anomalies = analytics.detect_anomaly_delta(compressed, first_val)
        
        # Query range
        queried = analytics.query_range_delta(compressed, first_val, 0, 1000)
        
        # Compare with actual statistics
        actual_mean = np.mean(segments[i])
        actual_var = np.var(segments[i])
        
        results.append({
            'compressed_stats': stats,
            'actual_mean': actual_mean,
            'actual_variance': actual_var,
            'mean_error': abs(stats['mean'] - actual_mean),
            'variance_error': abs(stats['variance'] - actual_var),
            'num_anomalies': len(anomalies),
            'query_range_length': len(queried)
        })
    
    return {
        'participant_id': participant_id,
        'num_segments': len(results),
        'results': results,
        'avg_mean_error': np.mean([r['mean_error'] for r in results]),
        'avg_variance_error': np.mean([r['variance_error'] for r in results])
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Milestone 3 compression methods')
    parser.add_argument('--participants', type=str, default='P001',
                      help='Comma-separated list of participant IDs')
    parser.add_argument('--max-segments', type=int, default=10,
                      help='Maximum number of segments to evaluate per participant')
    parser.add_argument('--methods', type=str, default='all',
                      help='Comma-separated list of methods: adaptive,hybrid,multi_axis,analytics,all')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse participants
    participant_ids = [p.strip() for p in args.participants.split(',')]
    
    # Parse methods
    if args.methods == 'all':
        methods = ['adaptive', 'hybrid', 'multi_axis', 'analytics']
    else:
        methods = [m.strip() for m in args.methods.split(',')]
    
    # Initialize loader
    loader = Capture24Loader()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each method
    all_results = {}
    
    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {method} compression")
        logger.info(f"{'='*60}\n")
        
        method_results = []
        for participant_id in participant_ids:
            try:
                if method == 'adaptive':
                    result = evaluate_adaptive_compression(loader, participant_id, args.max_segments)
                elif method == 'hybrid':
                    result = evaluate_hybrid_compression(loader, participant_id, args.max_segments)
                elif method == 'multi_axis':
                    result = evaluate_multi_axis_compression(loader, participant_id, args.max_segments)
                elif method == 'analytics':
                    result = evaluate_analytics(loader, participant_id, args.max_segments)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                method_results.append(result)
                logger.info(f"Completed {participant_id}")
                
            except Exception as e:
                logger.error(f"Error evaluating {participant_id}: {e}")
                continue
        
        all_results[method] = method_results
        
        # Save results
        output_file = output_dir / f"milestone3_{method}_results.json"
        with open(output_file, 'w') as f:
            json.dump(method_results, f, indent=2, default=str)
        logger.info(f"Saved results to {output_file}")
    
    # Save combined results
    combined_file = output_dir / "milestone3_all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved combined results to {combined_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    
    for method, results in all_results.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Participants evaluated: {len(results)}")
        if results:
            logger.info(f"  Total segments: {sum(r.get('num_segments', 0) for r in results)}")


if __name__ == "__main__":
    main()


