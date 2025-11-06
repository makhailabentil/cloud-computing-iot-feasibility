#!/usr/bin/env python3
"""
Milestone 2: Systematic Evaluation of Compression Algorithms on CAPTURE-24

This script implements systematic evaluation of compression algorithms (delta encoding,
run-length encoding, and quantization) on CAPTURE-24 accelerometer data segments.

Usage:
    python scripts/evaluate_capture24.py [--participants P001,P002] [--max-segments 10] [--synthetic]
"""

import argparse
import sys
from pathlib import Path
import logging
import numpy as np

# Add src to path (scripts/ is one level deeper than root)
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.capture24_loader import Capture24Loader
from evaluation.systematic_evaluation import SystematicEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function for Milestone 2."""
    parser = argparse.ArgumentParser(
        description="Evaluate compression algorithms on CAPTURE-24 data"
    )
    parser.add_argument(
        '--participants',
        type=str,
        default='P001',
        help='Comma-separated list of participant IDs (e.g., P001,P002)'
    )
    parser.add_argument(
        '--max-segments',
        type=int,
        default=10,
        help='Maximum number of segments to evaluate per participant'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic CAPTURE-24 data instead of real data'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=10000,
        help='Size of data segments in samples (default: 10000 = 100 seconds at 100Hz)'
    )
    parser.add_argument(
        '--axes',
        type=str,
        default='x,y,z',
        help='Comma-separated list of axes to evaluate (default: x,y,z)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Milestone 2: Systematic Evaluation of Compression Algorithms")
    logger.info("=" * 60)
    
    # Initialize components
    loader = Capture24Loader()
    evaluator = SystematicEvaluator()
    
    # Parse participant IDs
    participant_ids = [p.strip() for p in args.participants.split(',')]
    axes = [a.strip() for a in args.axes.split(',')]
    
    # Generate synthetic data if requested (before loading)
    if args.synthetic:
        logger.info("\nUsing synthetic CAPTURE-24 data for evaluation")
        for participant_id in participant_ids:
            logger.info(f"\nGenerating synthetic data for {participant_id}...")
            # Generate synthetic data - this will create P001.csv etc.
            synthetic_data = loader.create_synthetic_capture24_data(
                participant_id=participant_id,
                n_samples=args.window_size * args.max_segments,
                sampling_rate=100.0
            )
    
    # Evaluate each participant
    for participant_id in participant_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating participant: {participant_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Load participant data
            logger.info(f"Loading data for {participant_id}...")
            data = loader.load_participant_data(
                participant_id=participant_id,
                axes=axes,
                max_samples=args.window_size * args.max_segments
            )
            
            # Evaluate each axis
            for axis in axes:
                if axis not in data:
                    logger.warning(f"Axis {axis} not found in data for {participant_id}, skipping")
                    continue
                
                logger.info(f"\nEvaluating {axis}-axis data for {participant_id}...")
                
                # Segment data
                segments = loader.segment_data(
                    data[axis],
                    window_size=args.window_size,
                    overlap=0
                )
                
                # Limit number of segments
                segments = segments[:args.max_segments]
                logger.info(f"Processing {len(segments)} segments of {args.window_size} samples each")
                
                # Evaluate segments
                segment_results = evaluator.evaluate_segments(
                    segments,
                    axis_name=f"{axis}-axis",
                    participant_id=participant_id
                )
                
                logger.info(f"Completed evaluation of {len(segments)} segments")
                
                # Print summary for this axis
                if len(segment_results) > 0:
                    print(f"\n{axis}-axis Results Summary:")
                    print("-" * 60)
                    for algorithm in segment_results['algorithm'].unique():
                        alg_data = segment_results[segment_results['algorithm'] == algorithm]
                        print(f"{algorithm}:")
                        print(f"  Mean Compression Ratio: {alg_data['compression_ratio'].mean():.2f}x")
                        print(f"  Mean Reconstruction Error: {alg_data['reconstruction_error'].mean():.6f}")
                        print(f"  Mean Compression Time: {alg_data['compression_time'].mean():.4f}s")
                        print(f"  Mean Memory Usage: {alg_data['memory_usage_mb'].mean():.2f} MB")
                
        except FileNotFoundError as e:
            logger.warning(f"Data not found for {participant_id}: {e}")
            logger.info("Try using --synthetic flag to generate test data")
            continue
        except Exception as e:
            logger.error(f"Error evaluating {participant_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate final report
    logger.info("\n" + "=" * 60)
    logger.info("Generating evaluation report...")
    evaluator.save_results(f"milestone2_evaluation_{participant_ids[0]}.json")
    evaluator.generate_report(f"milestone2_report_{participant_ids[0]}.md")
    
    # Print overall summary
    summary = evaluator.get_summary()
    logger.info("\n" + "=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)
    
    for algorithm, stats in summary.items():
        logger.info(f"\n{algorithm.replace('_', ' ').title()}:")
        logger.info(f"  Mean Compression Ratio: {stats['mean_compression_ratio']:.2f}x "
                   f"(±{stats['std_compression_ratio']:.2f})")
        logger.info(f"  Mean Reconstruction Error: {stats['mean_reconstruction_error']:.6f}")
        logger.info(f"  Mean Compression Time: {stats['mean_compression_time']:.4f}s")
        logger.info(f"  Mean Memory Usage: {stats['mean_memory_usage_mb']:.2f} MB")
        logger.info(f"  Total Evaluations: {stats['n_evaluations']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {evaluator.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

