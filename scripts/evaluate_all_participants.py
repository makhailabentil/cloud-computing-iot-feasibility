#!/usr/bin/env python3
"""
Evaluate all CAPTURE-24 participants.

This script generates the full participant list and runs evaluation on all 151 participants.
It can also resume from a checkpoint if interrupted.
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.capture24_loader import Capture24Loader
from evaluation.systematic_evaluation import SystematicEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_participant_list(start=1, end=151):
    """Generate participant IDs from P001 to P151."""
    return [f"P{i:03d}" for i in range(start, end + 1)]


def check_existing_results(output_dir: Path, participant_id: str) -> bool:
    """Check if results already exist for a participant."""
    results_file = output_dir / f"milestone2_evaluation_{participant_id}.json"
    return results_file.exists()


def main():
    """Main evaluation function for all participants."""
    parser = argparse.ArgumentParser(
        description="Evaluate compression algorithms on all CAPTURE-24 participants"
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Starting participant number (default: 1)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=151,
        help='Ending participant number (default: 151)'
    )
    parser.add_argument(
        '--max-segments',
        type=int,
        default=10,
        help='Maximum number of segments to evaluate per participant'
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
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip participants that already have results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of participants to process before saving intermediate results'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Full CAPTURE-24 Evaluation: All Participants")
    logger.info("=" * 60)
    
    # Generate participant list
    participant_ids = generate_participant_list(args.start, args.end)
    logger.info(f"Evaluating {len(participant_ids)} participants: {participant_ids[0]} to {participant_ids[-1]}")
    
    # Initialize components
    loader = Capture24Loader()
    evaluator = SystematicEvaluator()
    axes = [a.strip() for a in args.axes.split(',')]
    
    # Output directory for checking existing results
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track progress
    total_participants = len(participant_ids)
    processed = 0
    skipped = 0
    errors = 0
    
    # Process each participant
    for idx, participant_id in enumerate(participant_ids, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing participant {idx}/{total_participants}: {participant_id}")
        logger.info(f"{'='*60}")
        
        # Check if we should skip
        if args.skip_existing and check_existing_results(output_dir, participant_id):
            logger.info(f"Skipping {participant_id} (results already exist)")
            skipped += 1
            continue
        
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
                
                logger.info(f"Evaluating {axis}-axis data for {participant_id}...")
                
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
                
                logger.info(f"Completed evaluation of {len(segments)} segments for {axis}-axis")
            
            # Save intermediate results every batch_size participants
            if idx % args.batch_size == 0:
                logger.info(f"\nSaving intermediate results (processed {idx}/{total_participants})...")
                evaluator.save_results(f"milestone2_evaluation_{participant_id}_intermediate.json")
                logger.info("Intermediate results saved")
            
            processed += 1
            
        except FileNotFoundError as e:
            logger.warning(f"Data not found for {participant_id}: {e}")
            errors += 1
            continue
        except Exception as e:
            logger.error(f"Error evaluating {participant_id}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
            continue
    
    # Generate final report
    logger.info("\n" + "=" * 60)
    logger.info("Generating final evaluation report...")
    evaluator.save_results(f"milestone2_evaluation_ALL_PARTICIPANTS.json")
    evaluator.generate_report(f"milestone2_report_ALL_PARTICIPANTS.md")
    
    # Print overall summary
    summary = evaluator.get_summary()
    logger.info("\n" + "=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)
    logger.info(f"Total participants processed: {processed}")
    logger.info(f"Total participants skipped: {skipped}")
    logger.info(f"Total participants with errors: {errors}")
    logger.info(f"Total evaluations: {summary.get('delta_encoding', {}).get('n_evaluations', 0)}")
    
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

