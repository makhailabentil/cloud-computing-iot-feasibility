#!/usr/bin/env python3
"""
Generate sample ESP32 streaming data for testing visualizations.

This script creates a realistic CSV file that mimics ESP32 streaming data,
allowing you to test visualization tools without the actual hardware.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

def generate_sample_data(num_segments=50, participant_id="P001"):
    """Generate sample ESP32 streaming data."""
    
    algorithms = ["Delta", "RLE", "Quant"]
    axes = ["x", "y", "z"]
    
    # Realistic ranges based on actual ESP32 performance
    # Delta: ~400-1500 bytes per axis, ~10-30ms upload time
    # RLE: ~50-800 bytes per axis (highly variable), ~15-40ms upload time
    # Quant: ~300-1000 bytes per axis, ~20-50ms upload time
    
    rows = []
    base_time = datetime.now() - timedelta(minutes=num_segments * 0.5)
    
    for seg_idx in range(num_segments):
        # Simulate time progression
        timestamp = (base_time + timedelta(seconds=seg_idx * 30)).strftime("%H:%M:%S")
        
        for algorithm in algorithms:
            # Algorithm-specific characteristics
            if algorithm == "Delta":
                base_bytes = random.randint(400, 1500)
                base_time_ms = random.uniform(10, 30)
            elif algorithm == "RLE":
                # RLE is highly variable - sometimes very small, sometimes larger
                if random.random() < 0.3:  # 30% chance of excellent compression (rest period)
                    base_bytes = random.randint(50, 200)
                else:  # 70% chance of moderate compression (active period)
                    base_bytes = random.randint(400, 800)
                base_time_ms = random.uniform(15, 40)
            else:  # Quantization
                base_bytes = random.randint(300, 1000)
                base_time_ms = random.uniform(20, 50)
            
            # Add some variation per axis
            for axis in axes:
                # Each axis has slightly different size
                axis_multiplier = random.uniform(0.9, 1.1)
                upload_bytes = int(base_bytes * axis_multiplier)
                
                # Upload time varies slightly
                time_variation = random.uniform(0.8, 1.2)
                upload_time_ms = round(base_time_ms * time_variation, 3)
                
                rows.append({
                    "timestamp": timestamp,
                    "participant": participant_id,
                    "segment_index": seg_idx,
                    "axis": axis,
                    "algorithm": algorithm,
                    "upload_bytes": upload_bytes,
                    "upload_time_ms": upload_time_ms
                })
    
    return pd.DataFrame(rows)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample ESP32 streaming data')
    parser.add_argument('--segments', type=int, default=50,
                       help='Number of segments to generate (default: 50)')
    parser.add_argument('--participant', type=str, default='P001',
                       help='Participant ID (default: P001)')
    parser.add_argument('--output', type=str, default='stream_compression_results.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    print(f"Generating sample ESP32 data...")
    print(f"  Segments: {args.segments}")
    print(f"  Participant: {args.participant}")
    print(f"  Output: {args.output}")
    
    df = generate_sample_data(num_segments=args.segments, participant_id=args.participant)
    
    # Save to CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(df)} rows of sample data")
    print(f"   Saved to: {output_path.absolute()}")
    print(f"\nYou can now test visualizations with:")
    print(f"   python scripts/visualize_esp32.py")
    print(f"   python scripts/visualize_esp32.py --all")
    print(f"   python scripts/visualize_esp32.py --monitor")

if __name__ == '__main__':
    main()

