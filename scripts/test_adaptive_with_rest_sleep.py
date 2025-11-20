"""
Test Adaptive Compression on Rest/Sleep Segments

This script tests the adaptive compressor on segments that contain
rest or sleep periods to verify that RLE is being used correctly.
"""

import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.capture24_loader import Capture24Loader
from src.compressors.adaptive_compressor import AdaptiveCompressor

def test_adaptive_on_rest_sleep():
    """Test adaptive compression on known rest/sleep segments."""
    
    # Load the rest/sleep segments we found
    segments_file = Path("results/rest_sleep_segments.json")
    if not segments_file.exists():
        print("⚠️  No rest/sleep segments file found. Run find_rest_sleep_segments.py first.")
        return
    
    with open(segments_file, 'r') as f:
        rest_sleep_segments = json.load(f)
    
    print("=" * 60)
    print("Testing Adaptive Compression on Rest/Sleep Segments")
    print("=" * 60)
    print()
    
    loader = Capture24Loader()
    compressor = AdaptiveCompressor()
    
    # Test on first 5 rest/sleep segments
    test_segments = rest_sleep_segments[:5]
    
    results = []
    for participant_id, seg_info in test_segments:
        print(f"\nTesting {participant_id} - Segment {seg_info['segment_index']} ({seg_info['activity']})")
        
        # Load data
        data = loader.load_participant_data(participant_id, max_samples=seg_info['end_sample'] + 1000)
        x = data['x'][seg_info['start_sample']:seg_info['end_sample']]
        y = data['y'][seg_info['start_sample']:seg_info['end_sample']]
        z = data['z'][seg_info['start_sample']:seg_info['end_sample']]
        
        # Compress with adaptive compressor
        result = compressor.compress(x, y, z)
        
        print(f"  Detected activity: {result['activity']}")
        print(f"  Selected algorithm: {result['algorithm']}")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}×")
        print(f"  Reconstruction error: {result['reconstruction_error']:.6f}")
        
        results.append({
            'participant': participant_id,
            'segment': seg_info['segment_index'],
            'expected_activity': seg_info['activity'],
            'detected_activity': result['activity'],
            'algorithm': result['algorithm'],
            'compression_ratio': result['compression_ratio']
        })
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    rle_used = sum(1 for r in results if r['algorithm'] == 'run_length')
    print(f"RLE used: {rle_used} / {len(results)} segments")
    print(f"Delta used: {len(results) - rle_used} / {len(results)} segments")
    
    if rle_used > 0:
        print("\n✅ SUCCESS! Adaptive compressor is using RLE for rest/sleep periods!")
        avg_ratio = np.mean([r['compression_ratio'] for r in results if r['algorithm'] == 'run_length'])
        print(f"   Average RLE compression ratio: {avg_ratio:.2f}×")
    else:
        print("\n⚠️  RLE not being used. Activity detection may need further tuning.")
    
    # Show algorithm usage
    print("\nAlgorithm Usage:")
    for r in results:
        print(f"  {r['participant']} Seg{r['segment']}: {r['algorithm']} ({r['compression_ratio']:.2f}×)")


if __name__ == "__main__":
    test_adaptive_on_rest_sleep()


