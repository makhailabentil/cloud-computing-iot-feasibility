"""
Find Rest/Sleep Segments in CAPTURE-24 Dataset

This script searches through the CAPTURE-24 dataset to find segments
that contain rest or sleep periods, which can be used to test
activity-aware adaptive compression with RLE.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.capture24_loader import Capture24Loader
from src.activity.activity_detector import ActivityDetector

def find_rest_sleep_segments(participant_id: str = 'P001', 
                             max_samples: int = 1000000,
                             window_size: int = 10000,
                             max_segments: int = 100):
    """
    Find segments with rest or sleep activity.
    
    Returns:
        List of (segment_index, activity_type, variance, start_sample, end_sample)
    """
    loader = Capture24Loader()
    detector = ActivityDetector()
    
    print(f"Loading data for {participant_id}...")
    data = loader.load_participant_data(participant_id, max_samples=max_samples)
    x, y, z = data['x'], data['y'], data['z']
    
    print(f"Analyzing {len(x)} samples in windows of {window_size}...")
    
    rest_sleep_segments = []
    all_activities = []
    
    for i in range(0, len(x) - window_size, window_size):
        if len(rest_sleep_segments) >= max_segments:
            break
            
        seg_x = x[i:i+window_size]
        seg_y = y[i:i+window_size]
        seg_z = z[i:i+window_size]
        
        # Detect activity
        activity = detector.detect_activity(seg_x, seg_y, seg_z)
        all_activities.append(activity.value)
        
        # Calculate variance
        var_x = np.var(seg_x)
        var_y = np.var(seg_y)
        var_z = np.var(seg_z)
        avg_variance = (var_x + var_y + var_z) / 3.0
        
        # Store rest/sleep segments
        if activity.value in ['rest', 'sleep']:
            rest_sleep_segments.append({
                'segment_index': i // window_size,
                'activity': activity.value,
                'variance': avg_variance,
                'start_sample': i,
                'end_sample': i + window_size,
                'variance_x': var_x,
                'variance_y': var_y,
                'variance_z': var_z
            })
    
    return rest_sleep_segments, all_activities


def main():
    """Find rest/sleep segments across multiple participants."""
    participants = ['P001', 'P002', 'P003', 'P004', 'P005']
    
    print("=" * 60)
    print("Finding Rest/Sleep Segments in CAPTURE-24 Dataset")
    print("=" * 60)
    print()
    
    all_rest_sleep = []
    all_activities = []
    
    for participant_id in participants:
        try:
            rest_sleep, activities = find_rest_sleep_segments(
                participant_id=participant_id,
                max_samples=500000,  # Check first 500k samples
                window_size=10000,
                max_segments=20
            )
            
            all_rest_sleep.extend([(participant_id, seg) for seg in rest_sleep])
            all_activities.extend([(participant_id, act) for act in activities])
            
            print(f"\n{participant_id}:")
            print(f"  Total segments analyzed: {len(activities)}")
            print(f"  Rest/Sleep segments found: {len(rest_sleep)}")
            if rest_sleep:
                print(f"  First rest/sleep segment: index {rest_sleep[0]['segment_index']}, "
                      f"variance={rest_sleep[0]['variance']:.6f}")
        
        except Exception as e:
            print(f"  Error processing {participant_id}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total rest/sleep segments found: {len(all_rest_sleep)}")
    
    if all_rest_sleep:
        print("\nRest/Sleep Segments Found:")
        for participant_id, seg in all_rest_sleep[:10]:  # Show first 10
            print(f"  {participant_id} - Segment {seg['segment_index']}: "
                  f"{seg['activity']} (variance={seg['variance']:.6f})")
        
        # Save to file
        output_file = Path("results/rest_sleep_segments.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w') as f:
            json.dump(all_rest_sleep, f, indent=2, default=str)
        
        print(f"\nSaved {len(all_rest_sleep)} segments to {output_file}")
        print("\nYou can use these segments to test adaptive compression with RLE!")
    else:
        print("\n⚠️  No rest/sleep segments found in the analyzed data.")
        print("   The dataset may contain mostly active periods, or thresholds need adjustment.")
        print("   Try:")
        print("   1. Analyzing more participants")
        print("   2. Checking different time periods (nighttime for sleep)")
        print("   3. Adjusting activity detection thresholds")


if __name__ == "__main__":
    main()


