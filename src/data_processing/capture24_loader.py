"""
CAPTURE-24 Dataset Loader

This module provides functionality to load and process CAPTURE-24 accelerometer data
for compression algorithm evaluation. CAPTURE-24 contains wrist-worn accelerometer
data from 151 participants at 100Hz sampling rate.

Reference: https://github.com/OxWearables/capture24
Dataset: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import logging
import os
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Capture24Loader:
    """
    Loader for CAPTURE-24 accelerometer dataset.
    
    CAPTURE-24 contains wrist-worn accelerometer data from 151 participants
    collected at 100Hz with 3 axes (x, y, z). Data is organized by participant ID.
    """
    
    def __init__(self, data_dir: str = "data/capture24"):
        """
        Initialize the CAPTURE-24 loader.
        
        Args:
            data_dir: Directory containing CAPTURE-24 data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.participants = []
        self.metadata = {}
        
    def list_participants(self) -> List[str]:
        """
        List available participant IDs in the dataset.
        
        Returns:
            List of participant IDs (e.g., ['P001', 'P002', ...])
        """
        if not self.participants:
            # Look for participant data files
            pattern = "P*.csv"
            files = list(self.data_dir.glob(pattern))
            
            # Extract participant IDs from filenames
            self.participants = sorted([f.stem for f in files])
            
            # If no CSV files, try to find other formats or use metadata
            if not self.participants:
                # Check for metadata file
                metadata_file = self.data_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.participants = metadata.get('participants', [])
                else:
                    # Generate list of expected participants (P001-P151)
                    self.participants = [f"P{i:03d}" for i in range(1, 152)]
        
        return self.participants
    
    def load_participant_data(self, participant_id: str, 
                            axes: List[str] = ['x', 'y', 'z'],
                            start_time: Optional[float] = None,
                            end_time: Optional[float] = None,
                            max_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Load accelerometer data for a specific participant.
        
        Args:
            participant_id: Participant ID (e.g., 'P001')
            axes: List of axes to load ('x', 'y', 'z', or 'magnitude')
            start_time: Start time in seconds (relative to recording start)
            end_time: End time in seconds (relative to recording start)
            max_samples: Maximum number of samples to load (for testing)
            
        Returns:
            Dictionary with axis names as keys and numpy arrays as values
        """
        # Try different file formats
        possible_files = [
            self.data_dir / f"{participant_id}.csv",
            self.data_dir / f"{participant_id}.csv.gz",
            self.data_dir / f"{participant_id}_accel.csv",
            self.data_dir / f"{participant_id}.parquet",
            self.data_dir / participant_id / "accel.csv",
        ]
        
        data_file = None
        for file_path in possible_files:
            if file_path.exists():
                data_file = file_path
                break
        
        if data_file is None:
            raise FileNotFoundError(
                f"Data file not found for participant {participant_id}. "
                f"Expected one of: {[str(f) for f in possible_files]}"
            )
        
        logger.info(f"Loading data for participant {participant_id} from {data_file}")
        
        # Load data based on file extension
        # handle compressed CSVs (.csv or .csv.gz)
        if data_file.suffix in ['.csv', '.gz', '.csv.gz']:
            df = pd.read_csv(data_file, compression='infer')
        elif data_file.suffix == '.parquet':
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        # Handle different column naming conventions
        # Standardize to lowercase
        df.columns = df.columns.str.lower()
        
        # Find timestamp and accelerometer columns
        timestamp_col = None
        for col in ['timestamp', 'time', 't']:
            if col in df.columns:
                timestamp_col = col
                break
        
        # Find accelerometer columns
        accel_cols = {}
        for axis in ['x', 'y', 'z']:
            for col in df.columns:
                if col.lower() in [f'{axis}', f'ax{axis}', f'accel_{axis}', f'accelerometer_{axis}']:
                    accel_cols[axis] = col
                    break
        
        if not accel_cols:
            raise ValueError(f"Could not find accelerometer columns in {data_file}")
        
        # Extract data
        data = {}
        for axis in axes:
            if axis == 'magnitude':
                # Calculate magnitude if not present
                if 'magnitude' in df.columns:
                    data['magnitude'] = df['magnitude'].values
                else:
                    # Calculate from x, y, z if available
                    x_data = df.get(accel_cols.get('x'), pd.Series([0]))
                    y_data = df.get(accel_cols.get('y'), pd.Series([0]))
                    z_data = df.get(accel_cols.get('z'), pd.Series([0]))
                    data['magnitude'] = np.sqrt(x_data**2 + y_data**2 + z_data**2).values
            else:
                if axis in accel_cols:
                    data[axis] = df[accel_cols[axis]].values
                else:
                    raise ValueError(f"Axis '{axis}' not found in data")
        
        # Apply time filtering if specified
        if timestamp_col and (start_time is not None or end_time is not None):
            if timestamp_col in df.columns:
                timestamps = df[timestamp_col].values
                if isinstance(timestamps[0], (pd.Timestamp, str)):
                    timestamps = pd.to_datetime(timestamps)
                    # Convert to seconds since start
                    timestamps = (timestamps - timestamps[0]).total_seconds()
                
                mask = np.ones(len(timestamps), dtype=bool)
                if start_time is not None:
                    mask &= timestamps >= start_time
                if end_time is not None:
                    mask &= timestamps <= end_time
                
                for axis in data:
                    data[axis] = data[axis][mask]
        
        # Limit samples if specified
        if max_samples is not None:
            for axis in data:
                data[axis] = data[axis][:max_samples]
        
        # Store metadata
        self.metadata[participant_id] = {
            'n_samples': len(data[axes[0]]) if axes else 0,
            'sampling_rate': 100.0,  # CAPTURE-24 is 100Hz
            'axes': axes,
            'data_file': str(data_file)
        }
        
        logger.info(f"Loaded {self.metadata[participant_id]['n_samples']} samples for {participant_id}")
        
        return data

    def segment_data(self, data: np.ndarray,
                     window_size: int = 10000,
                     overlap: int = 0,
                     min_gap_sec: float = 0,
                     sampling_rate: float = 100.0) -> List[np.ndarray]:
        """
        Segment time series data into non-overlapping windows separated
        by at least `min_gap_sec` seconds (default = 0.5 s).

        Args:
            data: Input time series data (1D numpy array)
            window_size: Number of samples per segment
            overlap: Overlap in samples between consecutive windows
            min_gap_sec: Minimum spacing between the *end* of one
                         segment and the *start* of the next, in seconds
            sampling_rate: Sampling rate of the signal (Hz)

        Returns:
            List of data segments
        """
        segments = []
        step = window_size - overlap
        gap = int(min_gap_sec * sampling_rate)  # convert seconds → samples
        cursor = 0

        while cursor + window_size <= len(data):
            segment = data[cursor:cursor + window_size]
            segments.append(segment)
            # move cursor forward: end of this segment + gap
            cursor += window_size + gap

        return segments

    def get_participant_info(self, participant_id: str) -> Dict:
        """
        Get metadata for a specific participant.
        
        Args:
            participant_id: Participant ID
            
        Returns:
            Dictionary with participant metadata
        """
        if participant_id not in self.metadata:
            # Try to load metadata
            metadata_file = self.data_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    all_metadata = json.load(f)
                    if participant_id in all_metadata.get('participants', {}):
                        return all_metadata['participants'][participant_id]
        
        return self.metadata.get(participant_id, {})
    
    def create_synthetic_capture24_data(self, participant_id: str = "P001",
                                       n_samples: int = 8640000,
                                       sampling_rate: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Create synthetic CAPTURE-24-like accelerometer data for testing.
        
        This generates realistic accelerometer patterns that mimic wrist-worn
        activity tracker data with realistic movement patterns.
        
        Args:
            participant_id: Participant ID for naming
            n_samples: Number of samples (8640000 = 24 hours at 100Hz)
            sampling_rate: Sampling rate in Hz (default 100Hz for CAPTURE-24)
            
        Returns:
            Dictionary with 'x', 'y', 'z' accelerometer data
        """
        logger.info(f"Generating synthetic CAPTURE-24 data for {participant_id}")
        
        t = np.arange(n_samples) / sampling_rate
        
        # Generate realistic accelerometer patterns
        # Base gravity + body movement + activity patterns + noise
        
        # X-axis: lateral movement
        x_base = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Slow body movement
        x_activity = 0.3 * np.sin(2 * np.pi * 1.5 * t)  # Walking/running
        x_noise = np.random.normal(0, 0.05, n_samples)
        x_data = 0.0 + x_base + x_activity + x_noise
        
        # Y-axis: forward/backward movement
        y_base = 0.8 + 0.2 * np.cos(2 * np.pi * 0.08 * t)  # Gravity component
        y_activity = 0.2 * np.cos(2 * np.pi * 1.8 * t)  # Activity patterns
        y_noise = np.random.normal(0, 0.05, n_samples)
        y_data = y_base + y_activity + y_noise
        
        # Z-axis: vertical movement
        z_base = 0.6 + 0.1 * np.sin(2 * np.pi * 0.12 * t)  # Gravity + slow variation
        z_activity = 0.25 * np.sin(2 * np.pi * 2.0 * t)  # Step patterns
        z_noise = np.random.normal(0, 0.05, n_samples)
        z_data = z_base + z_activity + z_noise
        
        # Add occasional high-activity bursts (running, jumping)
        burst_indices = np.random.choice(n_samples, size=n_samples // 1000, replace=False)
        for idx in burst_indices:
            burst_length = np.random.randint(100, 500)
            burst_end = min(idx + burst_length, n_samples)
            x_data[idx:burst_end] += np.random.normal(0, 0.5, burst_end - idx)
            y_data[idx:burst_end] += np.random.normal(0, 0.5, burst_end - idx)
            z_data[idx:burst_end] += np.random.normal(0, 0.5, burst_end - idx)
        
        data = {
            'x': x_data,
            'y': y_data,
            'z': z_data
        }
        
        # Save to CSV for testing (use standard naming so it can be loaded)
        output_file = self.data_dir / f"{participant_id}.csv"
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='10ms'),
            'x': x_data,
            'y': y_data,
            'z': z_data
        })
        df.to_csv(output_file, index=False)
        logger.info(f"Saved synthetic data to {output_file}")
        
        return data


def test_capture24_loader():
    """Test the CAPTURE-24 loader functionality."""
    print("Testing CAPTURE-24 Loader")
    print("=" * 60)
    
    loader = Capture24Loader()
    
    # Generate synthetic data for testing
    print("\n1. Generating synthetic CAPTURE-24 data...")
    synthetic_data = loader.create_synthetic_capture24_data(
        participant_id="P001",
        n_samples=100000,  # 1000 seconds = ~16.7 minutes at 100Hz
        sampling_rate=100.0
    )
    
    print(f"   Generated {len(synthetic_data['x'])} samples")
    print(f"   X-axis range: [{synthetic_data['x'].min():.3f}, {synthetic_data['x'].max():.3f}]")
    print(f"   Y-axis range: [{synthetic_data['y'].min():.3f}, {synthetic_data['y'].max():.3f}]")
    print(f"   Z-axis range: [{synthetic_data['z'].min():.3f}, {synthetic_data['z'].max():.3f}]")
    
    # Test loading
    print("\n2. Testing data loading...")
    try:
        loaded_data = loader.load_participant_data("P001", axes=['x', 'y', 'z'])
        print(f"   Loaded {len(loaded_data['x'])} samples")
    except FileNotFoundError:
        print("   Note: Real CAPTURE-24 data not found (expected for first run)")
    
    # Test segmentation
    print("\n3. Testing data segmentation...")
    segments = loader.segment_data(synthetic_data['x'], window_size=10000, overlap=0)
    print(f"   Created {len(segments)} segments of 10000 samples each")
    print(f"   Total samples: {sum(len(s) for s in segments)}")
    
    # Test with different axes
    print("\n4. Testing magnitude calculation...")
    magnitude_data = loader.load_participant_data("P001", axes=['magnitude'], max_samples=10000)
    print(f"   Magnitude range: [{magnitude_data['magnitude'].min():.3f}, {magnitude_data['magnitude'].max():.3f}]")


if __name__ == "__main__":
    test_capture24_loader()

