"""
CSV Sensor Trace Replay Module

This module provides functionality to replay CSV sensor traces for testing
compression algorithms and evaluating performance on real IoT data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorTraceReplayer:
    """
    Replays CSV sensor traces for compression testing and evaluation.
    
    This class handles loading, preprocessing, and replaying sensor data
    from CSV files, which is essential for testing compression algorithms
    on real IoT data patterns.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the sensor trace replayer.
        
        Args:
            data_dir: Directory containing sensor data files
        """
        self.data_dir = Path(data_dir)
        self.traces = {}
        self.metadata = {}
        
    def load_csv_trace(self, filename: str, 
                      timestamp_col: str = 'timestamp',
                      value_cols: List[str] = None,
                      delimiter: str = ',') -> pd.DataFrame:
        """
        Load a CSV sensor trace file.
        
        Args:
            filename: Name of the CSV file to load
            timestamp_col: Name of the timestamp column
            value_cols: List of value column names (if None, auto-detect)
            delimiter: CSV delimiter
            
        Returns:
            Loaded DataFrame with sensor data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Load the CSV file
        df = pd.read_csv(filepath, delimiter=delimiter)
        
        # Auto-detect value columns if not specified
        if value_cols is None:
            value_cols = [col for col in df.columns if col != timestamp_col]
        
        # Ensure timestamp column exists
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV")
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Store trace and metadata
        trace_id = filename.replace('.csv', '')
        self.traces[trace_id] = df
        self.metadata[trace_id] = {
            'filename': filename,
            'timestamp_col': timestamp_col,
            'value_cols': value_cols,
            'n_points': len(df),
            'time_range': (df[timestamp_col].min(), df[timestamp_col].max()),
            'sampling_rate': self._estimate_sampling_rate(df[timestamp_col])
        }
        
        logger.info(f"Loaded trace '{trace_id}' with {len(df)} points")
        logger.info(f"Time range: {self.metadata[trace_id]['time_range']}")
        logger.info(f"Estimated sampling rate: {self.metadata[trace_id]['sampling_rate']:.2f} Hz")
        
        return df
    
    def _estimate_sampling_rate(self, timestamps: pd.Series) -> float:
        """Estimate the sampling rate from timestamps."""
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate time differences
        time_diffs = timestamps.diff().dropna()
        
        # Use median to avoid outliers
        median_interval = time_diffs.median()
        
        # Convert to Hz
        sampling_rate = 1.0 / median_interval.total_seconds()
        
        return sampling_rate
    
    def get_trace_data(self, trace_id: str, 
                      value_col: str = None,
                      start_time: Optional[pd.Timestamp] = None,
                      end_time: Optional[pd.Timestamp] = None) -> np.ndarray:
        """
        Extract time series data from a loaded trace.
        
        Args:
            trace_id: ID of the trace to extract data from
            value_col: Specific value column to extract (if None, use first)
            start_time: Start time for data extraction
            end_time: End time for data extraction
            
        Returns:
            Time series data as numpy array
        """
        if trace_id not in self.traces:
            raise ValueError(f"Trace '{trace_id}' not found")
        
        df = self.traces[trace_id]
        metadata = self.metadata[trace_id]
        
        # Select value column
        if value_col is None:
            value_col = metadata['value_cols'][0]
        
        if value_col not in metadata['value_cols']:
            raise ValueError(f"Value column '{value_col}' not found in trace")
        
        # Filter by time range if specified
        if start_time is not None or end_time is not None:
            timestamp_col = metadata['timestamp_col']
            mask = pd.Series(True, index=df.index)
            
            if start_time is not None:
                mask &= df[timestamp_col] >= start_time
            if end_time is not None:
                mask &= df[timestamp_col] <= end_time
            
            df = df[mask]
        
        return df[value_col].values
    
    def replay_trace(self, trace_id: str, 
                    value_col: str = None,
                    chunk_size: int = 1000,
                    start_time: Optional[pd.Timestamp] = None,
                    end_time: Optional[pd.Timestamp] = None) -> List[np.ndarray]:
        """
        Replay a sensor trace in chunks for streaming simulation.
        
        Args:
            trace_id: ID of the trace to replay
            value_col: Specific value column to replay
            chunk_size: Size of each chunk for streaming
            start_time: Start time for replay
            end_time: End time for replay
            
        Yields:
            Chunks of time series data
        """
        data = self.get_trace_data(trace_id, value_col, start_time, end_time)
        
        # Split data into chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            yield chunk
    
    def generate_synthetic_trace(self, n_points: int = 10000,
                               sampling_rate: float = 1.0,
                               signal_type: str = 'sensor') -> pd.DataFrame:
        """
        Generate a synthetic sensor trace for testing.
        
        Args:
            n_points: Number of data points to generate
            sampling_rate: Sampling rate in Hz
            signal_type: Type of signal ('sensor', 'accelerometer', 'temperature')
            
        Returns:
            Synthetic sensor trace DataFrame
        """
        # Generate timestamps
        start_time = pd.Timestamp.now() - pd.Timedelta(seconds=n_points/sampling_rate)
        timestamps = pd.date_range(start=start_time, periods=n_points, freq=f'{1/sampling_rate}S')
        
        # Generate signal based on type
        if signal_type == 'sensor':
            # Generic sensor data with trend and noise
            t = np.linspace(0, n_points/sampling_rate, n_points)
            signal = 20 + 5 * np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 0.5, n_points)
        elif signal_type == 'accelerometer':
            # Accelerometer-like data with more dynamic patterns
            t = np.linspace(0, n_points/sampling_rate, n_points)
            signal = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.1, n_points)
        elif signal_type == 'temperature':
            # Temperature-like data with slow variations
            t = np.linspace(0, n_points/sampling_rate, n_points)
            signal = 25 + 3 * np.sin(2 * np.pi * 0.01 * t) + 0.1 * t + np.random.normal(0, 0.2, n_points)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': signal
        })
        
        # Store as synthetic trace
        trace_id = f'synthetic_{signal_type}_{n_points}'
        self.traces[trace_id] = df
        self.metadata[trace_id] = {
            'filename': f'synthetic_{signal_type}.csv',
            'timestamp_col': 'timestamp',
            'value_cols': ['value'],
            'n_points': len(df),
            'time_range': (timestamps[0], timestamps[-1]),
            'sampling_rate': sampling_rate,
            'signal_type': signal_type
        }
        
        logger.info(f"Generated synthetic trace '{trace_id}' with {len(df)} points")
        
        return df
    
    def save_trace(self, trace_id: str, filename: str = None) -> str:
        """
        Save a trace to CSV file.
        
        Args:
            trace_id: ID of the trace to save
            filename: Output filename (if None, use trace_id)
            
        Returns:
            Path to saved file
        """
        if trace_id not in self.traces:
            raise ValueError(f"Trace '{trace_id}' not found")
        
        if filename is None:
            filename = f"{trace_id}.csv"
        
        filepath = self.data_dir / filename
        self.traces[trace_id].to_csv(filepath, index=False)
        
        logger.info(f"Saved trace '{trace_id}' to {filepath}")
        
        return str(filepath)
    
    def list_traces(self) -> List[str]:
        """List all loaded traces."""
        return list(self.traces.keys())
    
    def get_trace_info(self, trace_id: str) -> Dict:
        """Get information about a specific trace."""
        if trace_id not in self.traces:
            raise ValueError(f"Trace '{trace_id}' not found")
        
        return self.metadata[trace_id]


def test_trace_replayer():
    """
    Test the sensor trace replayer functionality.
    """
    print("Testing Sensor Trace Replayer")
    print("=" * 40)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Initialize replayer
    replayer = SensorTraceReplayer()
    
    # Generate synthetic traces
    print("Generating synthetic traces...")
    sensor_trace = replayer.generate_synthetic_trace(n_points=1000, signal_type='sensor')
    accel_trace = replayer.generate_synthetic_trace(n_points=2000, signal_type='accelerometer')
    temp_trace = replayer.generate_synthetic_trace(n_points=5000, signal_type='temperature')
    
    # List all traces
    print(f"\nAvailable traces: {replayer.list_traces()}")
    
    # Test trace replay
    print("\nTesting trace replay...")
    for trace_id in replayer.list_traces():
        print(f"\nReplaying trace '{trace_id}':")
        info = replayer.get_trace_info(trace_id)
        print(f"  Points: {info['n_points']}")
        print(f"  Sampling rate: {info['sampling_rate']:.2f} Hz")
        
        # Replay in chunks
        chunk_count = 0
        for chunk in replayer.replay_trace(trace_id, chunk_size=500):
            chunk_count += 1
            if chunk_count <= 3:  # Show first 3 chunks
                print(f"  Chunk {chunk_count}: {len(chunk)} points, range: [{chunk.min():.2f}, {chunk.max():.2f}]")
        
        print(f"  Total chunks: {chunk_count}")


if __name__ == "__main__":
    test_trace_replayer()
