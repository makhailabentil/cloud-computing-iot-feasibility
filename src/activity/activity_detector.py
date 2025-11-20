"""
Activity Detection Module for CAPTURE-24 Data

This module implements lightweight activity classification from accelerometer
signal characteristics, enabling activity-aware adaptive compression.

For Milestone 3: Activity-Aware Adaptive Compression
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Activity types for compression algorithm selection."""
    SLEEP = "sleep"
    REST = "rest"  # Sedentary, sitting
    WALKING = "walking"
    ACTIVE = "active"  # Running, jumping, high-intensity
    MIXED = "mixed"  # Transition periods or unclear


class ActivityDetector:
    """
    Lightweight activity detector for accelerometer data.
    
    Uses signal characteristics (variance, entropy, frequency analysis) to
    classify activities without requiring full activity annotations.
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the activity detector.
        
        Args:
            sampling_rate: Sampling rate in Hz (default 100Hz for CAPTURE-24)
        """
        self.sampling_rate = sampling_rate
        self.window_size = int(sampling_rate * 10)  # 10-second windows
        
        # Activity thresholds (tuned for CAPTURE-24 data)
        # Adjusted to better detect rest/sleep periods
        self.thresholds = {
            'variance_low': 0.05,      # Low variance = rest/sleep (relaxed from 0.01)
            'variance_high': 0.5,      # High variance = active
            'entropy_low': 1.5,        # Low entropy = repetitive (sleep/rest) (relaxed from 1.0)
            'entropy_high': 3.0,       # High entropy = varied (active)
            'magnitude_mean_low': 0.8, # Low magnitude = rest
            'magnitude_mean_high': 1.5, # High magnitude = active
            'variance_sleep': 0.02     # Very low variance = sleep
        }
    
    def calculate_variance(self, data: np.ndarray) -> float:
        """Calculate variance of the signal."""
        return np.var(data)
    
    def calculate_entropy(self, data: np.ndarray, bins: int = 50) -> float:
        """
        Calculate Shannon entropy of the signal distribution.
        
        Higher entropy indicates more variation in the signal.
        """
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]  # Remove zeros
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def calculate_magnitude_stats(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
        """Calculate magnitude statistics from triaxial data."""
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        return {
            'mean': np.mean(magnitude),
            'std': np.std(magnitude),
            'max': np.max(magnitude),
            'min': np.min(magnitude)
        }
    
    def calculate_frequency_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate frequency domain features using FFT.
        
        Returns dominant frequency and spectral energy distribution.
        """
        # Apply FFT
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1.0 / self.sampling_rate)
        
        # Power spectral density
        power = np.abs(fft) ** 2
        
        # Find dominant frequency (excluding DC component)
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        positive_power[0] = 0  # Remove DC
        
        if len(positive_power) > 0:
            dominant_idx = np.argmax(positive_power)
            dominant_freq = positive_freqs[dominant_idx]
        else:
            dominant_freq = 0.0
        
        # Spectral energy in different bands
        low_band = np.sum(positive_power[(positive_freqs >= 0) & (positive_freqs < 2)])
        mid_band = np.sum(positive_power[(positive_freqs >= 2) & (positive_freqs < 5)])
        high_band = np.sum(positive_power[positive_freqs >= 5])
        total_energy = low_band + mid_band + high_band
        
        return {
            'dominant_freq': abs(dominant_freq),
            'low_band_ratio': low_band / total_energy if total_energy > 0 else 0,
            'mid_band_ratio': mid_band / total_energy if total_energy > 0 else 0,
            'high_band_ratio': high_band / total_energy if total_energy > 0 else 0
        }
    
    def detect_activity(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> ActivityType:
        """
        Detect activity type from triaxial accelerometer data.
        
        Args:
            x: X-axis accelerometer data
            y: Y-axis accelerometer data
            z: Z-axis accelerometer data
            
        Returns:
            Detected ActivityType
        """
        # Ensure all arrays have same length
        min_len = min(len(x), len(y), len(z))
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        
        # Calculate features
        variance_x = self.calculate_variance(x)
        variance_y = self.calculate_variance(y)
        variance_z = self.calculate_variance(z)
        avg_variance = (variance_x + variance_y + variance_z) / 3.0
        
        entropy_x = self.calculate_entropy(x)
        entropy_y = self.calculate_entropy(y)
        entropy_z = self.calculate_entropy(z)
        avg_entropy = (entropy_x + entropy_y + entropy_z) / 3.0
        
        magnitude_stats = self.calculate_magnitude_stats(x, y, z)
        
        # Frequency features (use magnitude for frequency analysis)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        freq_features = self.calculate_frequency_features(magnitude)
        
        # Classification logic (order matters - check most specific first)
        # Sleep: Very low variance, low entropy, low magnitude
        if (avg_variance < self.thresholds.get('variance_sleep', 0.02) and
            avg_entropy < self.thresholds['entropy_low'] and
            magnitude_stats['mean'] < self.thresholds['magnitude_mean_low']):
            return ActivityType.SLEEP
        
        # Rest/Sedentary: Low variance, low entropy, moderate magnitude
        # More specific than before - must be low variance AND low entropy
        if (avg_variance < self.thresholds['variance_low'] and
            avg_entropy < self.thresholds['entropy_low'] and
            magnitude_stats['mean'] < self.thresholds['magnitude_mean_high']):
            return ActivityType.REST
        
        # Walking: Moderate variance, moderate entropy, moderate-high magnitude
        # Walking typically has 1-3 Hz dominant frequency
        if (self.thresholds['variance_low'] < avg_variance < self.thresholds['variance_high'] and
            1.0 < freq_features['dominant_freq'] < 3.5 and
            magnitude_stats['mean'] > self.thresholds['magnitude_mean_low']):
            return ActivityType.WALKING
        
        # Active: High variance, high entropy, high magnitude
        if (avg_variance > self.thresholds['variance_high'] or
            avg_entropy > self.thresholds['entropy_high'] or
            magnitude_stats['mean'] > self.thresholds['magnitude_mean_high']):
            return ActivityType.ACTIVE
        
        # Default to mixed if unclear
        return ActivityType.MIXED
    
    def detect_activity_from_single_axis(self, data: np.ndarray) -> ActivityType:
        """
        Detect activity from a single axis (simpler, less accurate).
        
        Useful when only one axis is available.
        """
        variance = self.calculate_variance(data)
        entropy = self.calculate_entropy(data)
        
        if variance < self.thresholds['variance_low'] and entropy < self.thresholds['entropy_low']:
            return ActivityType.SLEEP
        elif variance < self.thresholds['variance_high'] and entropy < self.thresholds['entropy_high']:
            return ActivityType.REST
        elif variance > self.thresholds['variance_high'] or entropy > self.thresholds['entropy_high']:
            return ActivityType.ACTIVE
        else:
            return ActivityType.MIXED
    
    def get_activity_features(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """
        Get all activity features for analysis.
        
        Returns dictionary of all calculated features.
        """
        min_len = min(len(x), len(y), len(z))
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        
        features = {
            'variance_x': self.calculate_variance(x),
            'variance_y': self.calculate_variance(y),
            'variance_z': self.calculate_variance(z),
            'entropy_x': self.calculate_entropy(x),
            'entropy_y': self.calculate_entropy(y),
            'entropy_z': self.calculate_entropy(z),
        }
        
        magnitude_stats = self.calculate_magnitude_stats(x, y, z)
        features.update(magnitude_stats)
        
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        freq_features = self.calculate_frequency_features(magnitude)
        features.update(freq_features)
        
        return features


def test_activity_detector():
    """Test the activity detector with synthetic data."""
    detector = ActivityDetector()
    
    # Generate synthetic data for different activities
    t = np.linspace(0, 10, 1000)  # 10 seconds at 100Hz
    
    # Sleep: very low variation
    sleep_x = np.random.normal(0, 0.01, 1000)
    sleep_y = np.random.normal(0.8, 0.01, 1000)
    sleep_z = np.random.normal(0.6, 0.01, 1000)
    
    # Walking: periodic movement
    walk_x = 0.2 * np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 0.05, 1000)
    walk_y = 0.8 + 0.3 * np.cos(2 * np.pi * 2.0 * t) + np.random.normal(0, 0.05, 1000)
    walk_z = 0.6 + 0.2 * np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 0.05, 1000)
    
    # Active: high variation
    active_x = 0.5 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 0.3, 1000)
    active_y = 0.8 + 0.5 * np.cos(2 * np.pi * 5.0 * t) + np.random.normal(0, 0.3, 1000)
    active_z = 0.6 + 0.4 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 0.3, 1000)
    
    print("Activity Detection Test")
    print("=" * 60)
    print(f"Sleep detection: {detector.detect_activity(sleep_x, sleep_y, sleep_z)}")
    print(f"Walking detection: {detector.detect_activity(walk_x, walk_y, walk_z)}")
    print(f"Active detection: {detector.detect_activity(active_x, active_y, active_z)}")


if __name__ == "__main__":
    test_activity_detector()

