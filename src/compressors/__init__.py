"""
Compression Algorithms Module

This module contains implementations of lightweight compression algorithms
for IoT sensor data: delta encoding, run-length encoding, and quantization.
"""

from .delta_encoding import DeltaEncodingCompressor
from .run_length import RunLengthCompressor
from .quantization import QuantizationCompressor

__all__ = [
    'DeltaEncodingCompressor',
    'RunLengthCompressor',
    'QuantizationCompressor'
]

