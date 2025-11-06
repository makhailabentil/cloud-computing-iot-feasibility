"""
Evaluation Module

This module provides systematic evaluation framework for compression algorithms,
measuring compression ratio, reconstruction error, and resource consumption.
"""

from .systematic_evaluation import SystematicEvaluator, CompressionMetrics

__all__ = [
    'SystematicEvaluator',
    'CompressionMetrics'
]

