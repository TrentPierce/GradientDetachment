"""
Mathematical Theory Module for Gradient Inversion Analysis

This module provides rigorous mathematical foundations for understanding
the gradient inversion phenomenon in ARX ciphers.
"""

from .mathematical_analysis import (
    ARXGradientAnalysis,
    SawtoothTopology,
    InformationTheoreticAnalysis
)

from .theorems import (
    GradientInversionTheorem,
    SawtoothConvergenceTheorem,
    EntropyBoundTheorem
)

__all__ = [
    'ARXGradientAnalysis',
    'SawtoothTopology',
    'InformationTheoreticAnalysis',
    'GradientInversionTheorem',
    'SawtoothConvergenceTheorem',
    'EntropyBoundTheorem'
]
