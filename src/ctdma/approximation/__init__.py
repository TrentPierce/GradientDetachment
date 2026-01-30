"""
Approximation Bridging Module

This module provides various approximation techniques for discrete cryptographic
operations, enabling gradient-based analysis while maintaining fidelity to the
original discrete functions.
"""

from .bridge import (
    ApproximationBridge,
    SigmoidApproximation,
    StraightThroughEstimator,
    GumbelSoftmaxApproximation,
    TemperatureSmoothing
)

from .metrics import (
    ApproximationMetrics,
    compute_fidelity,
    compute_gradient_similarity,
    compute_discrete_error
)

from .convergence import (
    ConvergenceAnalyzer,
    analyze_temperature_schedule,
    measure_approximation_quality
)

__all__ = [
    'ApproximationBridge',
    'SigmoidApproximation',
    'StraightThroughEstimator',
    'GumbelSoftmaxApproximation',
    'TemperatureSmoothing',
    'ApproximationMetrics',
    'compute_fidelity',
    'compute_gradient_similarity',
    'compute_discrete_error',
    'ConvergenceAnalyzer',
    'analyze_temperature_schedule',
    'measure_approximation_quality'
]
