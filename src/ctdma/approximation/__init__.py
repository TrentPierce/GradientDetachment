"""
Approximation Module for Discrete-to-Continuous Operation Bridging

This module provides multiple approximation techniques for discrete cryptographic
operations, enabling gradient flow while analyzing approximation fidelity.
"""

from .bridge import (
    ApproximationBridge,
    SigmoidApproximation,
    StraightThroughEstimator,
    GumbelSoftmaxApproximation,
    TemperatureAnnealing
)

from .metrics import (
    ApproximationMetrics,
    compute_approximation_error,
    compute_gradient_fidelity,
    compute_information_preservation
)

from .convergence import (
    ConvergenceAnalyzer,
    analyze_approximation_convergence,
    compute_approximation_bias,
    estimate_convergence_rate
)

__all__ = [
    'ApproximationBridge',
    'SigmoidApproximation',
    'StraightThroughEstimator',
    'GumbelSoftmaxApproximation',
    'TemperatureAnnealing',
    'ApproximationMetrics',
    'ConvergenceAnalyzer',
    'compute_approximation_error',
    'compute_gradient_fidelity',
    'compute_information_preservation',
    'analyze_approximation_convergence',
    'compute_approximation_bias',
    'estimate_convergence_rate'
]
