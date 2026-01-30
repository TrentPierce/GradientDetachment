"""
Approximation Bridging Module

Provides multiple techniques for approximating discrete cryptographic
operations with differentiable functions.
"""

from .bridge import (
    ApproximationBridge,
    SigmoidApproximation,
    StraightThroughEstimator,
    GumbelSoftmaxApproximation,
    TemperatureScheduler
)

from .metrics import (
    ApproximationFidelityMetrics,
    compute_approximation_error,
    compute_gradient_correlation,
    measure_smoothness
)

from .convergence import (
    ConvergenceAnalyzer,
    analyze_approximation_impact,
    plot_convergence_curves
)

__all__ = [
    'ApproximationBridge',
    'SigmoidApproximation',
    'StraightThroughEstimator',
    'GumbelSoftmaxApproximation',
    'TemperatureScheduler',
    'ApproximationFidelityMetrics',
    'compute_approximation_error',
    'compute_gradient_correlation',
    'measure_smoothness',
    'ConvergenceAnalyzer',
    'analyze_approximation_impact',
    'plot_convergence_curves'
]