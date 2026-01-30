"""
Approximation Bridging Module

Provides multiple techniques for approximating discrete cryptographic operations
with differentiable functions, enabling gradient-based analysis.
"""

from .bridge import (
    ApproximationBridge,
    SigmoidApproximation,
    StraightThroughEstimator,
    GumbelSoftmaxApproximation,
    TemperatureBasedSmoothing,
    create_approximator
)

from .metrics import (
    ApproximationMetrics,
    compute_fidelity,
    compute_gradient_bias,
    compute_convergence_rate
)

from .convergence import (
    ConvergenceAnalyzer,
    analyze_convergence_properties,
    plot_convergence_curves
)

__all__ = [
    'ApproximationBridge',
    'SigmoidApproximation',
    'StraightThroughEstimator',
    'GumbelSoftmaxApproximation',
    'TemperatureBasedSmoothing',
    'create_approximator',
    'ApproximationMetrics',
    'compute_fidelity',
    'compute_gradient_bias',
    'compute_convergence_rate',
    'ConvergenceAnalyzer',
    'analyze_convergence_properties',
    'plot_convergence_curves'
]