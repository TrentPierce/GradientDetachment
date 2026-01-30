"""
Approximation Methods Module

Provides multiple techniques for creating smooth approximations of discrete
cryptographic operations, enabling gradient flow for analysis.

Modules:
    - bridge: Basic approximation methods (Sigmoid, STE, Gumbel, Temperature)
    - metrics: Quality metrics for approximations
    - convergence: Convergence analysis tools
    - advanced_methods: Advanced techniques (Learnable, Spline, Adaptive, Hybrid)
    - advanced_metrics: Spectral, geometric, and convergence metrics
    - convergence_theory: Theoretical bounds and guarantees

Basic Methods:
    - SigmoidApproximation: Smooth sigmoid-based
    - StraightThroughEstimator: Discrete forward, identity backward
    - GumbelSoftmaxApproximation: Stochastic relaxation
    - TemperatureAnnealing: Annealed approximation

Advanced Methods (New):
    - LearnableApproximation: Neural network-based
    - SplineApproximation: Cubic spline interpolation
    - AdaptiveApproximation: Error-based refinement
    - HybridApproximation: Ensemble of methods

Example:
    >>> from ctdma.approximation import create_approximation_bridge
    >>> approx = create_approximation_bridge('sigmoid', steepness=10.0)
    >>> z = approx.forward(x, y, modulus=2**16)
    
    >>> # Advanced method
    >>> from ctdma.approximation import create_advanced_approximation
    >>> learnable = create_advanced_approximation('learnable', n_bits=16)
    >>> learnable.train_approximation(x_train, y_train, epochs=100)
"""

# Basic approximation methods
from .bridge import (
    ApproximationBridge,
    SigmoidApproximation,
    StraightThroughEstimator,
    GumbelSoftmaxApproximation,
    TemperatureAnnealing,
    ApproximationType,
    create_approximation_bridge
)

# Basic metrics
from .metrics import (
    ApproximationMetrics,
    compare_approximation_methods
)

# Basic convergence analysis
from .convergence import (
    ConvergenceAnalyzer
)

# Advanced methods (new)
from .advanced_methods import (
    LearnableApproximation,
    SplineApproximation,
    AdaptiveApproximation,
    HybridApproximation,
    create_advanced_approximation
)

# Advanced metrics (new)
from .advanced_metrics import (
    SpectralAnalyzer,
    GeometricAnalyzer,
    ConvergenceAnalyzer as AdvancedConvergenceAnalyzer,
    ComprehensiveApproximationAnalyzer
)

# Convergence theory (new)
from .convergence_theory import (
    ConvergenceTheorem,
    UniformConvergenceAnalyzer,
    AsymptoticAnalyzer,
    ProbabilisticConvergenceAnalyzer,
    ConvergenceGuarantees,
    demonstrate_convergence_theory
)

__all__ = [
    # Basic methods
    'ApproximationBridge',
    'SigmoidApproximation',
    'StraightThroughEstimator',
    'GumbelSoftmaxApproximation',
    'TemperatureAnnealing',
    'ApproximationType',
    'create_approximation_bridge',
    
    # Advanced methods (new)
    'LearnableApproximation',
    'SplineApproximation',
    'AdaptiveApproximation',
    'HybridApproximation',
    'create_advanced_approximation',
    
    # Basic metrics
    'ApproximationMetrics',
    'compare_approximation_methods',
    
    # Basic convergence
    'ConvergenceAnalyzer',
    
    # Advanced metrics (new)
    'SpectralAnalyzer',
    'GeometricAnalyzer',
    'AdvancedConvergenceAnalyzer',
    'ComprehensiveApproximationAnalyzer',
    
    # Convergence theory (new)
    'ConvergenceTheorem',
    'UniformConvergenceAnalyzer',
    'AsymptoticAnalyzer',
    'ProbabilisticConvergenceAnalyzer',
    'ConvergenceGuarantees',
    'demonstrate_convergence_theory'
]

__version__ = '1.0.0'
