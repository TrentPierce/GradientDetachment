"""
Theory Module for GradientDetachment

This module provides rigorous mathematical analysis of the gradient inversion
phenomenon in ARX ciphers, including formal theorems, proofs, and information-
theoretic analysis.
"""

from .mathematical_analysis import (
    ARXGradientAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer,
    compute_gradient_norm,
    compute_hessian_eigenvalues,
    analyze_loss_landscape_curvature
)

from .theorems import (
    GradientInversionTheorem,
    SawtoothLandscapeTheorem,
    InformationBottleneckTheorem,
    CriticalPointTheorem,
    verify_all_theorems
)

__all__ = [
    'ARXGradientAnalyzer',
    'SawtoothTopologyAnalyzer',
    'InformationTheoreticAnalyzer',
    'GradientInversionTheorem',
    'SawtoothLandscapeTheorem',
    'InformationBottleneckTheorem',
    'CriticalPointTheorem',
    'compute_gradient_norm',
    'compute_hessian_eigenvalues',
    'analyze_loss_landscape_curvature',
    'verify_all_theorems'
]
