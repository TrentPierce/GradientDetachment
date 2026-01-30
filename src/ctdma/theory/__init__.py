"""
Mathematical Theory Module for Gradient Inversion Analysis

This module provides rigorous mathematical analysis of the gradient inversion
phenomenon in ARX ciphers when attacked with Neural ODEs.
"""

from .mathematical_analysis import (
    GradientInversionAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer,
    compute_gradient_discontinuity,
    analyze_loss_landscape,
    theoretical_inversion_probability
)

from .theorems import (
    ModularAdditionTheorem,
    GradientInversionTheorem,
    SawtoothConvergenceTheorem,
    InformationLossTheorem,
    verify_gradient_inversion_conditions,
    prove_adversarial_attractor_existence
)

__all__ = [
    'GradientInversionAnalyzer',
    'SawtoothTopologyAnalyzer',
    'InformationTheoreticAnalyzer',
    'ModularAdditionTheorem',
    'GradientInversionTheorem',
    'SawtoothConvergenceTheorem',
    'InformationLossTheorem',
    'compute_gradient_discontinuity',
    'analyze_loss_landscape',
    'theoretical_inversion_probability',
    'verify_gradient_inversion_conditions',
    'prove_adversarial_attractor_existence'
]
