"""
Theoretical Analysis Module for Gradient Inversion in ARX Ciphers

This module provides formal mathematical analysis of why ARX operations
create gradient inversion phenomena in Neural ODE-based cryptanalysis.
"""

from .mathematical_analysis import (
    GradientInversionAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer,
    analyze_gradient_flow,
    compute_lipschitz_constant,
    measure_gradient_variance
)

from .theorems import (
    Theorems,
    verify_sawtooth_theorem,
    verify_gradient_inversion_theorem,
    verify_convergence_impossibility_theorem
)

__all__ = [
    'GradientInversionAnalyzer',
    'SawtoothTopologyAnalyzer',
    'InformationTheoreticAnalyzer',
    'analyze_gradient_flow',
    'compute_lipschitz_constant',
    'measure_gradient_variance',
    'Theorems',
    'verify_sawtooth_theorem',
    'verify_gradient_inversion_theorem',
    'verify_convergence_impossibility_theorem'
]