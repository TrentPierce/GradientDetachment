"""
Theoretical Analysis Module

Provides mathematical foundations and formal proofs for gradient inversion
phenomena in ARX ciphers.
"""

from .mathematical_analysis import (
    SawtoothTopologyAnalyzer,
    GradientInversionAnalyzer,
    InformationTheoreticAnalyzer,
    ARXMathematicalFramework
)

from .theorems import (
    GradientInversionTheorem,
    SawtoothConvergenceTheorem,
    InformationLeakageTheorem,
    ModularArithmeticLemma,
    prove_all_theorems
)

__all__ = [
    'SawtoothTopologyAnalyzer',
    'GradientInversionAnalyzer',
    'InformationTheoreticAnalyzer',
    'ARXMathematicalFramework',
    'GradientInversionTheorem',
    'SawtoothConvergenceTheorem',
    'InformationLeakageTheorem',
    'ModularArithmeticLemma',
    'prove_all_theorems'
]