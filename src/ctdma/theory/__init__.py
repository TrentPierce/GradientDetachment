"""
Mathematical Theory Module for Gradient Inversion Analysis

This module provides rigorous mathematical foundations for understanding
gradient inversion in ARX ciphers, including:

1. Formal theorem statements and proofs
2. Topological analysis of loss landscapes
3. Information-theoretic bounds
4. Convergence and stability analysis

Mathematical Framework:
    - Gradient discontinuity theorems
    - Sawtooth topology characterization
    - Lyapunov stability analysis
    - Shannon information theory
    - Channel capacity analysis

Usage:
    from ctdma.theory import GradientInversionAnalyzer
    from ctdma.theory import Theorem1_GradientDiscontinuity
    from ctdma.theory import InformationLossTheorem
"""

# Original mathematical analysis components
from .mathematical_analysis import (
    GradientInversionAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer,
)

# Original theorem implementations
from .theorems import (
    ModularAdditionTheorem,
    GradientInversionTheorem,
    SawtoothConvergenceTheorem,
    InformationLossTheorem,
)

# New formal proof components
from .formal_proofs import (
    FormalProof,
    Theorem1_GradientDiscontinuity,
    Theorem2_SystematicInversion,
)

# Topology analysis
from .topology_analysis import (
    TopologicalInvariant,
    SawtoothTopologyTheorem,
    LyapunovStabilityAnalysis,
)

# Information theory
from .information_theory import (
    InformationMetrics,
    InformationLossTheorem as InfoLossTheoremRigorous,
    GradientChannelAnalysis,
    EntropyProductionAnalysis,
)

# Convergence analysis
from .convergence_proofs import (
    ConvergenceTheorem,
    LyapunovStabilityAnalysis as LyapunovAnalyzer,
    FixedPointTheorem,
    ConvergenceRateTheorem,
    prove_convergence_failure_in_sawtooth,
)

# Utility functions
from .theorems import (
    verify_gradient_inversion_conditions,
    prove_adversarial_attractor_existence,
)

__all__ = [
    # Original analyzers
    'GradientInversionAnalyzer',
    'SawtoothTopologyAnalyzer',
    'InformationTheoreticAnalyzer',
    
    # Original theorems
    'ModularAdditionTheorem',
    'GradientInversionTheorem',
    'SawtoothConvergenceTheorem',
    'InformationLossTheorem',
    
    # Formal proofs
    'FormalProof',
    'Theorem1_GradientDiscontinuity',
    'Theorem2_SystematicInversion',
    
    # Topology
    'TopologicalInvariant',
    'SawtoothTopologyTheorem',
    'LyapunovStabilityAnalysis',
    
    # Information theory
    'InformationMetrics',
    'InfoLossTheoremRigorous',
    'GradientChannelAnalysis',
    'EntropyProductionAnalysis',
    
    # Convergence
    'ConvergenceTheorem',
    'LyapunovAnalyzer',
    'FixedPointTheorem',
    'ConvergenceRateTheorem',
    'prove_convergence_failure_in_sawtooth',
    
    # Utilities
    'verify_gradient_inversion_conditions',
    'prove_adversarial_attractor_existence',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Trent Pierce'
