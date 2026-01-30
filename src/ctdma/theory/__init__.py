"""
Mathematical Theory Module for Gradient Inversion Analysis

This module provides rigorous mathematical foundations for understanding
gradient inversion in ARX ciphers.

Modules:
    - mathematical_analysis: Core analysis tools
    - theorems: Formal theorem implementations
    - formal_proofs: Rigorous mathematical proofs
    - topology_theory: Topological analysis and Morse theory

Key Classes:
    - GradientInversionAnalyzer: Analyze gradient discontinuities
    - SawtoothTopologyAnalyzer: Study loss landscape topology
    - InformationTheoreticAnalyzer: Information-theoretic analysis
    - CompositeFormalProof: Unified proof framework
    - TopologicalAnalyzer: Complete topological characterization

Example:
    >>> from ctdma.theory import GradientInversionTheorem
    >>> theorem = GradientInversionTheorem()
    >>> results = theorem.validate(x, y)
    >>> print(f"Inversion rate: {results['inversion_rate']:.2%}")
"""

# Core analyzers
from .mathematical_analysis import (
    GradientInversionAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer,
    compute_gradient_discontinuity,
    analyze_loss_landscape,
    theoretical_inversion_probability
)

# Theorem classes
from .theorems import (
    ModularAdditionTheorem,
    GradientInversionTheorem as TheoremGradientInversion,
    SawtoothConvergenceTheorem,
    InformationLossTheorem
)

# Formal proofs (new)
from .formal_proofs import (
    GradientInversionTheorem,
    SawtoothTopologyTheorem,
    InformationTheoreticTheorem,
    CompositeFormalProof,
    FormalTheorem,
    print_theorem,
    print_proof
)

# Topology theory (new)
from .topology_theory import (
    TopologicalAnalyzer,
    SawtoothManifold,
    GradientFlowAnalyzer,
    CriticalPointTheory,
    StructuralStabilityAnalyzer,
    print_topology_summary
)

__all__ = [
    # Core analyzers
    'GradientInversionAnalyzer',
    'SawtoothTopologyAnalyzer',
    'InformationTheoreticAnalyzer',
    
    # Theorem classes (original)
    'ModularAdditionTheorem',
    'TheoremGradientInversion',
    'SawtoothConvergenceTheorem',
    'InformationLossTheorem',
    
    # Formal proofs (new)
    'GradientInversionTheorem',
    'SawtoothTopologyTheorem',
    'InformationTheoreticTheorem',
    'CompositeFormalProof',
    'FormalTheorem',
    'print_theorem',
    'print_proof',
    
    # Topology theory (new)
    'TopologicalAnalyzer',
    'SawtoothManifold',
    'GradientFlowAnalyzer',
    'CriticalPointTheory',
    'StructuralStabilityAnalyzer',
    'print_topology_summary',
    
    # Utility functions
    'compute_gradient_discontinuity',
    'analyze_loss_landscape',
    'theoretical_inversion_probability'
]

__version__ = '1.0.0'
