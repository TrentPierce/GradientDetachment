"""
Mathematical Theory Module for Gradient Inversion Analysis

This module provides rigorous mathematical foundations for understanding
gradient inversion in ARX ciphers, including:

1. Formal theorem statements with complete proofs
2. Topological analysis of loss landscapes  
3. Information-theoretic bounds
4. Convergence analysis using dynamical systems theory

Submodules:
===========
- formal_proofs: Complete theorem statements and detailed proofs
- topology_theory: Sawtooth topology and basin analysis
- information_theory: Shannon entropy and channel capacity
- convergence_theory: Lyapunov analysis and fixed point theory
- mathematical_analysis: Numerical analysis tools
- theorems: Theorem implementations and verification

Usage:
======
from ctdma.theory import GradientInversionTheorems, InformationTheoreticAnalysis
from ctdma.theory import ProofCompendium, SawtoothTopologyTheory

# Get formal proof
proof = ProofCompendium.proof_1_gradient_discontinuity()
proof.display()

# Verify theorem numerically
from ctdma.theory import verify_all_theorems
results = verify_all_theorems()
"""

# Core analyzers
from .mathematical_analysis import (
    GradientInversionAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer
)

# Theorem implementations
from .theorems import (
    ModularAdditionTheorem,
    GradientInversionTheorem,
    SawtoothConvergenceTheorem,
    InformationLossTheorem
)

# Formal proofs (new)
try:
    from .formal_proofs import (
        GradientInversionTheorems,
        FormalTheorem,
        CompleteProof,
        ProofCompendium
    )
except ImportError:
    # Backward compatibility
    pass

# Topology theory (new)
try:
    from .topology_theory import (
        SawtoothTopologyTheory,
        TopologicalTheorem
    )
except ImportError:
    pass

# Information theory (new)
try:
    from .information_theory import (
        InformationTheoreticAnalysis as InfoTheory,
        InformationTheorem
    )
except ImportError:
    pass

# Convergence theory (new)
try:
    from .convergence_theory import (
        ConvergenceTheory,
        BasinOfAttractionAnalysis,
        visualize_sawtooth_topology
    )
except ImportError:
    pass

# Export all
__all__ = [
    # Core analyzers
    'GradientInversionAnalyzer',
    'SawtoothTopologyAnalyzer',
    'InformationTheoreticAnalyzer',
    
    # Original theorems
    'ModularAdditionTheorem',
    'GradientInversionTheorem',
    'SawtoothConvergenceTheorem',
    'InformationLossTheorem',
    
    # New formal proofs
    'GradientInversionTheorems',
    'FormalTheorem',
    'CompleteProof',
    'ProofCompendium',
    
    # New topology theory
    'SawtoothTopologyTheory',
    'TopologicalTheorem',
    
    # New information theory
    'InfoTheory',
    'InformationTheorem',
    
    # New convergence theory
    'ConvergenceTheory',
    'BasinOfAttractionAnalysis',
    'visualize_sawtooth_topology'
]
